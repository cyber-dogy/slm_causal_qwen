"""
Qwen2-VL SLM 缺陷检测 - 训练脚本 (LoRA/QLoRA) - 适配重构数据集

该脚本实现使用 LoRA 或 QLoRA 对 Qwen2-VL-2B-Instruct 进行参数高效微调，
用于 SLM 缺陷检测任务。

适配特性:
- 支持重构版数据集 (二分类/三分类/joint 模式)
- 支持 letters/words token 风格
- 兼容新旧 manifest 格式

安全保障:
- 永远不会覆盖基础模型权重
- 只保存 LoRA 适配器、配置和日志
- 验证输出目录不与基础模型路径冲突
"""

import os
import sys
import json
import logging
import argparse
import shutil
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Optional: Weights & Biases logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# 检查版本并提供回退方案
try:
    import transformers
    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2VLProcessor,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError as e:
    print(f"导入 transformers 出错: {e}")
    print("请安装: pip install transformers>=4.40.0")
    sys.exit(1)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("警告: peft 不可用。LoRA/QLoRA 将无法工作。")

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

# Weights & Biases availability check (already defined above, but keep for clarity)
# WANDB_AVAILABLE is defined at the top of the file

# 导入本地模块 - 使用重构版数据集
from dataset_qwen2vl_slm import (
    SLMDataset,
    get_data_collator,
    get_class_weights,
    CLASS_NAMES_3,  # 三分类类别
    CLASS_NAMES_2,  # 二分类类别
    CLASS_NAME_TO_ID_3,
    CLASS_NAME_TO_ID_2,
    build_target_text,
    choose_prompt,
)
from utils_metrics import (
    compute_classification_metrics,
    normalize_prediction,
    save_metrics,
    print_metrics_table,
)
from evaluate_qwen2vl_slm import parse_prediction
from evaluate_qwen2vl_slm import evaluate_sample_with_candidate_scores, resolve_prediction_mode


# ============================================================================
# 日志设置
# ============================================================================

def setup_logging(log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """设置文件和控制台日志。"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("qwen2vl_slm")
    logger.setLevel(level)
    
    # 清除已有处理器
    logger.handlers = []
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'train.log'),
        mode='w'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# 安全检查
# ============================================================================

def validate_output_directory(output_dir: str, base_model_path: str, overwrite: bool = False) -> bool:
    """
    验证输出目录是否可安全使用。
    
    参数:
        output_dir: 期望的输出目录
        base_model_path: 基础模型路径/名称
        overwrite: 是否允许覆盖已有输出
        
    返回:
        有效返回 True，否则抛出异常
    """
    # 转换为绝对路径
    output_dir = os.path.abspath(output_dir)
    
    # 如果是本地路径则解析基础模型路径
    if os.path.exists(base_model_path):
        base_model_path = os.path.abspath(base_model_path)
        
        # 关键检查: output_dir 不能与 base_model_path 相同或在其内部
        if output_dir == base_model_path or output_dir.startswith(base_model_path + os.sep):
            raise ValueError(
                f"严重错误: 输出目录 ({output_dir}) 不能与基础模型路径 ({base_model_path}) "
                f"相同或在其内部。这会覆盖原始模型!"
            )
    
    # 检查输出目录是否存在
    if os.path.exists(output_dir):
        if os.listdir(output_dir) and not overwrite:
            raise ValueError(
                f"输出目录 {output_dir} 存在且非空。"
                f"使用 --overwrite_output_dir 强制覆盖。"
            )
        elif overwrite:
            warnings.warn(f"覆盖已有输出目录: {output_dir}")
    
    return True


def str2bool(value: Any) -> bool:
    """可靠地解析命令行布尔值。"""
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_processor(
    model_name_or_path: str,
    finetune_mode: str = "qlora",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bf16: bool = True,
    fp16: bool = False,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
):
    """
    加载 Qwen2-VL 模型和处理器。
    
    参数:
        model_name_or_path: 模型标识符或路径
        finetune_mode: "lora" 或 "qlora"
        load_in_4bit: 使用 4-bit 量化
        load_in_8bit: 使用 8-bit 量化
        bf16: 使用 bfloat16
        fp16: 使用 float16
        device_map: 设备映射策略
        trust_remote_code: 信任远程代码
        cache_dir: 模型缓存目录
        
    返回:
        (模型, 处理器) 元组
    """
    logger = logging.getLogger("qwen2vl_slm")
    
    # 加载处理器
    logger.info(f"从 {model_name_or_path} 加载处理器")
    processor = Qwen2VLProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    
    # 配置量化
    quantization_config = None
    
    if finetune_mode == "qlora" or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("使用 4-bit 量化 (QLoRA)")
        except ImportError:
            logger.warning("bitsandbytes 不可用，回退到标准 LoRA")
            load_in_4bit = False
    elif load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("使用 8-bit 量化")
        except ImportError:
            logger.warning("bitsandbytes 不可用，使用标准精度")
            load_in_8bit = False
    
    # 确定 torch 数据类型
    torch_dtype = torch.float32
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logger.info("使用 bfloat16")
    elif fp16:
        torch_dtype = torch.float16
        logger.info("使用 float16")
    
    # 加载模型
    logger.info(f"从 {model_name_or_path} 加载模型")
    
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    elif load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    if device_map:
        model_kwargs["device_map"] = device_map
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    
    # 如果量化则准备模型进行 k-bit 训练（必须在梯度检查点之前）
    if (load_in_4bit or load_in_8bit) and PEFT_AVAILABLE:
        model = prepare_model_for_kbit_training(model)
        logger.info("模型已准备用于 k-bit 训练")
    
    # 启用梯度检查点以节省显存
    # 为梯度检查点兼容性禁用缓存
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("梯度检查点已启用")
    
    return model, processor


def apply_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> nn.Module:
    """
    对模型应用 LoRA。
    
    参数:
        model: 基础模型
        r: LoRA 秩
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 应用 LoRA 的模块名列表
        bias: 偏置训练模式
        task_type: 任务类型
        
    返回:
        带 LoRA 的 PEFT 模型
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("LoRA 需要 peft 库")
    
    logger = logging.getLogger("qwen2vl_slm")
    
    # 如果未指定则自动检测目标模块
    # 对 12GB 显存使用完整模块以获得最佳性能
    if target_modules is None:
        # 同时使用注意力层和 MLP 层以获得最大适应能力
        possible_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        
        # 检查模型中存在的模块
        target_modules = []
        model_modules = [name for name, _ in model.named_modules()]
        
        for module in possible_modules:
            if any(module in m for m in model_modules):
                target_modules.append(module)
        
        logger.info(f"使用完整目标模块: {target_modules}")
    
    # 创建 LoRA 配置
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


# ============================================================================
# 训练
# ============================================================================

class SLMTrainer:
    """适配重构数据集的自定义训练器。"""
    
    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        train_dataset: SLMDataset,
        val_dataset: SLMDataset,
        output_dir: str,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        early_stopping_patience: int = 3,
        max_grad_norm: float = 1.0,
        max_new_tokens: int = 4,
        seed: int = 42,
        bf16: bool = False,
        fp16: bool = True,
        target_mode: str = "3class",  # 新增: 3class/2class/joint
        task_mode: str = "naive_sft",
        prediction_mode: str = "auto",
        compute_candidate_scores: bool = True,
        selection_metric: str = "auto",
        causal_loss_weight: float = 0.0,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.target_mode = target_mode
        
        # 训练超参数
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.bf16 = bf16
        self.fp16 = fp16
        self.task_mode = task_mode
        self.prediction_mode = prediction_mode
        self.compute_candidate_scores = compute_candidate_scores
        self.selection_metric = selection_metric
        self.causal_loss_weight = causal_loss_weight
        self.resolved_prediction_mode = resolve_prediction_mode(
            target_mode=target_mode,
            prediction_mode=prediction_mode,
            compute_candidate_scores=compute_candidate_scores,
        )
        
        # 状态
        self.global_step = 0
        self.best_val_metric = -float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # 设置
        self._setup_directories()
        self._setup_optimizer()
        self._setup_dataloaders()
        
        # 日志
        self.train_log = []
        self.writer = SummaryWriter(os.path.join(output_dir, 'logs', 'tensorboard'))
        
        # Initialize Weights & Biases (optional)
        self.use_wandb = WANDB_AVAILABLE and os.environ.get('WANDB_DISABLED', 'false').lower() != 'true'
        if self.use_wandb:
            try:
                # Get run name from config or generate default
                run_name = os.environ.get('WANDB_RUN_NAME', None)
                
                wandb.init(
                    project=os.environ.get('WANDB_PROJECT', 'qwen2vl-slm'),
                    name=run_name,
                    dir=output_dir,
                    config={
                        'target_mode': target_mode,
                        'num_train_epochs': num_train_epochs,
                        'learning_rate': learning_rate,
                        'per_device_train_batch_size': per_device_train_batch_size,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'warmup_ratio': warmup_ratio,
                        'weight_decay': weight_decay,
                        'bf16': bf16,
                        'fp16': fp16,
                        'output_dir': output_dir,
                        'task_mode': task_mode,
                        'prediction_mode': self.resolved_prediction_mode,
                        'selection_metric': selection_metric,
                        'causal_loss_weight': causal_loss_weight,
                    }
                )
                logging.getLogger("qwen2vl_slm").info(f"✓ W&B initialized: {wandb.run.url if wandb.run else 'N/A'}")
            except Exception as e:
                logging.getLogger("qwen2vl_slm").warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
        
        # 设备
        self.device = next(model.parameters()).device

    def _mixed_precision_enabled(self) -> bool:
        return torch.cuda.is_available() and (self.bf16 or self.fp16)

    def _mixed_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.bf16 else torch.float16

    def _primary_class_names(self) -> List[str]:
        return CLASS_NAMES_2 if self.target_mode == "2class" else CLASS_NAMES_3

    def _selection_metric_name(self) -> str:
        if self.selection_metric != "auto":
            return self.selection_metric
        if self.target_mode == "2class":
            return "balanced_accuracy"
        return "macro_f1"

    def _selection_metric_value(self, metrics: Dict[str, Any]) -> float:
        metric_name = self._selection_metric_name()
        value = metrics.get(metric_name)
        if value is None:
            return 0.0
        return float(value)

    def _get_true_label(self, batch: Dict[str, Any], sample_idx: int) -> str:
        if self.target_mode == "2class":
            return batch["labels_str_2c"][sample_idx]
        return batch["labels_str_3c"][sample_idx]

    @staticmethod
    def _extract_prompt_length(sample_labels: torch.Tensor) -> int:
        valid_positions = (sample_labels != -100).nonzero(as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return int(sample_labels.shape[0])
        return int(valid_positions[0].item())

    def _slice_image_grid_thw(
        self,
        image_grid_thw: Optional[torch.Tensor],
        batch_size: int,
        sample_idx: int,
    ) -> Optional[torch.Tensor]:
        if image_grid_thw is None:
            return None

        if image_grid_thw.dim() == 2 and batch_size > 0:
            images_per_sample = image_grid_thw.shape[0] // batch_size
            start = sample_idx * images_per_sample
            end = start + images_per_sample
            return image_grid_thw[start:end]

        if image_grid_thw.dim() == 3:
            return image_grid_thw[sample_idx]

        return image_grid_thw

    @staticmethod
    def _pool_supervised_hidden_states(
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        supervision_mask = (labels != -100).unsqueeze(-1).to(hidden_states.dtype)
        token_counts = supervision_mask.sum(dim=1).clamp(min=1.0)
        pooled = (hidden_states * supervision_mask).sum(dim=1) / token_counts
        return F.normalize(pooled, dim=-1)

    def _compute_causal_consistency_loss(
        self,
        anchor_hidden_states: torch.Tensor,
        anchor_labels: torch.Tensor,
        positive_indices: List[Optional[int]],
    ) -> torch.Tensor:
        valid_pairs = [
            (anchor_idx, pos_idx)
            for anchor_idx, pos_idx in enumerate(positive_indices)
            if pos_idx is not None
        ]
        if not valid_pairs:
            return torch.zeros((), device=self.device, dtype=anchor_hidden_states.dtype)

        anchor_positions = torch.tensor(
            [anchor_idx for anchor_idx, _ in valid_pairs],
            device=self.device,
            dtype=torch.long,
        )
        positive_items = [self.train_dataset[pos_idx] for _, pos_idx in valid_pairs]
        positive_batch = self.collate_fn(positive_items)

        pos_input_ids = positive_batch["input_ids"].to(self.device)
        pos_attention_mask = positive_batch["attention_mask"].to(self.device)
        pos_pixel_values = positive_batch["pixel_values"].to(self.device)
        pos_labels = positive_batch["labels"].to(self.device)
        pos_image_grid_thw = positive_batch.get("image_grid_thw")
        if pos_image_grid_thw is not None:
            pos_image_grid_thw = pos_image_grid_thw.to(self.device)

        with torch.no_grad():
            pos_outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                pixel_values=pos_pixel_values,
                image_grid_thw=pos_image_grid_thw,
                labels=pos_labels,
                output_hidden_states=True,
            )
            positive_repr = self._pool_supervised_hidden_states(
                pos_outputs.hidden_states[-1],
                pos_labels,
            ).detach()

        anchor_repr = self._pool_supervised_hidden_states(
            anchor_hidden_states.index_select(0, anchor_positions),
            anchor_labels.index_select(0, anchor_positions),
        )
        cosine_sim = F.cosine_similarity(anchor_repr, positive_repr, dim=-1)
        return (1.0 - cosine_sim).mean()

    @torch.no_grad()
    def _generate_prediction_for_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        labels: torch.Tensor,
        sample_idx: int,
        true_label_2c: str,
        true_label_3c: str,
    ) -> Dict[str, str]:
        prompt_len = self._extract_prompt_length(labels[sample_idx])
        prompt_len = max(prompt_len, 1)

        sample_input_ids = input_ids[sample_idx:sample_idx + 1, :prompt_len]
        sample_attention_mask = attention_mask[sample_idx:sample_idx + 1, :prompt_len]
        sample_pixel_values = pixel_values[sample_idx:sample_idx + 1]
        sample_image_grid_thw = self._slice_image_grid_thw(
            image_grid_thw=image_grid_thw,
            batch_size=input_ids.shape[0],
            sample_idx=sample_idx,
        )

        sample_payload = {
            "input_ids": sample_input_ids.squeeze(0).detach().cpu(),
            "attention_mask": sample_attention_mask.squeeze(0).detach().cpu(),
            "pixel_values": sample_pixel_values.squeeze(0).detach().cpu(),
            "image_grid_thw": (
                sample_image_grid_thw.detach().cpu()
                if sample_image_grid_thw is not None
                else None
            ),
            "label": true_label_2c if self.target_mode == "2class" else true_label_3c,
            "label_2c": true_label_2c,
            "label_3c": true_label_3c,
        }
        result = evaluate_sample_with_candidate_scores(
            model=self.model,
            processor=self.processor,
            sample=sample_payload,
            target_mode=self.target_mode,
            token_style=self.train_dataset.token_style,
            max_new_tokens=self.max_new_tokens,
            prediction_mode=self.prediction_mode,
            compute_candidate_scores=self.compute_candidate_scores,
        )

        return {
            "generated_text": result["generated_text"],
            "pred_label": result["pred_label"],
        }
        
    def _setup_directories(self):
        """创建输出目录。"""
        dirs = [
            'checkpoints',
            'adapter_best',
            'eval_after',
            'predictions',
            'logs',
        ]
        for d in dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)
    
    def _setup_optimizer(self):
        """设置优化器和学习率调度器。"""
        # 区分应该/不应该有权重衰减的参数
        no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )
        
        # 调度器在 train() 中知道总步数后设置
        self.scheduler = None
    
    def _setup_dataloaders(self):
        """设置训练和验证数据加载器。"""
        # 使用重构版数据集的 collator
        self.collate_fn = get_data_collator(self.processor)
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,  # 调试设为 0，生产环境可增加
            pin_memory=True,
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=True,
        )
    
    def train(self):
        """主训练循环。"""
        logger = logging.getLogger("qwen2vl_slm")
        
        # 计算总训练步数
        steps_per_epoch = max(
            1,
            math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps),
        )
        total_steps = steps_per_epoch * self.num_train_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        logger.info(f"总训练步数: {total_steps}")
        logger.info(f"预热步数: {warmup_steps}")
        logger.info(f"验证预测模式: {self.resolved_prediction_mode}")
        logger.info(f"最佳模型选择指标: {self._selection_metric_name()}")
        if self.task_mode == "causal_train":
            logger.info(f"因果一致性损失权重: {self.causal_loss_weight}")
        
        # 设置调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # 训练循环
        for epoch in range(self.num_train_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"轮次 {epoch + 1}/{self.num_train_epochs}")
            logger.info(f"{'='*60}")
            
            # 训练
            train_summary = self._train_epoch(epoch)
            train_loss = train_summary["loss"]
            
            # 评估
            val_metrics = self._evaluate()
            val_loss = val_metrics.get("loss", 0)
            val_f1 = val_metrics.get("macro_f1", 0)
            val_selection = self._selection_metric_value(val_metrics)
            
            # 记录
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_ce_loss": train_summary.get("ce_loss", train_loss),
                "train_causal_loss": train_summary.get("causal_loss", 0.0),
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_balanced_accuracy": val_metrics.get("balanced_accuracy", 0.0),
                "val_selection_metric": val_selection,
                "selection_metric_name": self._selection_metric_name(),
                "learning_rate": self.scheduler.get_last_lr()[0],
            }
            self.train_log.append(log_entry)
            
            logger.info(f"训练损失: {train_loss:.4f}")
            logger.info(f"训练 CE 损失: {train_summary.get('ce_loss', train_loss):.4f}")
            if self.task_mode == "causal_train":
                logger.info(f"训练因果损失: {train_summary.get('causal_loss', 0.0):.4f}")
            logger.info(f"验证损失: {val_loss:.4f}")
            logger.info(f"验证宏 F1: {val_f1:.4f}")
            logger.info(
                f"验证选择指标 {self._selection_metric_name()}: {val_selection:.4f}"
            )
            
            # TensorBoard 日志
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/train_ce", train_summary.get("ce_loss", train_loss), epoch)
            self.writer.add_scalar("Loss/train_causal", train_summary.get("causal_loss", 0.0), epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metrics/val_macro_f1", val_f1, epoch)
            self.writer.add_scalar("Metrics/val_balanced_accuracy", val_metrics.get("balanced_accuracy", 0.0), epoch)
            self.writer.add_scalar(
                f"Metrics/val_{self._selection_metric_name()}",
                val_selection,
                epoch,
            )
            self.writer.add_scalar("Learning_rate", self.scheduler.get_last_lr()[0], epoch)
            
            # Weights & Biases logging
            if self.use_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/ce_loss": train_summary.get("ce_loss", train_loss),
                    "train/causal_loss": train_summary.get("causal_loss", 0.0),
                    "train/epoch": epoch + 1,
                    "val/loss": val_loss,
                    "val/macro_f1": val_f1,
                    "val/accuracy": val_metrics.get("accuracy", 0),
                    "val/balanced_accuracy": val_metrics.get("balanced_accuracy", 0),
                    f"val/{self._selection_metric_name()}": val_selection,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/global_step": self.global_step,
                }, step=self.global_step)
                
                # Log per-class metrics if available
                if "per_class" in val_metrics:
                    for cls_name, cls_metrics in val_metrics["per_class"].items():
                        wandb.log({
                            f"val/{cls_name}_precision": cls_metrics.get("precision", 0),
                            f"val/{cls_name}_recall": cls_metrics.get("recall", 0),
                            f"val/{cls_name}_f1": cls_metrics.get("f1", 0),
                        }, step=self.global_step)
            
            # 保存检查点
            if (epoch + 1) % max(1, self.num_train_epochs // 5) == 0:
                self._save_checkpoint(epoch + 1)
            
            # 检查最佳模型
            if val_selection > self.best_val_metric:
                logger.info(
                    f"新的最佳模型! 验证 {self._selection_metric_name()}: {val_selection:.4f}"
                )
                self.best_val_metric = val_selection
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                self._save_best_adapter()
            else:
                self.epochs_without_improvement += 1
                logger.info(f"已 {self.epochs_without_improvement} 轮无改进")
            
            # 早停
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"轮次 {epoch + 1} 后触发早停")
                break
        
        # 保存训练日志
        self._save_train_log()
        
        # Log final summary to W&B
        if self.use_wandb:
            wandb.log({
                "best/epoch": self.best_epoch,
                f"best/val_{self._selection_metric_name()}": self.best_val_metric,
            })
            # Save adapter_best as artifact
            try:
                adapter_dir = os.path.join(self.output_dir, 'adapter_best')
                if os.path.exists(adapter_dir):
                    artifact = wandb.Artifact(f"adapter-{wandb.run.id}", type="model")
                    artifact.add_dir(adapter_dir)
                    wandb.log_artifact(artifact)
                    logger.info("✓ Best adapter uploaded to W&B as artifact")
            except Exception as e:
                logger.warning(f"Failed to upload adapter artifact: {e}")
            wandb.finish()
        
        logger.info(f"\n训练完成!")
        logger.info(f"最佳轮次: {self.best_epoch}")
        logger.info(
            f"最佳验证 {self._selection_metric_name()}: {self.best_val_metric:.4f}"
        )
        
        return self.best_val_metric
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一轮，带进度条和显存监控。"""
        self.model.train()
        
        total_loss = 0
        total_ce_loss = 0
        total_causal_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        # 创建进度条
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"轮次 {epoch+1}/{self.num_train_epochs}",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for step, batch in pbar:
            # 移至设备
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)
            
            # 获取 image_grid_thw (Qwen2-VL 需要)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            
            # 混合精度前向传播以节省显存
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self._mixed_precision_enabled(),
                dtype=self._mixed_precision_dtype(),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                    output_hidden_states=(self.task_mode == "causal_train" and self.causal_loss_weight > 0),
                )

                ce_loss = outputs.loss
                causal_loss = torch.zeros((), device=self.device, dtype=ce_loss.dtype)
                if (
                    self.task_mode == "causal_train"
                    and self.causal_loss_weight > 0
                    and labels is not None
                ):
                    causal_loss = self._compute_causal_consistency_loss(
                        anchor_hidden_states=outputs.hidden_states[-1],
                        anchor_labels=labels,
                        positive_indices=batch.get("positive_indices", []),
                    )

                combined_loss = ce_loss + self.causal_loss_weight * causal_loss
                loss = combined_loss / self.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 释放前向传播显存
            del outputs
            
            step_loss_value = float(combined_loss.detach().item())
            step_ce_loss_value = float(ce_loss.detach().item())
            step_causal_loss_value = float(causal_loss.detach().item())
            total_loss += step_loss_value
            total_ce_loss += step_ce_loss_value
            total_causal_loss += step_causal_loss_value
            
            # 梯度累积
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 清理缓存防止显存溢出
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log to W&B at each step (every gradient accumulation)
                if self.use_wandb and self.global_step % self.logging_steps == 0:
                    log_dict = {
                        "train/step_loss": step_loss_value,
                        "train/step_ce_loss": step_ce_loss_value,
                        "train/step_causal_loss": step_causal_loss_value,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step / len(self.train_dataloader)),
                    }
                    # Add GPU memory stats if available
                    if torch.cuda.is_available():
                        log_dict["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                        log_dict["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
                    wandb.log(log_dict, step=self.global_step)
            
            # 释放输入张量
            del input_ids, attention_mask, pixel_values, labels, image_grid_thw
            
            num_batches += 1
            
            # 用损失和显存信息更新进度条
            mem_info = ""
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_info = f"显存: {mem_allocated:.1f}G/{mem_reserved:.1f}G"
            
            pbar.set_postfix({
                'loss': f"{step_loss_value:.4f}",
                'ce': f"{step_ce_loss_value:.4f}",
                'causal': f"{step_causal_loss_value:.4f}",
                'mem': mem_info
            })
        
        pbar.close()
        
        # 处理剩余梯度
        if (step + 1) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        if num_batches == 0:
            return {"loss": 0.0, "ce_loss": 0.0, "causal_loss": 0.0}

        return {
            "loss": total_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "causal_loss": total_causal_loss / num_batches,
        }
    
    def _evaluate(self) -> Dict[str, float]:
        """在验证集上计算 teacher-forcing 损失和真实生成分类指标。"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        y_true: List[str] = []
        y_pred: List[str] = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移至设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch.get("labels")
                if labels is not None:
                    labels = labels.to(self.device)
                
                # 获取 image_grid_thw
                image_grid_thw = batch.get("image_grid_thw")
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)
                
                # 前向传播计算损失
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self._mixed_precision_enabled(),
                    dtype=self._mixed_precision_dtype(),
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels,
                    )
                    
                    if outputs.loss is not None:
                        total_loss += outputs.loss.item()
                
                if labels is not None:
                    for sample_idx in range(input_ids.shape[0]):
                        prediction = self._generate_prediction_for_sample(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            labels=labels,
                            sample_idx=sample_idx,
                            true_label_2c=batch["labels_str_2c"][sample_idx],
                            true_label_3c=batch["labels_str_3c"][sample_idx],
                        )
                        y_true.append(self._get_true_label(batch, sample_idx))
                        y_pred.append(prediction["pred_label"])

                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self._primary_class_names(),
            subset_name="val",
        )
        metrics["loss"] = avg_loss

        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """保存训练检查点。"""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints', f'checkpoint-epoch-{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存适配器
        self.model.save_pretrained(checkpoint_dir)
        
        # 保存优化器和调度器状态
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        logger = logging.getLogger("qwen2vl_slm")
        logger.info(f"检查点保存至 {checkpoint_dir}")
    
    def _save_best_adapter(self):
        """保存最佳适配器。"""
        adapter_dir = os.path.join(self.output_dir, 'adapter_best')
        
        # 删除已有最佳适配器
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        
        os.makedirs(adapter_dir, exist_ok=True)
        
        # 保存适配器
        self.model.save_pretrained(adapter_dir)
        
        # 保存处理器
        self.processor.save_pretrained(adapter_dir)
        
        logger = logging.getLogger("qwen2vl_slm")
        logger.info(f"最佳适配器保存至 {adapter_dir}")
    
    def _save_train_log(self):
        """保存训练日志到文件，并生成可视化图表。"""
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # JSON
        with open(os.path.join(log_dir, 'train_log.json'), 'w') as f:
            json.dump({
                'best_epoch': self.best_epoch,
                'best_val_metric': self.best_val_metric,
                'logs': self.train_log,
            }, f, indent=2)
        
        # CSV
        import csv
        if self.train_log:
            with open(os.path.join(log_dir, 'train_log.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.train_log[0].keys())
                writer.writeheader()
                writer.writerows(self.train_log)
        
        # 可视化训练曲线
        self._plot_training_curves(log_dir)
    
    def _plot_training_curves(self, log_dir: str):
        """绘制并保存训练曲线。"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.train_log:
                return
            
            epochs = [log['epoch'] for log in self.train_log]
            train_losses = [log['train_loss'] for log in self.train_log]
            val_losses = [log['val_loss'] for log in self.train_log]
            val_f1s = [log['val_macro_f1'] for log in self.train_log]
            val_selection = [log.get('val_selection_metric', log['val_macro_f1']) for log in self.train_log]
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 损失曲线
            ax1 = axes[0, 0]
            ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
            ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4)
            ax1.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training & Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 验证 F1 曲线
            ax2 = axes[0, 1]
            ax2.plot(epochs, val_f1s, 'g-^', label='Val Macro F1', markersize=4)
            ax2.plot(
                epochs,
                val_selection,
                'c-o',
                label=f"Val {self._selection_metric_name()}",
                markersize=4,
            )
            ax2.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
            ax2.axhline(
                y=self.best_val_metric,
                color='r',
                linestyle=':',
                alpha=0.5,
                label=f"Best {self._selection_metric_name()} ({self.best_val_metric:.4f})",
            )
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Macro F1')
            ax2.set_title('Validation Macro F1')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 学习率曲线
            ax3 = axes[1, 0]
            learning_rates = [log.get('learning_rate', 0) for log in self.train_log]
            ax3.plot(epochs, learning_rates, 'm-o', label='Learning Rate', markersize=4)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # 训练摘要
            ax4 = axes[1, 1]
            ax4.axis('off')
            summary_text = f"""
Training Summary
================
Best Epoch: {self.best_epoch}
Best Val {self._selection_metric_name()}: {self.best_val_metric:.4f}
Final Train Loss: {train_losses[-1]:.4f}
Final Val Loss: {val_losses[-1]:.4f}
Total Epochs: {len(self.train_log)}
            """
            ax4.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            logger = logging.getLogger("qwen2vl_slm")
            logger.info(f"训练曲线已保存: {os.path.join(log_dir, 'training_curves.png')}")
            
        except ImportError:
            logger = logging.getLogger("qwen2vl_slm")
            logger.warning("matplotlib 未安装，跳过训练曲线可视化")
        except Exception as e:
            logger = logging.getLogger("qwen2vl_slm")
            logger.warning(f"绘制训练曲线失败: {e}")


# ============================================================================
# 主入口
# ============================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="在 SLM 缺陷检测任务上训练 Qwen2-VL (适配重构数据集)"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="模型缓存目录",
    )
    
    # 数据参数 - 支持新旧格式
    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="训练清单 JSONL 路径 (兼容新旧格式)",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        required=True,
        help="验证清单 JSONL 路径 (兼容新旧格式)",
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        default=None,
        help="测试清单 JSONL 路径 (可选，仅作参考)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="自定义提示文本 (None则使用target_mode对应的默认prompt)",
    )
    
    # 新增: 任务模式参数 (重构数据集特有)
    parser.add_argument(
        "--target_mode",
        type=str,
        default="3class",
        choices=["3class", "2class", "joint"],
        help="目标分类模式: 3class(三分类), 2class(二分类), joint(联合)",
    )
    parser.add_argument(
        "--token_style",
        type=str,
        default="letters",
        choices=["letters", "words"],
        help="Token风格: letters(A/B/C/N/Y) 或 words(normal/HEW/LEL)",
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default="naive_sft",
        choices=["naive_sft", "causal_train", "inference"],
        help="任务模式: naive_sft(标准SFT), causal_train(因果训练), inference(推理)",
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="结果输出目录",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=str2bool,
        default=False,
        help="覆盖已有输出目录",
    )
    
    # 训练参数
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="qlora",
        choices=["lora", "qlora"],
        help="微调模式",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=lambda x: x.lower() == "true",
        default=True,
        help="使用 4-bit 量化",
    )
    parser.add_argument(
        "--load_in_8bit",
        type=str2bool,
        default=False,
        help="使用 8-bit 量化",
    )
    parser.add_argument(
        "--bf16",
        type=str2bool,
        default=True,
        help="使用 bfloat16",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=False,
        help="使用 float16",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="每设备训练批次大小",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="每设备评估批次大小",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="预热比例",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="早停耐心值",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪最大范数",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4,
        help="评估时生成的最大 token 数",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="auto",
        choices=["auto", "generate", "candidate_score"],
        help="验证预测模式: auto(推荐), generate(自由生成), candidate_score(闭集候选打分)",
    )
    parser.add_argument(
        "--compute_candidate_scores",
        type=str2bool,
        default=True,
        help="验证时是否启用候选打分能力；配合 prediction_mode=auto/candidate_score 使用",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="auto",
        choices=["auto", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
        help="最佳模型选择指标；auto=2class 用 balanced_accuracy，3class 用 macro_f1",
    )
    parser.add_argument(
        "--causal_loss_weight",
        type=float,
        default=0.0,
        help="causal_train 下的一致性损失权重",
    )
    
    # LoRA 参数
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA 秩",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    
    # Weights & Biases logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen2vl-slm",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name",
    )
    
    # 其他参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点目录恢复",
    )
    
    return parser.parse_args()


def main():
    """主入口。"""
    # 优化 CUDA 显存管理
    if torch.cuda.is_available():
        # 设置显存分配器配置以避免碎片化
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        # 清理缓存显存
        torch.cuda.empty_cache()
        # 设置显存比例以留出一些余量
        torch.cuda.set_per_process_memory_fraction(0.90)
    
    args = parse_args()
    
    # 验证输出目录
    validate_output_directory(
        args.output_dir,
        args.model_name_or_path,
        args.overwrite_output_dir,
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['config', 'checkpoints', 'adapter_best', 'eval_after', 'predictions', 'reports', 'logs']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # 设置日志
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    logger.info("="*60)
    logger.info("Qwen2-VL SLM 缺陷检测 - 训练 (适配重构数据集)")
    logger.info("="*60)
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"基础模型: {args.model_name_or_path}")
    logger.info(f"微调模式: {args.finetune_mode}")
    logger.info(f"目标模式: {args.target_mode} ({args.token_style})")
    logger.info(f"任务模式: {args.task_mode}")
    logger.info(
        "验证预测模式: %s",
        resolve_prediction_mode(
            target_mode=args.target_mode,
            prediction_mode=args.prediction_mode,
            compute_candidate_scores=args.compute_candidate_scores,
        ),
    )
    
    # 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'config', 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载模型和处理器
    logger.info("加载模型和处理器...")
    model, processor = load_model_and_processor(
        args.model_name_or_path,
        finetune_mode=args.finetune_mode,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bf16=args.bf16,
        fp16=args.fp16,
        cache_dir=args.cache_dir,
    )
    
    # 应用 LoRA
    if PEFT_AVAILABLE and args.finetune_mode in ["lora", "qlora"]:
        logger.info("应用 LoRA...")
        model = apply_lora(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
    # 加载数据集 - 使用重构版参数
    logger.info("加载数据集...")
    train_dataset = SLMDataset(
        manifest_path=args.train_manifest,
        processor=processor,
        prompt=args.prompt,
        split="train",
        target_mode=args.target_mode,      # 新增
        token_style=args.token_style,      # 新增
        task_mode=args.task_mode,          # 新增
    )
    
    val_dataset = SLMDataset(
        manifest_path=args.val_manifest,
        processor=processor,
        prompt=args.prompt,
        split="val",
        target_mode=args.target_mode,      # 新增
        token_style=args.token_style,      # 新增
        task_mode="inference" if args.task_mode == "inference" else "naive_sft",
    )
    
    logger.info(f"训练样本: {len(train_dataset)}")
    logger.info(f"验证样本: {len(val_dataset)}")
    
    # 显示类别分布
    logger.info("训练集类别分布:")
    if args.target_mode in ["3class", "joint"]:
        for name in CLASS_NAMES_3:
            count = len(train_dataset.indices_by_label_3c.get(name, []))
            logger.info(f"  {name}: {count}")
    if args.target_mode in ["2class", "joint"]:
        for name in CLASS_NAMES_2:
            count = len(train_dataset.indices_by_label_2c.get(name, []))
            logger.info(f"  {name}: {count}")
    
    # 设置 W&B 环境变量
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("W&B requested but not installed. Install with: pip install wandb")
        else:
            os.environ['WANDB_PROJECT'] = args.wandb_project
            if args.wandb_run_name:
                os.environ['WANDB_RUN_NAME'] = args.wandb_run_name
            if args.wandb_entity:
                os.environ['WANDB_ENTITY'] = args.wandb_entity
            logger.info(f"W&B logging enabled: project={args.wandb_project}")
    else:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # 创建训练器
    trainer = SLMTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        max_grad_norm=args.max_grad_norm,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        target_mode=args.target_mode,      # 新增
        task_mode=args.task_mode,
        prediction_mode=args.prediction_mode,
        compute_candidate_scores=args.compute_candidate_scores,
        selection_metric=args.selection_metric,
        causal_loss_weight=args.causal_loss_weight,
    )
    
    # 训练
    logger.info("开始训练...")
    best_metric = trainer.train()
    
    logger.info("="*60)
    logger.info("训练完成!")
    logger.info(f"最佳验证宏 F1: {best_metric:.4f}")
    logger.info(f"最佳适配器保存至: {os.path.join(args.output_dir, 'adapter_best')}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
