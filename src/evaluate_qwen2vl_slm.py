"""
Qwen2-VL SLM 缺陷检测 - 评估脚本 (适配重构数据集)

该脚本评估 Qwen2-VL 模型（基础模型或使用 LoRA 微调的模型）在
SLM 缺陷检测任务上的性能，计算综合指标并生成对比报告。

适配特性:
- 支持重构版数据集 (二分类/三分类/joint 模式)
- 支持 letters/words token 风格
- 兼容新旧 manifest 格式

支持:
- 基础模型评估（微调前）
- 微调模型评估（使用 LoRA 适配器）
- 子集评估（源域、目标域、未见过样本）
- 前后对比报告
"""

import os
import sys
import json
import logging
import argparse
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Transformers 导入
try:
    import transformers
    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2VLProcessor,
        AutoTokenizer,
    )
except ImportError as e:
    print(f"导入 transformers 出错: {e}")
    sys.exit(1)

# PEFT 导入
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("警告: peft 不可用。无法加载 LoRA 适配器。")

# 导入本地模块 - 使用重构版数据集
from dataset_qwen2vl_slm import (
    SLMDataset,
    SLMDatasetForInference,
    get_data_collator,
    CLASS_NAMES_3,        # 三分类类别
    CLASS_NAMES_2,        # 二分类类别
    CLASS_TOKEN_MAP_3,
    CLASS_TOKEN_MAP_2,
    CLASS_NAME_TO_ID_3,   # 三分类映射
    CLASS_NAME_TO_ID_2,   # 二分类映射
    ID_TO_CLASS_NAME_3,
    ID_TO_CLASS_NAME_2,
    build_target_text,
    choose_prompt,
)
from utils_metrics import (
    compute_classification_metrics,
    normalize_prediction,
    save_metrics,
    save_predictions,
    save_transfer_gap,
    compute_transfer_gap,
    generate_before_after_comparison,
    print_metrics_table,
    plot_confusion_matrix,
)


# ============================================================================
# 日志设置
# ============================================================================

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """设置日志。"""
    logger = logging.getLogger("qwen2vl_eval")
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'eval.log'),
                mode='w'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


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

def load_model_for_evaluation(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bf16: bool = True,
    fp16: bool = False,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any]:
    """
    加载用于评估的模型和处理器。
    
    参数:
        model_name_or_path: 基础模型路径或名称
        adapter_path: LoRA 适配器路径（None 表示基础模型）
        load_in_4bit: 使用 4-bit 量化
        load_in_8bit: 使用 8-bit 量化
        bf16: 使用 bfloat16
        fp16: 使用 float16
        device_map: 设备映射策略
        cache_dir: 缓存目录
        
    返回:
        (模型, 处理器) 元组
    """
    logger = logging.getLogger("qwen2vl_eval")
    
    # 加载处理器
    logger.info(f"从 {model_name_or_path} 加载处理器")
    processor = Qwen2VLProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    
    # 配置量化
    quantization_config = None
    
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("使用 4-bit 量化")
        except ImportError:
            logger.warning("bitsandbytes 不可用")
    elif load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("使用 8-bit 量化")
        except ImportError:
            logger.warning("bitsandbytes 不可用")
    
    # 确定 torch 数据类型
    torch_dtype = torch.float32
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logger.info("使用 bfloat16")
    elif fp16:
        torch_dtype = torch.float16
        logger.info("使用 float16")
    
    # 加载基础模型
    logger.info(f"从 {model_name_or_path} 加载基础模型")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
        "device_map": device_map,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    elif load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    
    # 加载 LoRA 适配器如果提供
    if adapter_path is not None:
        if not PEFT_AVAILABLE:
            raise RuntimeError("需要 peft 库来加载 LoRA 适配器")
        
        logger.info(f"从 {adapter_path} 加载 LoRA 适配器")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("LoRA 适配器加载成功")
    else:
        logger.info("使用基础模型（无适配器）")
    
    model.eval()
    
    return model, processor


# ============================================================================
# Token 映射和预测解析
# ============================================================================

def parse_prediction(
    generated_text: str,
    target_mode: str = "3class",
    token_style: str = "letters",
) -> Tuple[str, Optional[str]]:
    """
    解析模型生成的文本为预测标签。
    
    参数:
        generated_text: 模型生成的文本
        target_mode: 3class/2class/joint
        token_style: letters/words
        
    返回:
        (主预测标签, 次预测标签或None)
    """
    tokens = re.findall(r"[A-Za-z]+", generated_text.strip().lower())

    label_map_3class = (
        {"a": "normal", "b": "HEW", "c": "LEL"}
        if token_style == "letters"
        else {"normal": "normal", "hew": "HEW", "hev": "HEW", "lel": "LEL"}
    )
    label_map_2class = (
        {"n": "normal", "y": "abnormal"}
        if token_style == "letters"
        else {"normal": "normal", "abnormal": "abnormal"}
    )

    def find_first_label(sequence: List[str], mapping: Dict[str, str]) -> Tuple[str, int]:
        for idx, token in enumerate(sequence):
            if token in mapping:
                return mapping[token], idx
        return "UNKNOWN", -1

    if target_mode == "3class":
        pred_label, _ = find_first_label(tokens, label_map_3class)
        return pred_label, None

    if target_mode == "2class":
        pred_label, _ = find_first_label(tokens, label_map_2class)
        return pred_label, None

    if target_mode == "joint":
        pred_2c, idx_2c = find_first_label(tokens, label_map_2class)
        remaining_tokens = tokens[idx_2c + 1:] if idx_2c >= 0 else tokens
        pred_3c, _ = find_first_label(remaining_tokens, label_map_3class)
        return pred_2c, pred_3c

    return "UNKNOWN", None


def fold_3class_label_to_2class(label: str) -> str:
    if label == "normal":
        return "normal"
    if label in {"HEW", "LEL"}:
        return "abnormal"
    return "UNKNOWN"


def resolve_prediction_mode(
    target_mode: str,
    prediction_mode: str,
    compute_candidate_scores: bool,
) -> str:
    if prediction_mode != "auto":
        return prediction_mode
    if target_mode in {"2class", "3class"} and compute_candidate_scores:
        return "candidate_score"
    return "generate"


def build_candidate_definitions(
    target_mode: str,
    token_style: str,
) -> List[Dict[str, str]]:
    if target_mode == "3class":
        return [
            {
                "text": CLASS_TOKEN_MAP_3[token_style][label],
                "pred_label": label,
                "pred_label_2c": fold_3class_label_to_2class(label),
                "pred_label_3c": label,
            }
            for label in CLASS_NAMES_3
        ]

    if target_mode == "2class":
        return [
            {
                "text": CLASS_TOKEN_MAP_2[token_style][label],
                "pred_label": label,
                "pred_label_2c": label,
                "pred_label_3c": "normal" if label == "normal" else "UNKNOWN",
            }
            for label in CLASS_NAMES_2
        ]

    if target_mode == "joint":
        return [
            {
                "text": build_target_text(label, fold_3class_label_to_2class(label), "joint", token_style),
                "pred_label": fold_3class_label_to_2class(label),
                "pred_label_2c": fold_3class_label_to_2class(label),
                "pred_label_3c": label,
            }
            for label in CLASS_NAMES_3
        ]

    raise ValueError(f"不支持的 target_mode: {target_mode}")


def score_candidate_continuations(
    model: torch.nn.Module,
    processor: Any,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    candidate_defs: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    results: List[Dict[str, Any]] = []

    prompt_input_ids = prompt_input_ids.unsqueeze(0).to(device)
    prompt_attention_mask = prompt_attention_mask.unsqueeze(0).to(device)
    pixel_values = pixel_values.unsqueeze(0).to(device)
    image_grid_thw = image_grid_thw.to(device) if image_grid_thw is not None else None

    tokenizer = processor.tokenizer

    for candidate in candidate_defs:
        candidate_text = candidate["text"]
        candidate_ids = tokenizer(
            candidate_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)

        if candidate_ids.shape[-1] == 0:
            raise ValueError(f"候选文本无法被 tokenizer 编码: {candidate_text!r}")

        candidate_attention = torch.ones_like(candidate_ids, device=device)
        full_input_ids = torch.cat([prompt_input_ids, candidate_ids], dim=1)
        full_attention_mask = torch.cat([prompt_attention_mask, candidate_attention], dim=1)
        labels = full_input_ids.clone()
        labels[:, :prompt_input_ids.shape[1]] = -100

        outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        avg_logprob = float(-outputs.loss.item())
        results.append(
            {
                **candidate,
                "avg_logprob": avg_logprob,
                "token_count": int(candidate_ids.shape[-1]),
            }
        )

    score_tensor = torch.tensor([item["avg_logprob"] for item in results], dtype=torch.float64)
    prob_tensor = torch.softmax(score_tensor, dim=0)
    for item, prob in zip(results, prob_tensor.tolist()):
        item["probability"] = float(prob)

    return results


# ============================================================================
# 评估核心
# ============================================================================

@torch.no_grad()
def evaluate_sample_with_candidate_scores(
    model: torch.nn.Module,
    processor: Any,
    sample: Dict[str, Any],
    target_mode: str = "3class",
    token_style: str = "letters",
    max_new_tokens: int = 4,
    prediction_mode: str = "auto",
    compute_candidate_scores: bool = True,
) -> Dict[str, Any]:
    """
    评估单个样本，计算生成结果和候选分数。
    
    参数:
        model: 模型
        processor: 处理器
        sample: 样本字典
        target_mode: 3class/2class/joint
        token_style: letters/words
        max_new_tokens: 最大生成 token 数
        
    返回:
        包含预测和分数的字典
    """
    device = next(model.parameters()).device
    
    true_label = sample["label"]
    resolved_prediction_mode = resolve_prediction_mode(
        target_mode=target_mode,
        prediction_mode=prediction_mode,
        compute_candidate_scores=compute_candidate_scores,
    )

    prompt_input_ids = sample["input_ids"]
    prompt_attention_mask = sample["attention_mask"]
    pixel_values = sample["pixel_values"]
    image_grid_thw = sample.get("image_grid_thw")

    generated_text = ""
    candidate_scores: Optional[List[Dict[str, Any]]] = None

    if resolved_prediction_mode == "candidate_score":
        candidate_scores = score_candidate_continuations(
            model=model,
            processor=processor,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            candidate_defs=build_candidate_definitions(target_mode, token_style),
        )
        best_candidate = max(candidate_scores, key=lambda item: item["avg_logprob"])
        pred_label = best_candidate["pred_label"]
        pred_label_2 = best_candidate["pred_label_3c"] if target_mode == "joint" else None
        pred_label_2c = best_candidate["pred_label_2c"]
        pred_label_3c = best_candidate["pred_label_3c"]
        generated_text = best_candidate["text"]
    else:
        inputs = {
            "input_ids": prompt_input_ids.unsqueeze(0).to(device),
            "attention_mask": prompt_attention_mask.unsqueeze(0).to(device),
            "pixel_values": pixel_values.unsqueeze(0).to(device),
        }
        if image_grid_thw is not None:
            inputs["image_grid_thw"] = image_grid_thw.to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        pred_label, pred_label_2 = parse_prediction(generated_text, target_mode, token_style)
        if target_mode == "3class":
            pred_label_3c = pred_label
            pred_label_2c = fold_3class_label_to_2class(pred_label)
        elif target_mode == "2class":
            pred_label_3c = sample.get("label_3c", "UNKNOWN")
            pred_label_2c = pred_label
        else:
            pred_label_2c = pred_label
            pred_label_3c = pred_label_2 or "UNKNOWN"

    true_label_3c = sample.get("label_3c", true_label if target_mode == "3class" else "UNKNOWN")
    true_label_2c = sample.get("label_2c", true_label if target_mode == "2class" else fold_3class_label_to_2class(true_label_3c))

    is_correct = pred_label == true_label
    if target_mode == "joint":
        is_correct = (pred_label_2c == true_label_2c) and (pred_label_3c == true_label_3c)

    result = {
        "generated_text": generated_text,
        "prediction_mode": resolved_prediction_mode,
        "candidate_scores": candidate_scores,
        "pred_label": pred_label,
        "pred_label_2": pred_label_2c,
        "pred_label_3c": pred_label_3c,
        "true_label": true_label,
        "true_label_3c": true_label_3c,
        "true_label_2c": true_label_2c,
        "is_correct": is_correct,
    }

    return result


def extract_training_aligned_prompt_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    将带监督标签的样本裁成训练验证时实际使用的 prompt 前缀。

    训练脚本在验证时不是直接使用 inference 数据集的 input_ids，而是从
    带 target_text 的样本里，按 labels 第一个非 -100 位置截出 prompt。
    独立评估复用这个逻辑，才能与训练验证口径一致。
    """
    labels = sample.get("labels")
    if not isinstance(labels, torch.Tensor):
        return sample

    valid_positions = (labels != -100).nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        prompt_len = int(labels.shape[0])
    else:
        prompt_len = int(valid_positions[0].item())
    prompt_len = max(prompt_len, 1)

    aligned_sample = dict(sample)

    input_ids = sample.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        aligned_sample["input_ids"] = input_ids[:prompt_len]

    attention_mask = sample.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        aligned_sample["attention_mask"] = attention_mask[:prompt_len]

    return aligned_sample


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    processor: Any,
    dataset: SLMDataset,
    output_dir: str,
    target_mode: str = "3class",
    token_style: str = "letters",
    batch_size: int = 1,
    max_new_tokens: int = 4,
    compute_candidate_scores: bool = True,
    prediction_mode: str = "auto",
    subset_name: str = "eval",
    prompt_alignment: str = "train_validation",
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    评估整个数据集。
    
    参数:
        model: 模型
        processor: 处理器
        dataset: 要评估的数据集
        output_dir: 结果输出目录
        target_mode: 3class/2class/joint
        token_style: letters/words
        batch_size: 批次大小
        max_new_tokens: 最大生成 token 数
        compute_candidate_scores: 是否计算候选分数
        subset_name: 该子集的名称
        
    返回:
        (指标字典, 预测列表) 元组
    """
    logger = logging.getLogger("qwen2vl_eval")
    logger.info(f"评估 {len(dataset)} 个样本...")
    logger.info(f"目标模式: {target_mode}, Token风格: {token_style}")
    logger.info(
        "预测策略: %s",
        resolve_prediction_mode(
            target_mode=target_mode,
            prediction_mode=prediction_mode,
            compute_candidate_scores=compute_candidate_scores,
        ),
    )
    
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    
    # 根据 target_mode 确定类别名称
    if target_mode == "3class":
        class_names = CLASS_NAMES_3
        class_name_to_id = CLASS_NAME_TO_ID_3
    elif target_mode == "2class":
        class_names = CLASS_NAMES_2
        class_name_to_id = CLASS_NAME_TO_ID_2
    else:  # joint，使用三分类作为主要评估
        class_names = CLASS_NAMES_3
        class_name_to_id = CLASS_NAME_TO_ID_3
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    open(predictions_path, "w", encoding="utf-8").close()
    
    # 使用进度条处理每个样本
    pbar = tqdm(
        range(len(dataset)),
        desc=f"评估 {subset_name}",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for idx in pbar:
        sample = dataset[idx]
        if prompt_alignment == "train_validation":
            sample = extract_training_aligned_prompt_sample(sample)
        
        # 详细评估
        result = evaluate_sample_with_candidate_scores(
            model, processor, sample, 
            target_mode=target_mode,
            token_style=token_style,
            max_new_tokens=max_new_tokens,
            prediction_mode=prediction_mode,
            compute_candidate_scores=compute_candidate_scores,
        )
        
        # 构建预测记录
        pred_record = {
            "id": sample["sample_id"],
            "true_label": sample["label"],
            "true_label_3c": result.get("true_label_3c", sample.get("label_3c", sample["label"])),
            "true_label_2c": result.get("true_label_2c", sample.get("label_2c", "normal" if sample["label"] == "normal" else "abnormal")),
            "pred_label": result["pred_label"],
            "pred_label_2c": result.get("pred_label_2"),
            "pred_label_3c": result.get("pred_label_3c"),
            "generated_text": result["generated_text"],
            "prediction_mode": result.get("prediction_mode"),
            "is_correct": result["is_correct"],
            "split": sample["metadata"].get("split", sample.get("split", "unknown")),
        }
        if result.get("candidate_scores") is not None:
            pred_record["candidate_scores"] = result["candidate_scores"]
        
        # 添加元数据
        for key, value in sample["metadata"].items():
            pred_record[key] = value
        
        all_predictions.append(pred_record)
        
        # 定期保存和显存清理
        if (idx + 1) % 50 == 0:
            # 增量写入文件
            with open(predictions_path, 'a', encoding='utf-8') as f:
                for pred in all_predictions[-50:]:
                    f.write(json.dumps(pred, ensure_ascii=False) + '\n')
            
            # 清理缓存防止显存溢出
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    pbar.close()
    
    # 获取预测列表
    if target_mode == "2class":
        pred_labels = [p["pred_label_2c"] for p in all_predictions]
        true_labels = [p["true_label_2c"] for p in all_predictions]
    else:
        pred_labels = [p["pred_label_3c"] for p in all_predictions]
        true_labels = [p["true_label_3c"] for p in all_predictions]

    y_probs = None
    if target_mode in {"2class", "3class"}:
        prob_rows: List[List[float]] = []
        class_set = set(class_names)
        for record in all_predictions:
            candidate_scores = record.get("candidate_scores")
            if not candidate_scores:
                prob_rows = []
                break

            probability_map: Dict[str, float] = {}
            for item in candidate_scores:
                key = item["pred_label_2c"] if target_mode == "2class" else item["pred_label_3c"]
                if key in class_set:
                    probability_map[key] = float(item.get("probability", 0.0))

            if any(name not in probability_map for name in class_names):
                prob_rows = []
                break

            prob_rows.append([probability_map[name] for name in class_names])

        if prob_rows:
            y_probs = np.asarray(prob_rows, dtype=np.float32)

    # 计算指标
    metrics = compute_classification_metrics(
        true_labels,
        pred_labels,
        y_probs=y_probs,
        class_names=class_names,
        subset_name=subset_name,
    )
    
    # 保存剩余预测（最后批次 < 50）
    if len(all_predictions) % 50 != 0:
        with open(predictions_path, 'a', encoding='utf-8') as f:
            for pred in all_predictions[len(all_predictions) - (len(all_predictions) % 50):]:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    if target_mode == "3class":
        folded_metrics = compute_classification_metrics(
            [p["true_label_2c"] for p in all_predictions],
            [fold_3class_label_to_2class(p["pred_label_3c"]) for p in all_predictions],
            y_probs=None,
            class_names=CLASS_NAMES_2,
            subset_name=f"{subset_name}_folded_2class",
        )
        folded_output_dir = os.path.join(output_dir, "folded_2class")
        folded_metrics["prompt_alignment"] = prompt_alignment
        save_metrics(folded_metrics, folded_output_dir)
        metrics["folded_2class"] = folded_metrics
        print_metrics_table(folded_metrics, title=f"结果: {subset_name} (Folded 2Class)")

    # 保存结果
    metrics["prompt_alignment"] = prompt_alignment
    save_metrics(metrics, output_dir)

    # 记录结果
    print_metrics_table(metrics, title=f"结果: {subset_name}")
    
    return metrics, all_predictions


# ============================================================================
# 主入口
# ============================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="在 SLM 缺陷检测上评估 Qwen2-VL (适配重构数据集)"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA 适配器路径（None 表示基础模型）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="模型缓存目录",
    )
    
    # 数据参数
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="清单 JSONL 路径 (兼容新旧格式)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="自定义提示文本",
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
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="eval",
        help="评估名称",
    )
    
    # 评估参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--compute_candidate_scores",
        type=str2bool,
        default=False,
        help="计算候选答案分数（较慢，需要更多显存）",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="auto",
        choices=["auto", "generate", "candidate_score"],
        help="预测模式: auto(推荐), generate(自由生成), candidate_score(闭集候选打分)",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=str2bool,
        default=False,
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
        "--prompt_alignment",
        type=str,
        default="train_validation",
        choices=["train_validation", "inference"],
        help="prompt 对齐方式: train_validation(默认，和训练验证一致) 或 inference(纯推理 prompt)",
    )
    
    return parser.parse_args()


def write_bilingual_eval_note(
    output_dir: str,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
) -> None:
    note_path = os.path.join(output_dir, "evaluation_note.zh_en.md")
    resolved_mode = resolve_prediction_mode(
        target_mode=args.target_mode,
        prediction_mode=args.prediction_mode,
        compute_candidate_scores=args.compute_candidate_scores,
    )
    lines = [
        "# 评估说明 / Evaluation Note",
        "",
        "## 本次设置 / Current Setup",
        "",
        f"- 目标模式 / Target mode: `{args.target_mode}`",
        f"- Token 风格 / Token style: `{args.token_style}`",
        f"- 预测模式 / Prediction mode: `{resolved_mode}`",
        f"- Prompt 对齐 / Prompt alignment: `{args.prompt_alignment}`",
        f"- 样本数 / Number of samples: `{metrics.get('n_samples', 0)}`",
        "",
        "## 指标解读 / Metric Guide",
        "",
        "- `accuracy`: 整体准确率 / overall accuracy.",
        "- `balanced_accuracy`: 各类别召回率的平均，更适合类不均衡场景 / mean recall across classes, better for imbalanced data.",
        "- `macro_f1`: 各类别 F1 的简单平均，最适合作为多分类主指标 / unweighted mean of class F1 scores, recommended primary metric for multi-class runs.",
        "- `weighted_f1`: 按类别样本数加权后的 F1 / support-weighted F1.",
        "- `unknown_rate`: 预测无法解析成合法标签的比例 / rate of predictions that cannot be parsed into a valid label.",
        "- `prompt_alignment=train_validation`: 复用训练验证时的 prompt 前缀截断逻辑 / reuse the same prompt-prefix truncation logic as train-time validation.",
        "",
    ]
    if args.target_mode == "3class":
        lines.extend(
            [
                "## 三分类补充 / 3-Class Extra",
                "",
                "- `folded_2class`: 将 `HEW/LEL` 折叠成 `abnormal` 后得到的异常识别指标 / anomaly-vs-normal metrics obtained by folding `HEW/LEL` into `abnormal`.",
                "",
            ]
        )

    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """主入口。"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['metrics', 'predictions', 'plots', 'logs']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # 设置日志
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    logger.info("="*60)
    logger.info("Qwen2-VL SLM 缺陷检测 - 评估 (适配重构数据集)")
    logger.info("="*60)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"适配器: {args.adapter_path if args.adapter_path else '无（基础模型）'}")
    logger.info(f"清单: {args.manifest_path}")
    logger.info(f"目标模式: {args.target_mode} ({args.token_style})")
    logger.info(
        "预测模式: %s",
        resolve_prediction_mode(
            target_mode=args.target_mode,
            prediction_mode=args.prediction_mode,
            compute_candidate_scores=args.compute_candidate_scores,
        ),
    )
    logger.info(f"Prompt 对齐: {args.prompt_alignment}")
    logger.info(f"输出: {args.output_dir}")
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'eval_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载模型
    logger.info("加载模型...")
    model, processor = load_model_for_evaluation(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bf16=args.bf16,
        fp16=args.fp16,
        cache_dir=args.cache_dir,
    )
    
    # 创建数据集 - 使用重构版参数
    logger.info("加载数据集...")
    if args.prompt_alignment == "train_validation":
        dataset = SLMDataset(
            manifest_path=args.manifest_path,
            processor=processor,
            prompt=args.prompt,
            require_images=False,
            split=None,
            target_mode=args.target_mode,
            token_style=args.token_style,
            task_mode="naive_sft",
        )
    else:
        dataset = SLMDatasetForInference(
            manifest_path=args.manifest_path,
            processor=processor,
            prompt=args.prompt,
            require_images=False,
            target_mode=args.target_mode,
            token_style=args.token_style,
        )
    logger.info(f"加载了 {len(dataset)} 个样本")
    
    # 评估
    logger.info("开始评估...")
    metrics, predictions = evaluate_dataset(
        model=model,
        processor=processor,
        dataset=dataset,
        output_dir=args.output_dir,
        target_mode=args.target_mode,      # 新增
        token_style=args.token_style,      # 新增
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        compute_candidate_scores=args.compute_candidate_scores,
        prediction_mode=args.prediction_mode,
        subset_name=args.eval_name,
        prompt_alignment=args.prompt_alignment,
    )

    write_bilingual_eval_note(args.output_dir, args, metrics)

    logger.info("="*60)
    logger.info("评估完成！")
    logger.info(f"结果保存至: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
