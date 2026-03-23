from __future__ import annotations

import json
import sys
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.path_utils import resolve_path
from qwen_vis_fusion.dataset import CLASS_NAMES_2, CLASS_NAMES_3, resolve_view_keys
from qwen_vis_fusion.model import QwenVisionFusionClassifier

LABEL_ZH = {
    "normal": "正常",
    "HEW": "激光功率过高",
    "LEL": "激光功率过低",
    "abnormal": "异常",
}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cudaerrormemoryallocation" in message


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_fold_from_run_root(run_root: Path) -> Tuple[str, Dict[str, Any]]:
    payload = _load_json(run_root / "run_summary.json")
    fold_result = max(payload["fold_results"], key=lambda item: float(item.get("best_metric", -1.0)))
    return str(fold_result["fold"]), fold_result


def _resolve_artifacts(cfg: Dict[str, Any]) -> Tuple[Path, Path, Dict[str, Any], str]:
    if cfg.get("checkpoint_path") and cfg.get("config_path"):
        checkpoint_path = resolve_path(cfg["checkpoint_path"])
        config_path = resolve_path(cfg["config_path"])
        fold_name = cfg.get("fold", "custom")
        return checkpoint_path, config_path, _load_json(config_path), fold_name

    if not cfg.get("run_root"):
        raise ValueError("Specialized demo config needs either checkpoint_path+config_path or run_root.")

    run_root = resolve_path(cfg["run_root"])
    fold_name = cfg.get("fold", "best")
    if fold_name == "best":
        fold_name, _ = _best_fold_from_run_root(run_root)

    fold_dir = run_root / fold_name
    checkpoint_path = fold_dir / "best.pt"
    config_path = fold_dir / "config.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return checkpoint_path, config_path, _load_json(config_path), fold_name


class SpecializedQwenVisPredictor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.get("device", "auto"))
        self.model = None
        self.processor = None
        self.meta: Dict[str, Any] = {}
        self._current_device = torch.device("cpu")

    def _reset_state(self) -> None:
        self.model = None
        self.processor = None
        self.meta = {}
        self._current_device = torch.device("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> None:
        if self.model is None or self.processor is None or not self.meta:
            try:
                checkpoint_path, config_path, fold_config, fold_name = _resolve_artifacts(self.cfg)
                model_name_or_path = fold_config["model_name_or_path"]
                task_mode = fold_config["task_mode"]
                class_names = CLASS_NAMES_2 if task_mode == "2class" else CLASS_NAMES_3

                self.processor = Qwen2VLProcessor.from_pretrained(model_name_or_path, use_fast=False)
                self.model = QwenVisionFusionClassifier(
                    model_name_or_path=model_name_or_path,
                    num_views=len(resolve_view_keys(fold_config["input_mode"])),
                    num_classes=len(class_names),
                    hidden_dim=int(fold_config.get("hidden_dim", 512)),
                    dropout=float(fold_config.get("dropout", 0.1)),
                    model_dtype=str(fold_config.get("model_dtype", "float16")),
                    unfreeze_last_n_blocks=int(fold_config.get("unfreeze_last_n_blocks", 0)),
                )
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                except TypeError:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.meta = {
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "fold_name": fold_name,
                    "task_mode": task_mode,
                    "input_mode": fold_config["input_mode"],
                    "class_names": class_names,
                    "question_hint": self.cfg.get("question_hint", ""),
                    "runtime_device": "cpu",
                }
                self._current_device = torch.device("cpu")
            except Exception:
                self._reset_state()
                raise

        if self.model is None:
            raise RuntimeError("Specialized model failed to initialize.")

        if self._current_device != self.device:
            try:
                self.model.to(self.device)
                self._current_device = self.device
            except Exception as exc:
                if self.device.type == "cuda" and self.cfg.get("cpu_fallback_on_oom", True) and _is_cuda_oom(exc):
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.model.to("cpu")
                    self._current_device = torch.device("cpu")
                else:
                    self._reset_state()
                    raise

    def offload_to_cpu(self) -> None:
        if self.model is None:
            return
        if self._current_device.type == "cuda":
            self.model.to("cpu")
            self._current_device = torch.device("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    def _prepare_inputs(
        self,
        rgb_view1: Image.Image | None,
        rgb_view2: Image.Image | None,
        ir: Image.Image | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        assert self.processor is not None
        view_map = {
            "rgb_view1": rgb_view1,
            "rgb_view2": rgb_view2,
            "ir": ir,
        }
        view_keys = resolve_view_keys(self.meta["input_mode"])
        images: List[Image.Image] = []
        for key in view_keys:
            image = view_map.get(key)
            if image is None:
                raise ValueError(f"Missing required image for specialized model: {key}")
            images.append(image.convert("RGB"))

        processed = self.processor.image_processor.preprocess(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self._current_device)
        image_grid_thw = processed["image_grid_thw"].to(self._current_device)
        return pixel_values, image_grid_thw, [len(images)]

    @torch.no_grad()
    def predict(
        self,
        rgb_view1: Image.Image | None,
        rgb_view2: Image.Image | None,
        ir: Image.Image | None,
        question: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self.model is not None
        start = time.perf_counter()
        pixel_values, image_grid_thw, sample_image_counts = self._prepare_inputs(rgb_view1, rgb_view2, ir)
        logits, _ = self.model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sample_image_counts=sample_image_counts,
        )
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=-1)[0].detach().cpu().item())
        pred_label = self.meta["class_names"][pred_idx]
        binary_label = "normal" if pred_label == "normal" else "abnormal"
        elapsed = time.perf_counter() - start
        probabilities = {
            label: float(prob)
            for label, prob in zip(self.meta["class_names"], probs)
        }
        return {
            "pred_label": pred_label,
            "binary_label": binary_label,
            "confidence": float(probabilities[pred_label]),
            "probabilities": probabilities,
            "elapsed_seconds": elapsed,
            "question": question.strip() or self.meta.get("question_hint", ""),
            "runtime_device": self._current_device.type,
            **self.meta,
        }

    def format_markdown(self, prediction: Dict[str, Any]) -> str:
        pred_label = prediction.get("pred_label", "unknown")
        binary_label = prediction.get("binary_label", "unknown")
        confidence = float(prediction.get("confidence", 0.0))
        fold_name = prediction.get("fold_name", "unknown")
        input_mode = prediction.get("input_mode", "unknown")
        elapsed = float(prediction.get("elapsed_seconds", 0.0))
        runtime_device = prediction.get("runtime_device", "unknown")
        probabilities = prediction.get("probabilities", {})
        lines = [
            "### 专用模型结论 / Specialized QwenVis",
            "",
            f"- 预测类别 / Predicted class: `{pred_label}` ({LABEL_ZH.get(pred_label, pred_label)})",
            f"- 异常判断 / Binary decision: `{binary_label}` ({LABEL_ZH.get(binary_label, binary_label)})",
            f"- 置信度 / Confidence: `{confidence:.4f}`",
            f"- 使用折次 / Fold: `{fold_name}`",
            f"- 输入模式 / Input mode: `{input_mode}`",
            f"- 运行设备 / Runtime device: `{runtime_device}`",
            f"- 推理耗时 / Inference time: `{elapsed:.2f}s`",
        ]
        lines.extend(
            [
                "",
                "**VQA 风格回答 / VQA-style answer**",
                "",
                (
                    f"基于当前专用缺陷识别模型，我判断该样本更接近 `{pred_label}`"
                    f"（{LABEL_ZH.get(pred_label, pred_label)}）。"
                    f" 若仅做异常识别，则结果为 `{binary_label}`"
                    f"（{LABEL_ZH.get(binary_label, binary_label)}）。"
                ),
            ]
        )
        return "\n".join(lines)

    def format_chat_answer(self, prediction: Dict[str, Any], question: str) -> str:
        pred_label = prediction["pred_label"]
        binary_label = prediction["binary_label"]
        confidence = prediction["confidence"]
        probs = prediction["probabilities"]
        pred_label_zh = LABEL_ZH.get(pred_label, pred_label)
        binary_label_zh = LABEL_ZH.get(binary_label, binary_label)

        if any(token in question.lower() for token in ["正常", "异常", "normal", "abnormal", "safe"]):
            return (
                f"当前专用模型判断该样本为 `{binary_label}`（{binary_label_zh}）。"
                f" 如果映射回三分类，最可能的类别是 `{pred_label}`（{pred_label_zh}），"
                f"置信度约为 `{confidence:.4f}`。"
            )

        if any(token in question.lower() for token in ["hew", "lel", "缺陷", "哪种", "类别", "type"]):
            return (
                f"当前专用模型更倾向于 `{pred_label}`（{pred_label_zh}）。"
                f" 概率分布为 normal={probs.get('normal', 0.0):.4f}, "
                f"HEW={probs.get('HEW', 0.0):.4f}, LEL={probs.get('LEL', 0.0):.4f}。"
            )

        if any(token in question.lower() for token in ["为什么", "依据", "原因", "confidence", "概率", "reason"]):
            return (
                f"这是一个判别式专用模型，所以它不会像通用 VQA 那样自由生成解释。"
                f" 当前最强判断是 `{pred_label}`（{pred_label_zh}），对应置信度 `{confidence:.4f}`；"
                f" 你可以结合右侧通用模型的自然语言回答一起看。"
            )

        return (
            f"基于当前图像，专用模型判断结果为 `{pred_label}`（{pred_label_zh}），"
            f"二分类上属于 `{binary_label}`（{binary_label_zh}），置信度 `{confidence:.4f}`。"
        )
