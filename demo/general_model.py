from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.path_utils import resolve_path

warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.bfloat16)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cudaerrormemoryallocation" in message


class GeneralQwenVQAModel:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.get("device", "auto"))
        self.model = None
        self.processor = None
        self._current_device = torch.device("cpu")

    def _reset_state(self) -> None:
        self.model = None
        self.processor = None
        self._current_device = torch.device("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> None:
        if self.model is None or self.processor is None:
            try:
                model_path = resolve_path(self.cfg["model_name_or_path"])
                torch_dtype = _resolve_dtype(self.cfg.get("model_dtype", "bfloat16"))
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(model_path),
                    dtype=torch_dtype,
                    device_map=None,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                self.model.eval()
                self.processor = Qwen2VLProcessor.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    trust_remote_code=True,
                    use_fast=False,
                )
                self._current_device = torch.device("cpu")
            except Exception:
                self._reset_state()
                raise

        if self.model is None:
            raise RuntimeError("General model failed to initialize.")

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

    def _normalize_images(self, image_or_images: Image.Image | Sequence[Image.Image]) -> List[Image.Image]:
        if isinstance(image_or_images, Image.Image):
            return [image_or_images.convert("RGB")]
        images = [image.convert("RGB") for image in image_or_images if image is not None]
        if not images:
            raise ValueError("General model requires at least one image.")
        return images

    def _build_general_prompt(self, question: str) -> str:
        user_question = question.strip() or "请描述这组图像并判断是否存在缺陷。"
        analysis_instruction = self.cfg.get(
            "analysis_instruction",
            (
                "你现在扮演通用工业多视角视觉助手，而不是闭集缺陷分类器。"
                "请优先利用多张图像做开放式观察与解释。"
                "回答时先概述可见现象，再给出谨慎判断。"
                "如果用户要求在 normal、HEW、LEL 中选择，请先说明证据，"
                "不要只给标签；若把握不足，可以明确说明不确定。"
            ),
        )
        answer_style = self.cfg.get(
            "answer_style_instruction",
            (
                "请尽量用简洁、专业、可核查的语言回答。"
                "优先描述图像中真正可见的现象，例如边缘、纹理、热分布、对称性、局部异常区域等，"
                "避免空泛套话。"
            ),
        )
        lower_question = user_question.lower()
        asks_closed_set = any(token in lower_question for token in ["normal", "hew", "lel", "异常", "正常", "激光功率过高", "激光功率过低"])
        closed_set_instruction = ""
        if asks_closed_set:
            closed_set_instruction = (
                "如果用户要求在 normal、HEW、LEL 中进行映射，请不要一上来直接给标签。"
                "只有当图像证据非常明确时，才在最后给出“候选映射”。"
                "如果证据不足，请明确写“候选映射：无法仅凭当前图像可靠确定”，这比强行分类更好。"
            )
        return (
            f"{analysis_instruction}\n"
            f"{answer_style}\n"
            f"{closed_set_instruction}\n"
            "请按下面结构回答：\n"
            "1. 可见现象\n"
            "2. 初步判断\n"
            "3. 依据与不确定性\n"
            "4. 候选映射（如果无法可靠映射，请明确说明）\n"
            f"用户问题：{user_question}"
        )

    def _build_messages(
        self,
        images: List[Image.Image],
        question: str,
        history: List[List[str]] | None = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.cfg.get(
                    "system_prompt",
                    "你是一个工业视觉问答助手，请基于输入图像给出谨慎、专业、简洁的分析。",
                ),
            }
        ]
        for user_text, assistant_text in history or []:
            messages.append({"role": "user", "content": str(user_text)})
            messages.append({"role": "assistant", "content": str(assistant_text)})
        user_content: List[Dict[str, Any]] = []
        for idx, image in enumerate(images, start=1):
            user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": f"图像 {idx} / Image {idx}"})
        messages.append(
            {
                "role": "user",
                "content": user_content + [{"type": "text", "text": question}],
            }
        )
        return messages

    def _history_to_chatbot(self, history: List[List[str]]) -> List[Dict[str, str]]:
        chat: List[Dict[str, str]] = []
        for user_text, assistant_text in history:
            chat.append({"role": "user", "content": str(user_text)})
            chat.append({"role": "assistant", "content": str(assistant_text)})
        return chat

    @torch.no_grad()
    def answer(self, image_or_images: Image.Image | Sequence[Image.Image], question: str) -> Dict[str, Any]:
        self.load()
        assert self.model is not None and self.processor is not None
        images = self._normalize_images(image_or_images)
        prompt = self._build_general_prompt(question)
        messages = self._build_messages(images=images, question=prompt, history=[])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self._current_device)
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.get("max_new_tokens", 256)),
            do_sample=bool(self.cfg.get("do_sample", False)),
            use_cache=True,
        )
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - start
        return {
            "answer": answer,
            "elapsed_seconds": elapsed,
            "question": question.strip() or "请描述这组图像并判断是否存在缺陷。",
            "model_name_or_path": str(self.cfg["model_name_or_path"]),
            "runtime_device": self._current_device.type,
            "num_images": len(images),
        }

    @torch.no_grad()
    def answer_with_history(
        self,
        image_or_images: Image.Image | Sequence[Image.Image],
        question: str,
        history: List[List[str]] | None = None,
        max_history_rounds: int = 8,
    ) -> Dict[str, Any]:
        self.load()
        assert self.model is not None and self.processor is not None
        images = self._normalize_images(image_or_images)
        raw_question = question.strip() or "请描述这组图像并判断是否存在缺陷。"
        prompt = self._build_general_prompt(raw_question)
        trimmed_history = list(history or [])[-max_history_rounds:]
        messages = self._build_messages(images=images, question=prompt, history=trimmed_history)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self._current_device)
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.get("max_new_tokens", 256)),
            do_sample=bool(self.cfg.get("do_sample", False)),
            use_cache=True,
        )
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - start
        updated_history = trimmed_history + [[raw_question, answer]]
        return {
            "answer": answer,
            "elapsed_seconds": elapsed,
            "question": raw_question,
            "history": updated_history,
            "chat_display": self._history_to_chatbot(updated_history),
            "model_name_or_path": str(self.cfg["model_name_or_path"]),
            "runtime_device": self._current_device.type,
            "num_images": len(images),
        }

    def format_markdown(self, result: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "### 通用模型回答 / General Qwen2-VL",
                "",
                f"- 运行设备 / Runtime device: `{result.get('runtime_device', 'unknown')}`",
                f"- 输入图像数 / Num images: `{result.get('num_images', 'unknown')}`",
                f"- 推理耗时 / Inference time: `{result['elapsed_seconds']:.2f}s`",
                "",
                "**回答 / Answer**",
                "",
                result["answer"] or "_No response._",
            ]
        )
