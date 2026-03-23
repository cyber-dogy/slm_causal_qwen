from __future__ import annotations

import argparse
import html
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.config import load_demo_config
from demo.general_model import GeneralQwenVQAModel
from demo.image_utils import build_montage
from demo.specialized_model import LABEL_ZH, SpecializedQwenVisPredictor

warnings.filterwarnings(
    "ignore",
    message="The 'theme' parameter in the Blocks constructor will be removed.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The 'css' parameter in the Blocks constructor will be removed.*",
    category=DeprecationWarning,
)

@lru_cache(maxsize=2)
def _load_services(config_path: str):
    cfg = load_demo_config(config_path)
    specialized = SpecializedQwenVisPredictor(cfg["specialized"])
    general = GeneralQwenVQAModel(cfg["general"])
    return cfg, specialized, general


def _default_question(config_path: str) -> str:
    cfg, _, _ = _load_services(config_path)
    return cfg["specialized"].get(
        "question_hint",
        "请判断该样本属于 normal、HEW 还是 LEL，并说明是否异常。",
    )


def _history_to_chatbot(history: list[list[str]] | None) -> list[dict[str, str]]:
    chat: list[dict[str, str]] = []
    for user_text, assistant_text in history or []:
        chat.append({"role": "user", "content": str(user_text)})
        chat.append({"role": "assistant", "content": str(assistant_text)})
    return chat


def _default_specialized_markdown() -> str:
    return "### 专用模型结论 / Specialized QwenVis\n\n上传图像并提问后，这里会显示结构化判别结果。"


def _default_general_markdown() -> str:
    return "### 通用模型结论 / General Qwen2-VL\n\n上传图像并提问后，这里会显示通用视觉问答模型的本轮回答摘要。"


def _default_verdict_html() -> str:
    return """
    <div class="verdict-card verdict-neutral">
      <div class="verdict-title">当前最终结论 / Current Final Verdict</div>
      <div class="verdict-main">等待输入图像与提问</div>
      <div class="verdict-sub">
        专用模型会给出最终质检判断，通用模型提供补充解释。这里会在每次推理后同步更新。
      </div>
    </div>
    """


def _probability_help_text() -> str:
    return (
        "专用模型类别概率 / Specialized Probabilities\n"
        "这里单独展示判别式专用模型对 `normal / HEW / LEL` 的闭集概率，"
        "便于和右侧通用 VQA 的自然语言回答分开理解。"
    )


def _truncate_text(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _theme_class_from_label(label: str | None) -> str:
    mapping = {
        "normal": "verdict-normal",
        "HEW": "verdict-hew",
        "LEL": "verdict-lel",
        "abnormal": "verdict-abnormal",
    }
    return mapping.get(str(label), "verdict-neutral")


def _build_final_verdict_html(
    specialized_result: dict | None,
    general_result: dict | None,
) -> str:
    if not specialized_result and not general_result:
        return _default_verdict_html()

    if specialized_result:
        pred_label = specialized_result.get("pred_label", "unknown")
        pred_label_zh = LABEL_ZH.get(pred_label, pred_label)
        binary_label = specialized_result.get("binary_label", "unknown")
        binary_label_zh = LABEL_ZH.get(binary_label, binary_label)
        confidence = float(specialized_result.get("confidence", 0.0))
        theme_class = _theme_class_from_label(pred_label)
        headline = f"{pred_label_zh} / {pred_label}"
        summary = (
            f"建议以专用模型作为最终质检依据。当前样本最可能属于 "
            f"{pred_label_zh}，二分类判断为 {binary_label_zh}，置信度 {confidence:.4f}。"
        )
        general_excerpt = _truncate_text(general_result.get("answer", ""), 150) if general_result else "暂无通用模型辅助解释。"
        return f"""
        <div class="verdict-card {theme_class}">
          <div class="verdict-title">当前最终结论 / Current Final Verdict</div>
          <div class="verdict-main">{html.escape(headline)}</div>
          <div class="verdict-sub">{html.escape(summary)}</div>
          <div class="verdict-grid">
            <div class="verdict-block">
              <div class="verdict-block-title">专用模型 / Specialized</div>
              <div class="verdict-block-body">
                最终类别：<strong>{html.escape(pred_label_zh)}</strong><br/>
                二分类：<strong>{html.escape(binary_label_zh)}</strong><br/>
                置信度：<strong>{confidence:.4f}</strong>
              </div>
            </div>
            <div class="verdict-block">
              <div class="verdict-block-title">通用模型 / General</div>
              <div class="verdict-block-body">{html.escape(general_excerpt)}</div>
            </div>
          </div>
        </div>
        """

    general_excerpt = _truncate_text(general_result.get("answer", ""), 180) if general_result else "暂无结果。"
    return f"""
    <div class="verdict-card verdict-neutral">
      <div class="verdict-title">当前最终结论 / Current Final Verdict</div>
      <div class="verdict-main">专用模型不可用，已回退到通用模型辅助解释</div>
      <div class="verdict-sub">{html.escape(general_excerpt)}</div>
    </div>
    """


@lru_cache(maxsize=1)
def _discover_demo_examples() -> list[tuple[str, dict[str, object]]]:
    manifests: list[Path] = []
    env_root = os.environ.get("SLM_DATA1_PROTOCOL_ROOT")
    if env_root:
        root = Path(env_root)
        manifests.extend(
            [
                root / "quick_dev/test_manifest.jsonl",
                root / "official_cv/fold_01/test_manifest.jsonl",
            ]
        )

    examples: list[tuple[str, dict[str, object]]] = []
    seen_ids: set[str] = set()
    for manifest_path in manifests:
        if not manifest_path.exists():
            continue
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", "unknown"))
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            after = row.get("after", {})
            rgb1 = after.get("rgb_view1")
            rgb2 = after.get("rgb_view2")
            if not rgb1 or not rgb2:
                continue
            label = row.get("label_3c") or row.get("label") or "unknown"
            condition = row.get("condition_uid") or row.get("condition_id") or "unknown"
            layer = row.get("layer_id") or "unknown"
            role = row.get("role") or "unknown"
            display = f"{sample_id} | {LABEL_ZH.get(label, label)} | {condition} | layer {layer} | {role}"
            examples.append(
                (
                    display,
                    {
                        "rgb1": rgb1,
                        "rgb2": rgb2,
                        "ir": after.get("ir"),
                    },
                )
            )
            if len(examples) >= 8:
                return examples
    return examples


@lru_cache(maxsize=2)
def _configured_demo_examples(config_path: str) -> list[tuple[str, dict[str, object]]]:
    cfg, _, _ = _load_services(config_path)
    presets = cfg.get("ui", {}).get("example_presets", []) or []
    examples: list[tuple[str, dict[str, object]]] = []
    for idx, preset in enumerate(presets):
        if not isinstance(preset, dict):
            continue
        rgb1 = preset.get("rgb1")
        rgb2 = preset.get("rgb2")
        if not rgb1 or not rgb2:
            continue
        preset_id = preset.get("id", f"preset_{idx+1}")
        label = str(preset.get("label", "unknown"))
        name = str(preset.get("name", preset_id))
        condition = str(preset.get("condition", "unknown"))
        layer = str(preset.get("layer", "unknown"))
        display = f"{name} | {LABEL_ZH.get(label, label)} | {condition} | layer {layer}"
        examples.append(
            (
                display,
                {
                    "id": preset_id,
                    "name": name,
                    "label": label,
                    "rgb1": rgb1,
                    "rgb2": rgb2,
                    "ir": preset.get("ir"),
                    "question": preset.get("question"),
                },
            )
        )
    return examples


def _get_demo_examples(config_path: str) -> list[tuple[str, dict[str, object]]]:
    configured = _configured_demo_examples(config_path)
    if configured:
        return configured
    return _discover_demo_examples()


def load_selected_example(example_label: str | None, config_path: str):
    if not example_label:
        return None, None, None, _default_question(config_path)
    example_map = dict(_get_demo_examples(config_path))
    payload = example_map.get(example_label)
    if not payload:
        return None, None, None, _default_question(config_path)

    rgb1 = Image.open(str(payload["rgb1"])).convert("RGB")
    rgb2 = Image.open(str(payload["rgb2"])).convert("RGB")
    ir_path = payload.get("ir")
    ir = Image.open(str(ir_path)).convert("RGB") if ir_path else None
    preset_question = str(payload.get("question") or _default_question(config_path))
    return rgb1, rgb2, ir, preset_question


def _append_specialized_history(
    history: list[list[str]] | None,
    question: str,
    answer: str,
    max_rounds: int = 8,
) -> tuple[list[list[str]], list[dict[str, str]]]:
    updated = list(history or [])[-max_rounds:]
    updated.append([question, answer])
    return updated, _history_to_chatbot(updated)


def clear_chat_only(config_path: str):
    return (
        _default_question(config_path),
        _default_verdict_html(),
        _default_specialized_markdown(),
        _default_general_markdown(),
        None,
        [],
        [],
        [],
        [],
    )


def new_conversation(config_path: str):
    return (
        None,
        None,
        None,
        None,
        _default_question(config_path),
        None,
        _default_verdict_html(),
        _default_specialized_markdown(),
        _default_general_markdown(),
        None,
        [],
        [],
        [],
        [],
    )


def run_compare(
    rgb_view1: Image.Image | None,
    rgb_view2: Image.Image | None,
    ir: Image.Image | None,
    question: str,
    config_path: str,
    specialized_history: list[list[str]] | None,
    general_history: list[list[str]] | None,
):
    if rgb_view1 is None or rgb_view2 is None:
        error = "请至少上传 RGB View 1 和 RGB View 2。"
        return (
            None,
            _default_verdict_html(),
            error,
            _default_general_markdown(),
            None,
            _history_to_chatbot(specialized_history),
            _history_to_chatbot(general_history),
            specialized_history or [],
            general_history or [],
            question,
        )

    _cfg, specialized, general = _load_services(config_path)
    question = (question or "").strip() or _default_question(config_path)
    montage = build_montage(rgb_view1, rgb_view2, ir)

    specialized_history = list(specialized_history or [])
    general_history = list(general_history or [])
    specialized_chat = _history_to_chatbot(specialized_history)
    general_chat = _history_to_chatbot(general_history)
    specialized_result = None
    general_result = None
    specialized_md = _default_specialized_markdown()
    general_md = _default_general_markdown()
    specialized_probs = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        specialized_result = specialized.predict(rgb_view1, rgb_view2, ir, question)
        specialized_md = specialized.format_markdown(specialized_result)
        specialized_probs = specialized_result.get("probabilities")
        specialized_answer = specialized.format_chat_answer(specialized_result, question)
        specialized_history, specialized_chat = _append_specialized_history(
            specialized_history,
            question,
            specialized_answer,
        )
    except Exception as exc:
        error_text = f"专用模型失败 / Specialized failed:\n\n`{exc}`"
        specialized_chat = specialized_chat + [{"role": "assistant", "content": error_text}]
        specialized_md = error_text
    finally:
        specialized.offload_to_cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        general_images = [image.convert("RGB") for image in [rgb_view1, rgb_view2, ir] if image is not None]
        general_result = general.answer_with_history(
            general_images,
            question,
            history=general_history,
        )
        general_chat = general_result["chat_display"]
        general_history = general_result["history"]
        general_md = general.format_markdown(general_result)
    except Exception as exc:
        error_text = f"通用模型失败 / General model failed:\n\n`{exc}`"
        general_chat = general_chat + [{"role": "assistant", "content": error_text}]
        general_md = error_text
    finally:
        general.offload_to_cpu()

    return (
        montage,
        _build_final_verdict_html(specialized_result, general_result),
        specialized_md,
        general_md,
        specialized_probs,
        specialized_chat,
        general_chat,
        specialized_history,
        general_history,
        "",
    )


def create_ui(config_path: str) -> gr.Blocks:
    cfg, _, _ = _load_services(config_path)
    title = cfg["ui"].get("title", "SLM Defect VQA Comparator")

    custom_css = """
    :root {
      color-scheme: light dark;
    }
    .gradio-container {
      max-width: 1560px !important;
      margin: 0 auto;
      font-family: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(25, 118, 210, 0.08), transparent 26%),
        linear-gradient(180deg, #f3f6f9 0%, #edf2f7 100%);
    }
    .app-shell {max-width: 1540px; margin: 0 auto;}
    .hero {
      background:
        linear-gradient(135deg, rgba(10,24,38,0.98) 0%, rgba(26,46,67,0.98) 100%),
        repeating-linear-gradient(
          135deg,
          rgba(255,255,255,0.04) 0px,
          rgba(255,255,255,0.04) 1px,
          transparent 1px,
          transparent 12px
        );
      color: white;
      border-radius: 20px;
      padding: 26px 30px;
      margin-bottom: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
    }
    .hero p {color: rgba(255,255,255,0.88);}
    .soft-card {
      border: 1px solid rgba(25, 55, 85, 0.12);
      border-radius: 18px;
      background: rgba(255,255,255,0.88);
      color: var(--body-text-color);
      box-shadow: 0 12px 28px rgba(12, 26, 42, 0.08);
      backdrop-filter: blur(8px);
    }
    .soft-card * {
      color: var(--body-text-color) !important;
    }
    .soft-card .md,
    .soft-card .prose,
    .soft-card .prose * {
      color: var(--body-text-color) !important;
    }
    .chat-shell {
      border: 1px solid rgba(25, 55, 85, 0.10);
      border-radius: 18px;
      background: rgba(255,255,255,0.9);
      padding: 6px;
      box-shadow: 0 12px 28px rgba(12, 26, 42, 0.08);
    }
    .chat-shell [data-testid="chatbot"] {
      background: transparent !important;
    }
    .chat-shell .message,
    .chat-shell .message * {
      color: var(--body-text-color) !important;
    }
    .chat-shell .message.user {
      background: color-mix(in srgb, var(--color-accent-soft) 70%, transparent);
    }
    .chat-shell .message.bot {
      background: color-mix(in srgb, var(--block-background-fill) 88%, var(--body-background-fill));
    }
    .section-title {
      margin: 0 0 10px 0;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #16324a;
      font-size: 1.04rem;
    }
    .row-card {
      border: 1px solid rgba(25, 55, 85, 0.10);
      border-radius: 20px;
      background: rgba(255,255,255,0.9);
      padding: 16px;
      box-shadow: 0 12px 28px rgba(12, 26, 42, 0.08);
      backdrop-filter: blur(8px);
    }
    .status-strip {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 10px 0 0 0;
    }
    .status-pill {
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(255,255,255,0.18);
      color: white;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.92rem;
    }
    .input-tip {
      color: var(--body-text-color-subdued);
      font-size: 0.95rem;
    }
    .panel-note {
      color: var(--body-text-color-subdued);
      font-size: 0.93rem;
      margin-top: 6px;
    }
    .prob-card {
      border: 1px solid rgba(25, 55, 85, 0.10);
      border-radius: 20px;
      background: rgba(255,255,255,0.9);
      padding: 14px 16px 10px 16px;
      box-shadow: 0 12px 28px rgba(12, 26, 42, 0.08);
    }
    .verdict-card {
      border-radius: 22px;
      padding: 18px 20px;
      border: 1px solid rgba(25, 55, 85, 0.10);
      box-shadow: 0 14px 34px rgba(12, 26, 42, 0.10);
      background: rgba(255,255,255,0.94);
      margin-bottom: 4px;
    }
    .verdict-title {
      font-size: 0.94rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #4a6175;
      margin-bottom: 10px;
    }
    .verdict-main {
      font-size: 2rem;
      line-height: 1.15;
      font-weight: 800;
      color: #0f2740;
      margin-bottom: 8px;
    }
    .verdict-sub {
      font-size: 0.98rem;
      color: #486173;
      margin-bottom: 14px;
    }
    .verdict-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .verdict-block {
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(245,248,251,0.95);
      border: 1px solid rgba(25,55,85,0.08);
    }
    .verdict-block-title {
      font-size: 0.86rem;
      font-weight: 700;
      color: #51687c;
      margin-bottom: 6px;
    }
    .verdict-block-body {
      font-size: 0.95rem;
      color: #193047;
      line-height: 1.55;
    }
    .verdict-normal {
      border-left: 7px solid #1e8e5a;
    }
    .verdict-hew {
      border-left: 7px solid #c96d1a;
    }
    .verdict-lel {
      border-left: 7px solid #9a7b00;
    }
    .verdict-abnormal {
      border-left: 7px solid #b42318;
    }
    .verdict-neutral {
      border-left: 7px solid #36536b;
    }
    .specialized-card {
      border-left: 6px solid #1e5aa7 !important;
      background: linear-gradient(180deg, rgba(236,244,255,0.92) 0%, rgba(255,255,255,0.95) 100%) !important;
    }
    .general-card {
      border-left: 6px solid #9a6700 !important;
      background: linear-gradient(180deg, rgba(255,248,230,0.92) 0%, rgba(255,255,255,0.95) 100%) !important;
    }
    .specialized-chat-shell {
      border-left: 6px solid #1e5aa7 !important;
    }
    .general-chat-shell {
      border-left: 6px solid #9a6700 !important;
    }
    .specialized-chat-shell .message.bot {
      background: rgba(30, 90, 167, 0.08) !important;
    }
    .general-chat-shell .message.bot {
      background: rgba(154, 103, 0, 0.08) !important;
    }
    .dark .hero {
      background: linear-gradient(135deg, #0a1627 0%, #17324f 100%);
    }
    .dark .gradio-container {
      background:
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.08), transparent 28%),
        linear-gradient(180deg, #0e1620 0%, #111827 100%);
    }
    .dark .soft-card,
    .dark .chat-shell,
    .dark .row-card,
    .dark .prob-card {
      background: rgba(18, 24, 34, 0.94) !important;
      border-color: rgba(255, 255, 255, 0.12) !important;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
    }
    .dark .soft-card *,
    .dark .chat-shell *,
    .dark .row-card *,
    .dark .prob-card *,
    .dark .input-tip,
    .dark .panel-note {
      color: #edf2f7 !important;
    }
    .dark .section-title {
      color: #e8eef6 !important;
    }
    .dark .verdict-card {
      background: rgba(18, 24, 34, 0.96) !important;
      border-color: rgba(255,255,255,0.12) !important;
      box-shadow: 0 14px 34px rgba(0,0,0,0.35);
    }
    .dark .verdict-main {
      color: #f8fbff !important;
    }
    .dark .verdict-title,
    .dark .verdict-sub,
    .dark .verdict-block-title,
    .dark .verdict-block-body {
      color: #dce6f0 !important;
    }
    .dark .verdict-block {
      background: rgba(255,255,255,0.04) !important;
      border-color: rgba(255,255,255,0.08) !important;
    }
    .dark .chat-shell .message.user {
      background: rgba(56, 189, 248, 0.16) !important;
    }
    .dark .chat-shell .message.bot {
      background: rgba(255, 255, 255, 0.06) !important;
    }
    """

    with gr.Blocks(title=title, css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            f"""
            <div class="app-shell">
              <div class="hero">
                <h1 style="margin:0;">{title}</h1>
                <p style="margin:8px 0 0 0;">
                  左侧是专用因果 QwenVis 缺陷识别模型，右侧是通用 Qwen2-VL VQA 回答。
                  用同一组图像和同一个问题，直观看两者的差异。
                </p>
                <div class="status-strip">
                  <span class="status-pill">最佳专用模型：QwenVisFusion + rgb_dual + causal</span>
                  <span class="status-pill">类别映射：normal=正常，HEW=激光功率过高，LEL=激光功率过低</span>
                  <span class="status-pill">支持连续追问</span>
                </div>
              </div>
            </div>
            """
        )

        state_config = gr.State(config_path)
        state_specialized_history = gr.State([])
        state_general_history = gr.State([])
        example_choices = [item[0] for item in _get_demo_examples(config_path)]

        gr.HTML("<div class='section-title'>图像输入与拼图 / Inputs and Montage</div>")
        gr.Markdown(
            "<div class='input-tip'>第一行只负责看图。上传同一零件的多视角图像，右侧 montage 会作为通用 VQA 的统一输入。</div>"
        )
        with gr.Row(equal_height=True, elem_classes=["row-card"]):
            rgb1 = gr.Image(label="RGB View 1", type="pil", sources=["upload", "clipboard"], height=240)
            rgb2 = gr.Image(label="RGB View 2", type="pil", sources=["upload", "clipboard"], height=240)
            ir = gr.Image(label="IR After (Optional)", type="pil", sources=["upload", "clipboard"], height=240)
            montage = gr.Image(label="Montage for General VQA", type="pil", height=240)

        gr.HTML("<div class='section-title' style='margin-top:16px;'>提问与控制 / Controls</div>")
        with gr.Column(elem_classes=["row-card"]):
            with gr.Row():
                example_selector = gr.Dropdown(
                    label="示例样本 / Quick Example",
                    choices=example_choices,
                    value=example_choices[0] if example_choices else None,
                    allow_custom_value=False,
                    scale=5,
                )
                example_btn = gr.Button("加载示例 / Load Example", scale=1)
            question = gr.Textbox(
                label="问题 / Question",
                value=_default_question(config_path),
                lines=3,
                placeholder="可以先问分类，再继续追问“为什么这样判断？”、“异常程度如何？” 等问题。",
            )
            with gr.Row():
                ask_btn = gr.Button("开始对比 / Compare", variant="primary", size="lg")
                clear_btn = gr.Button("清空对话 / Clear Chat")
                new_chat_btn = gr.Button("新对话 / New Conversation", variant="secondary")
            with gr.Row():
                q1 = gr.Button("设备是否正常？", size="sm")
                q2 = gr.Button("属于 HEW 还是 LEL？", size="sm")
                q3 = gr.Button("请给出专业质检判断。", size="sm")
            gr.Markdown(
                "<div class='panel-note'>连续问答会保留当前会话上下文。若你更换了图像，建议点击“新对话”，避免把旧图上下文混进来。</div>"
            )

        verdict_html = gr.HTML(_default_verdict_html())

        gr.HTML("<div class='section-title' style='margin-top:16px;'>回答对比 / Side-by-side Comparison</div>")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["row-card", "specialized-panel"]):
                    gr.Markdown(
                        "<div class='panel-note'>专用模型更聚焦 `normal / HEW / LEL / abnormal` 的结构化质检判断。</div>"
                    )
                    specialized_md = gr.Markdown(value=_default_specialized_markdown(), elem_classes=["soft-card", "specialized-card"])

            with gr.Column(scale=1):
                with gr.Column(elem_classes=["row-card", "general-panel"]):
                    gr.Markdown(
                        "<div class='panel-note'>通用模型保留开放式视觉问答能力，更适合补充自然语言解释、现象描述和工艺推测。</div>"
                    )
                    general_md = gr.Markdown(value=_default_general_markdown(), elem_classes=["soft-card", "general-card"])

        with gr.Column(elem_classes=["prob-card"]):
            gr.Markdown(f"**{_probability_help_text()}**")
            probs = gr.Label(label="Specialized Probabilities")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["row-card", "specialized-panel"]):
                    gr.HTML("<div class='section-title'>专用模型连续问答 / Specialized Chat</div>")
                    with gr.Group(elem_classes=["chat-shell", "specialized-chat-shell"]):
                        specialized_chat = gr.Chatbot(
                            height=420,
                            show_label=False,
                            type="messages",
                        )
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["row-card", "general-panel"]):
                    gr.HTML("<div class='section-title'>通用模型连续问答 / General Chat</div>")
                    with gr.Group(elem_classes=["chat-shell", "general-chat-shell"]):
                        general_chat = gr.Chatbot(
                            height=420,
                            show_label=False,
                            type="messages",
                        )

        ask_btn.click(
            fn=run_compare,
            inputs=[rgb1, rgb2, ir, question, state_config, state_specialized_history, state_general_history],
            outputs=[montage, verdict_html, specialized_md, general_md, probs, specialized_chat, general_chat, state_specialized_history, state_general_history, question],
        )
        question.submit(
            fn=run_compare,
            inputs=[rgb1, rgb2, ir, question, state_config, state_specialized_history, state_general_history],
            outputs=[montage, verdict_html, specialized_md, general_md, probs, specialized_chat, general_chat, state_specialized_history, state_general_history, question],
        )
        example_btn.click(
            fn=load_selected_example,
            inputs=[example_selector, state_config],
            outputs=[rgb1, rgb2, ir, question],
        )

        clear_btn.click(
            fn=clear_chat_only,
            inputs=[state_config],
            outputs=[question, verdict_html, specialized_md, general_md, probs, specialized_chat, general_chat, state_specialized_history, state_general_history],
        )
        new_chat_btn.click(
            fn=new_conversation,
            inputs=[state_config],
            outputs=[rgb1, rgb2, ir, example_selector, question, montage, verdict_html, specialized_md, general_md, probs, specialized_chat, general_chat, state_specialized_history, state_general_history],
        )
        q1.click(lambda: "设备是否正常？", outputs=question)
        q2.click(lambda: "请判断它属于 HEW 还是 LEL？", outputs=question)
        q3.click(lambda: "请给出专业质检判断，并说明是否异常。", outputs=question)

        with gr.Accordion("说明 / Notes", open=False):
            gr.Markdown(
                """
                - 专用模型使用 `QwenVisFusion + rgb_dual + causal consistency` 最佳实验折次。
                - 通用模型使用单图 VQA 方式，因此会先把多张输入图拼接成一张 montage。
                - 专用模型更适合回答 `normal / HEW / LEL / abnormal` 这类质检问题。
                - 如果显存吃紧，可以把 `configs/demo.local.yaml` 里的 `general.device` 或 `specialized.device` 调成 `cpu`。
                """
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side demo for specialized QwenVis and general Qwen2-VL.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/demo.local.yaml"))
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cfg = load_demo_config(args.config)
    ui_cfg = cfg["ui"]
    demo = create_ui(args.config)
    demo.launch(
        server_name=args.host or ui_cfg.get("host", "0.0.0.0"),
        server_port=args.port or int(ui_cfg.get("port", 7861)),
        share=bool(args.share or ui_cfg.get("share", False)),
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
