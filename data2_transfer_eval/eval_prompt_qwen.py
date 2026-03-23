#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RUN_ROOTS = [
    "/home/gjw/code/Qwen-SLM/runs/data1_official_cv_2class_baseline",
    "/home/gjw/code/Qwen-SLM/runs/data1_official_cv_3class_baseline",
    "/home/gjw/code/Qwen-SLM/runs/data1_official_cv_3class_causal",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_folds(run_root: Path) -> List[Path]:
    return sorted(path for path in run_root.glob("fold_*") if path.is_dir())


def eval_output_dir(base_output_root: Path, run_root: Path, fold_name: str, stage: str) -> Path:
    return base_output_root / run_root.name / fold_name / stage / "transfer_test"


def metrics_from(output_dir: Path) -> Dict[str, Any]:
    return load_json(output_dir / "metrics_summary.json")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def base_cache_key(config: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        config["model_name_or_path"],
        config["target_mode"],
        config.get("token_style", "letters"),
    )


def run_eval(
    *,
    model_name_or_path: str,
    adapter_path: Path | None,
    manifest_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
) -> None:
    cmd = [
        sys.executable,
        "src/evaluate_qwen2vl_slm.py",
        "--model_name_or_path",
        model_name_or_path,
        "--manifest_path",
        str(manifest_path),
        "--output_dir",
        str(output_dir),
        "--eval_name",
        "data2_transfer",
        "--target_mode",
        config["target_mode"],
        "--token_style",
        config.get("token_style", "letters"),
        "--batch_size",
        "1",
        "--max_new_tokens",
        str(config.get("max_new_tokens", 4)),
        "--compute_candidate_scores",
        str(config.get("compute_candidate_scores", True)).lower(),
        "--prediction_mode",
        str(config.get("prediction_mode", "auto")),
        "--prompt_alignment",
        "train_validation",
        "--load_in_4bit",
        str(config.get("load_in_4bit", True)).lower(),
        "--load_in_8bit",
        str(config.get("load_in_8bit", False)).lower(),
        "--bf16",
        str(config.get("bf16", True)).lower(),
        "--fp16",
        str(config.get("fp16", False)).lower(),
    ]
    if adapter_path is not None:
        cmd.extend(["--adapter_path", str(adapter_path)])
    if config.get("cache_dir"):
        cmd.extend(["--cache_dir", str(config["cache_dir"])])
    if config.get("prompt"):
        cmd.extend(["--prompt", str(config["prompt"])])

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prompt-Qwen runs on data2 transfer.")
    parser.add_argument("--manifest_path", type=str, default="/home/gjw/code/SLM_data/processed_qwen_data2_transfer/transfer_test_manifest.jsonl")
    parser.add_argument("--run_roots", nargs="+", default=DEFAULT_RUN_ROOTS)
    parser.add_argument("--output_root", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/prompt_qwen")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cache_dirs: Dict[Tuple[str, str, str], Path] = {}
    all_records: List[Dict[str, Any]] = []

    for run_root_str in args.run_roots:
        run_root = Path(run_root_str).resolve()
        for fold_dir in discover_folds(run_root):
            config = load_json(fold_dir / "config" / "train_config.json")
            model_name = config["model_name_or_path"]
            key = base_cache_key(config)

            before_out = eval_output_dir(output_root, run_root, fold_dir.name, "before")
            if key not in base_cache_dirs:
                shared_dir = output_root / "_shared_base" / f"{config['target_mode']}_{config.get('token_style', 'letters')}"
                if not (args.skip_existing and (shared_dir / "metrics_summary.json").exists()):
                    run_eval(
                        model_name_or_path=model_name,
                        adapter_path=None,
                        manifest_path=manifest_path,
                        output_dir=shared_dir,
                        config=config,
                    )
                base_cache_dirs[key] = shared_dir
            if not (args.skip_existing and (before_out / "metrics_summary.json").exists()):
                copy_tree(base_cache_dirs[key], before_out)

            after_out = eval_output_dir(output_root, run_root, fold_dir.name, "after")
            if not (args.skip_existing and (after_out / "metrics_summary.json").exists()):
                run_eval(
                    model_name_or_path=model_name,
                    adapter_path=fold_dir / "adapter_best",
                    manifest_path=manifest_path,
                    output_dir=after_out,
                    config=config,
                )

            for stage, out_dir in [("before", before_out), ("after", after_out)]:
                metrics = metrics_from(out_dir)
                folded_macro_f1 = None
                if "folded_2class" in metrics:
                    folded_macro_f1 = float(metrics["folded_2class"]["macro_f1"])
                all_records.append(
                    {
                        "family": "prompt_qwen",
                        "experiment_name": run_root.name,
                        "task_mode": config["target_mode"],
                        "method": "causal" if "causal" in run_root.name else "baseline",
                        "backbone_name": "qwen2vl_prompt",
                        "input_mode": "triple_view",
                        "fusion_type": "prompt_candidate_score",
                        "stage": stage,
                        "split": "transfer_test",
                        "fold": fold_dir.name,
                        "accuracy": float(metrics["accuracy"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "weighted_f1": float(metrics["weighted_f1"]),
                        "folded_2class_macro_f1": folded_macro_f1,
                        "run_root": str(run_root),
                        "output_dir": str(out_dir),
                    }
                )
            print(f"[prompt_qwen] evaluated {run_root.name}/{fold_dir.name}")

    write_json(output_root / "prompt_qwen_summary.json", all_records)
    print("=" * 72)
    print("prompt_qwen data2 transfer evaluation finished / prompt_qwen data2 transfer 评估完成")
    print(f"Output / 输出: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
