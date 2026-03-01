import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class EvaluationResult:
    question: str
    ground_truth: Union[str, List[float]]
    model_response: str
    score: float

class EvaluatorFactory:
    @staticmethod
    def create_evaluator(dataset: str) -> "BaseEvaluator":
        if dataset == "chattime":
            return ChatTimeEvaluator()
        raise ValueError(f"Unsupported dataset: {dataset}")

class BaseEvaluator:
    """Dataset evaluator interface."""

    def evaluate_file(self, inference_jsonl: str, results_jsonl: str, summary_csv: str) -> None:
        raise NotImplementedError

class ChatTimeEvaluator(BaseEvaluator):
    """
    ChatTime: parse (a)/(b)/(c) from model response, compare with ground_truth.
    Writes per-sample results to jsonl and a grouped summary csv.
    """

    _OPT_RE = re.compile(r"\(\s*([abc])\s*\)|\b([abc])\b", re.IGNORECASE)
    
    def _normalize_gt(self, gt: Any) -> Optional[str]:
        s = str(gt).strip().lower()
        m = re.search(r"\(\s*([abc])\s*\)", s)
        if m:
            return f"({m.group(1)})"
        if s in {"a", "b", "c"}:
            return f"({s})"
        return s if s else None

    def _extract_pred(self, resp: Any) -> Optional[str]:
        s = str(resp).strip().lower()
        m = self._OPT_RE.search(s)
        if not m:
            return None
        letter = (m.group(1) or m.group(2) or "").lower()
        return f"({letter})" if letter in {"a", "b", "c"} else None

    def evaluate_file(self, inference_jsonl: str, results_jsonl: str, summary_csv: str) -> None:
        inf_path = Path(inference_jsonl)
        out_path = Path(results_jsonl)
        csv_path = Path(summary_csv)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        n_ok, n_fail = 0, 0

        with open(inf_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                data = json.loads(line)
                meta = data.get("metadata", {}) or {}

                question = data.get("question", "")
                gt_raw = data.get("ground_truth", "")
                resp = data.get("model_response", "")

                gt = self._normalize_gt(gt_raw)
                pred = self._extract_pred(resp)

                score = 1.0 if (pred is not None and gt is not None and pred == gt) else 0.0
                n_ok += int(score == 1.0)
                n_fail += int(score == 0.0)

                er = EvaluationResult(
                    question=question,
                    ground_truth=gt_raw,
                    model_response=str(resp),
                    score=score,
                )

                record = asdict(er)
                record["id"] = data.get("id", i)
                record["pred"] = pred
                record["dataset"] = data.get("dataset", "chattime")
                record["metadata"] = meta
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                rows.append(
                    {
                        "task": meta.get("task", "unknown"),
                        "size": int(meta.get("size", -1)) if str(meta.get("size", "")).isdigit() else meta.get("size", -1),
                        "correct": int(score == 1.0),
                    }
                )

        df = pd.DataFrame(rows)
        overall = float(df["correct"].mean()) if len(df) else 0.0

        # Summary csv
        if len(df):
            summary = df.groupby(["task", "size"])["correct"].mean().reset_index()
            summary.rename(columns={"correct": "accuracy"}, inplace=True)
        else:
            summary = pd.DataFrame(columns=["task", "size", "accuracy"])

        summary.to_csv(csv_path, index=False)

        print("Evaluation")
        print(f"  input   : {inf_path}")
        print(f"  results : {out_path}")
        print(f"  summary : {csv_path}")
        print(f"  overall : {overall:.4f} ({n_ok}/{n_ok+n_fail})")