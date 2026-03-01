import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import yaml

from models import ModelFactory
from dataloader import DataLoader
from evaluator import EvaluatorFactory


class EvaluationPipeline:
    """Inference -> evaluator."""
    
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.model = None
        self.loader = None

        self.model_name = self.cfg["model"]["model_name"]
        self.dataset_name = ""

        paths = self.cfg.get("paths", {}) or {}
        self.inference_dir = Path(paths.get("inference_dir", "inference_results"))
        self.results_dir = Path(paths.get("results_dir", "results"))

    def _safe_name(self, s: str) -> str:
        return s.replace("/", "_").replace(":", "_").replace("\\", "_")

    def load_model(self) -> None:
        self.model = ModelFactory.create_model(**self.cfg["model"])

    def load_data(self, dataset: str) -> None:
        self.dataset_name = dataset
        dcfg = self.cfg["datasets"][dataset]
        self.loader = DataLoader(dataset=dataset, **dcfg)

    def _inference_path(self) -> Path:
        out_dir = self.inference_dir / self._safe_name(self.model_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{self._safe_name(self.model_name)}-{self.dataset_name}.jsonl"

    def _results_paths(self) -> tuple[Path, Path]:
        out_dir = self.results_dir / self._safe_name(self.model_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = f"{self._safe_name(self.model_name)}-{self.dataset_name}"
        return out_dir / f"{base}_results.jsonl", out_dir / f"{base}_summary.csv"

    def run_inference(self) -> str:
        assert self.model is not None and self.loader is not None

        out_path = self._inference_path()
        total = len(self.loader)

        print("Inference")
        print(f"  model   : {self._safe_name(self.model_name)}")
        print(f"  dataset : {self.dataset_name} ({total} samples)")
        print(f"  output  : {out_path}")

        ok, fail = 0, 0

        with open(out_path, "w", encoding="utf-8") as f:
            for i in tqdm(range(total), desc="infer", unit="sample", dynamic_ncols=True):
                sample = self.loader[i]
                try:
                    resp = self.model.generate(sample["full_question"])
                    ok += 1
                except Exception as e:
                    fail += 1
                    tqdm.write(f"[{i+1}/{total}] FAIL: {e}")
                    continue
                    
                record = {
                    "id": i,
                    "dataset": self.dataset_name,
                    "question": sample["question"],
                    "timeseries": sample["timeseries"],
                    "ground_truth": sample["answer"],
                    "model_response": resp,
                    "metadata": sample["metadata"],
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Done: ok={ok}, fail={fail}")
        return str(out_path)

    def run_eval(self, inference_jsonl: str) -> None:
        results_jsonl, summary_csv = self._results_paths()
        
        evaluator = EvaluatorFactory.create_evaluator(self.dataset_name)
        evaluator.evaluate_file(
            inference_jsonl=inference_jsonl,
            results_jsonl=str(results_jsonl),
            summary_csv=str(summary_csv),
        )

    def run(self, dataset: str) -> None:
        self.load_model()
        self.load_data(dataset)

        inf_path = self.run_inference()
        self.run_eval(inf_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evaluation pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    args = parser.parse_args()

    EvaluationPipeline(args.config).run(dataset=args.dataset)