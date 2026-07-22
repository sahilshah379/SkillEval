"""Entrance script.

    python main.py --dataset seedance
    python main.py --dataset kimi

Loads the dataset's mapping.json + top-N issue categories from config.yaml, then
runs the agent loop for every (sample, issue) pair and writes verdicts to output_dir.
"""
import argparse
import json
from pathlib import Path

import yaml

from agent.runner import AgentRunner

CONFIG_PATH = "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Agentic I2V quality evaluation")
    parser.add_argument("--dataset", required=True, help="dataset/model to run on (e.g. kimi, seedance)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})
    if args.dataset not in datasets:
        raise KeyError(f"Unknown dataset '{args.dataset}'. Available: {sorted(datasets)}")
    dataset = datasets[args.dataset]

    tax = config.get("taxonomy", {})
    categories = tax.get("categories", [])[: tax.get("top_n", 5)]

    # samples: [{"prompt": str, "input": {"images": [...], "audio": [...], "video": [...]}, "output": [url, ...]}, ...]
    samples = json.loads((Path(dataset["path"]) / "mapping.json").read_text())
    print(f"[{args.dataset}] {len(samples)} samples x {len(categories)} issue categories")

    runner = AgentRunner(config.get("agent", {}))
    out_dir = Path(config.get("output_dir", "results")) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        out_path = out_dir / f"sample_{i:03d}.json"
        if out_path.exists():
            print(f"  sample {i}: already done, skipping (delete {out_path} to redo)")
            continue
        if not sample["output"]:
            print(f"  sample {i}: no output video, skipping")
            continue
        video = sample["output"][0]  # the generated video under evaluation
        verdicts = []
        for issue in categories:
            print(f"  sample {i} :: {issue['name']}")
            try:
                verdict = runner.run(video, issue, sample=sample).to_dict()
            except Exception as e:
                print(f"    failed: {e}")
                verdict = {"video": video, "issue": issue["name"], "exists": None,
                           "confidence": 0.0, "localization": None, "reasoning": "",
                           "error": f"{type(e).__name__}: {e}"}
            verdicts.append(verdict)
        out_path.write_text(json.dumps(verdicts, indent=2))

    print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
