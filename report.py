"""Aggregate per-sample verdicts into a per-dataset quality report.

    python report.py --dataset seedance

Reads results/<dataset>/sample_*.json (written by main.py), prints a per-issue
summary, and writes results/<dataset>/report.json.
"""
import argparse
import json
from pathlib import Path

import yaml

CONFIG_PATH = "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Summarize agent verdicts for a dataset")
    parser.add_argument("--dataset", required=True, help="dataset/model whose results to summarize")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    out_dir = Path(config.get("output_dir", "results")) / args.dataset
    files = sorted(out_dir.glob("sample_*.json"))
    if not files:
        raise SystemExit(f"no verdict files in {out_dir} — run main.py first")

    issues, flagged, total = {}, [], 0
    for f in files:
        for v in json.loads(f.read_text()):
            total += 1
            s = issues.setdefault(v["issue"], {"present": 0, "absent": 0, "unsure": 0,
                                               "error": 0, "confidences": []})
            if v.get("error"):
                s["error"] += 1
                continue
            s["confidences"].append(v.get("confidence", 0.0))
            if v["exists"] is True:
                s["present"] += 1
                flagged.append((f.stem, v))
            elif v["exists"] is False:
                s["absent"] += 1
            else:
                s["unsure"] += 1

    print(f"[{args.dataset}] {len(files)} samples, {total} verdicts\n")
    print(f"{'issue':<36} {'present':>7} {'absent':>7} {'unsure':>7} {'error':>6} {'mean_conf':>10}")
    report_issues = {}
    for name, s in issues.items():
        judged = s["present"] + s["absent"] + s["unsure"]
        mean_conf = sum(s["confidences"]) / len(s["confidences"]) if s["confidences"] else 0.0
        rate = s["present"] / judged if judged else 0.0
        print(f"{name:<36} {s['present']:>7} {s['absent']:>7} {s['unsure']:>7} {s['error']:>6} {mean_conf:>10.2f}")
        report_issues[name] = {"present": s["present"], "absent": s["absent"], "unsure": s["unsure"],
                               "error": s["error"], "present_rate": round(rate, 3),
                               "mean_confidence": round(mean_conf, 3)}

    report_flagged = []
    if flagged:
        print("\nflagged (issue present):")
        for sample_name, v in flagged:
            loc = v.get("localization") or {}
            span = (f"t={loc.get('start_time')}-{loc.get('end_time')}"
                    if loc.get("start_time") is not None else "t=?")
            print(f"  {sample_name} :: {v['issue']}  {span}  conf={v.get('confidence', 0.0):.2f}")
            report_flagged.append({"sample": sample_name, "issue": v["issue"],
                                   "start_time": loc.get("start_time"), "end_time": loc.get("end_time"),
                                   "boxes": loc.get("boxes", []),
                                   "confidence": v.get("confidence", 0.0),
                                   "reasoning": v.get("reasoning", "")})

    report = {"dataset": args.dataset, "samples": len(files), "verdicts": total,
              "issues": report_issues, "flagged": report_flagged}
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {report_path}")


if __name__ == "__main__":
    main()
