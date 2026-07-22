"""Taxonomy discovery: evolve the issue-category list from a batch of user feedback.

    python3 discover.py --feedback ~/Documents/badcase_data/h1/mapping_cn.json
    python3 discover.py --feedback feedback.json --examples labeled.json --apply

feedback: a JSON list where each item is either a string or an object with a
"feedback" field (other fields ignored) — e.g. a badcase mapping export.
examples (optional): a small JSON list of human-labeled few-shot examples,
[{"feedback": str, "category": str}, ...], to anchor category granularity.

Proposes an updated category list (keeping/merging existing categories, adding
new ones, dropping obsolete ones with rationale), prints it for human review,
and writes it to results/taxonomy_proposal.json. With --apply, also rewrites
taxonomy.categories in config.yaml (note: yaml comments in the file are lost).
"""
import argparse
import json
from pathlib import Path

import yaml

from utils.LLM import LLM

CONFIG_PATH = "config.yaml"
MAX_FEEDBACK_ITEMS = 300
MAX_FEEDBACK_CHARS = 200


def load_feedback(path):
    data = json.loads(Path(path).read_text())
    texts = []
    for item in data:
        text = item if isinstance(item, str) else item.get("feedback", "")
        text = (text or "").strip()
        if text:
            texts.append(text[:MAX_FEEDBACK_CHARS])
    return texts


def main():
    parser = argparse.ArgumentParser(description="Propose taxonomy updates from user feedback")
    parser.add_argument("--feedback", required=True, help="JSON list of feedback strings/objects")
    parser.add_argument("--examples", help="optional JSON list of {feedback, category} few-shot labels")
    parser.add_argument("--apply", action="store_true",
                        help="write the proposed categories back to config.yaml")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    current = config.get("taxonomy", {}).get("categories", [])

    feedback = load_feedback(args.feedback)
    if len(feedback) > MAX_FEEDBACK_ITEMS:
        print(f"note: using the first {MAX_FEEDBACK_ITEMS} of {len(feedback)} feedback items")
        feedback = feedback[:MAX_FEEDBACK_ITEMS]

    examples_block = ""
    if args.examples:
        labeled = json.loads(Path(args.examples).read_text())
        rendered = "\n".join(f'- "{e["feedback"][:MAX_FEEDBACK_CHARS]}" -> {e["category"]}' for e in labeled)
        examples_block = (
            "Human-labeled examples anchoring the desired category granularity:\n"
            f"{rendered}\n\n"
        )

    prompt = (
        "You maintain the issue taxonomy of an evaluation framework for AI-generated "
        "(image-to-video) videos. Categories must be issues detectable in a generated video, "
        "mutually distinct, and at a consistent granularity.\n\n"
        f"Current categories:\n{json.dumps(current, indent=2)}\n\n"
        f"{examples_block}"
        f"New batch of user feedback ({len(feedback)} items):\n"
        + "\n".join(f"- {t}" for t in feedback)
        + "\n\n"
        "Propose the updated category list: keep existing categories that the feedback still "
        "supports (you may sharpen their descriptions), merge overlapping ones, add categories "
        "for recurring failure modes not yet covered, and drop categories with no support. "
        "Order by how frequently the feedback supports each category, most frequent first. "
        "Respond as JSON: {\"categories\": [{\"name\": snake_case_str, \"description\": str, "
        "\"support\": int}], \"changes\": str} where support is the approximate number of "
        "feedback items backing the category and changes summarizes what you changed and why."
    )
    proposal = LLM().prompt(prompt)

    print(f"\nproposed categories ({len(proposal['categories'])}):")
    for c in proposal["categories"]:
        print(f"  {c['name']}  (support ~{c.get('support', '?')})")
        print(f"    {c['description']}")
    print(f"\nchanges: {proposal.get('changes', '')}")

    out_dir = Path(config.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = out_dir / "taxonomy_proposal.json"
    proposal_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False))
    print(f"\nproposal written to {proposal_path}")

    if args.apply:
        config.setdefault("taxonomy", {})["categories"] = [
            {"name": c["name"], "description": c["description"]} for c in proposal["categories"]
        ]
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"applied to {CONFIG_PATH} (top_n={config['taxonomy'].get('top_n')} still selects the head)")
    else:
        print("review the proposal, then re-run with --apply to write it into config.yaml")


if __name__ == "__main__":
    main()
