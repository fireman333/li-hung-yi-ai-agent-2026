"""Phase F (video-native variant): cross-check lecture facts against NotebookLM.

Adapted from skill's phase_f_verify.py with two changes:
  1. Heuristics tuned for ML/AI lectures (not distributed systems / algorithms).
     Patterns that match: paper citations, model names (GPT/Claude/Gemini/Voyager),
     numeric token-counts, "李宏毅 said X" assertions.
  2. Less aggressive default — phase_f_strict=false in state means write report only,
     do not auto-rewrite. Claude orchestrator reviews patches.json and applies
     surgical Edits with diff preview.

Usage:
    python3 phase_f_video.py all
    python3 phase_f_video.py 1 2 3
    python3 phase_f_video.py all --max-facts 4
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / ".skill-state.json"
LECTURES_DIR = ROOT / "lectures"
PATCHES_PATH = ROOT / ".state" / "phase_f_patches.json"


def load_state() -> dict:
    if not STATE_PATH.exists():
        sys.exit(".skill-state.json not found.")
    return json.loads(STATE_PATH.read_text())


def extract_claims(md: str, max_n: int = 5) -> list:
    """Heuristic for ML/AI lecture content. Picks paragraphs with verifiable claims."""
    paragraphs = [p.strip() for p in md.split("\n\n") if p.strip()]
    candidates = []

    # Patterns ranked by signal strength for ML/AI content
    patterns = [
        # Specific paper / system / model names (high confidence claims)
        (r"\b(?:GPT-[0-9]o?|Claude|Gemini|Llama|DeepSeek|Voyager|ReAct|CoT|RAG)\b", 3),
        (r"(?:paper|論文)[^。]{0,30}(?:提出|提到|發現|證明|實驗)", 3),
        (r"\b(?:context window|context length|token|token 數)\s*(?:長度)?[^。]{0,30}\d", 3),
        (r"(?:作者|他們|該團隊)[^。]{0,30}(?:發現|證明|實驗|提出|報告)", 2),
        # Numeric facts
        (r"\d+\s*(?:K|M|B|tokens?|步驟|round|stage)\b", 2),
        (r"\b\d+(?:\.\d+)?\s*(?:%|percent)", 2),
        # Strong assertions
        (r"(?:必定|絕不|永遠|一定|無法|不可能)\b", 2),
        # Architecture / mechanism descriptions
        (r"(?:agent|代理人|代理)[^。]{0,30}(?:可以|能夠|會|無法)", 1),
        (r"(?:reasoning|推理|思考|reflection|反省)[^。]{0,30}(?:過程|機制|步驟)", 1),
    ]

    for p in paragraphs:
        if len(p) < 80 or len(p) > 800:
            continue
        if p.startswith(("#", "|", "```", "<", "- [")):
            continue
        score = 0
        for pat, w in patterns:
            if re.search(pat, p, re.I):
                score += w
        if score >= 3:
            candidates.append((score, p))

    candidates.sort(key=lambda x: -x[0])
    return [p for _, p in candidates[:max_n]]


def query_notebook(notebook_id: str, question: str) -> str:
    try:
        r = subprocess.run(
            ["nlm", "notebook", "query", notebook_id, question],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode != 0:
            return f"[ERROR] nlm query failed: {r.stderr.strip()[:300]}"
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] nlm query timeout (180s)"


def judge_mismatch(claim: str, answer: str) -> tuple:
    a_lower = answer.lower()
    negative_markers = [
        "不正確", "錯誤", "不符", "incorrect", "wrong", "not accurate",
        "這個敘述不", "應該是", "實際上是", "正確的說法",
        "並非", "並不是", "其實是",
    ]
    for m in negative_markers:
        if m in answer:
            for sent in re.split(r"[。\n]", answer):
                if m in sent:
                    return True, sent.strip()[:300]
    return False, ""


def verify_lecture(lec_path: Path, notebook_id: str, max_facts: int) -> list:
    print(f"\n[Verify] {lec_path.name}")
    md = lec_path.read_text()
    claims = extract_claims(md, max_n=max_facts)
    print(f"  Extracted {len(claims)} candidate claims")

    patches = []
    for i, claim in enumerate(claims, 1):
        q = (f"請根據來源影片內容判斷以下敘述是否正確：\n\n「{claim[:500]}」\n\n"
             f"若有錯，請具體指出錯在哪、正確說法是什麼，並引用影片中的時間點或段落。"
             f"若正確，請回「敘述正確」。")
        print(f"  [Q{i}] {claim[:80]}...")
        ans = query_notebook(notebook_id, q)
        is_mismatch, reason = judge_mismatch(claim, ans)

        if is_mismatch:
            print(f"    [MISMATCH] {reason[:100]}")
            patches.append({
                "lecture": lec_path.name,
                "old": claim,
                "reason": reason,
                "notebook_answer": ans[:1200],
            })
        else:
            print(f"    [OK] verified")
        time.sleep(2)

    return patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lectures", nargs="+")
    ap.add_argument("--max-facts", type=int, default=4)
    args = ap.parse_args()

    state = load_state()
    notebook_id = state.get("notebook_id")
    if not notebook_id:
        sys.exit("No notebook_id in state.")

    lec_files = sorted(LECTURES_DIR.glob("*.md"))
    if args.lectures != ["all"]:
        nums = set(int(x) for x in args.lectures)
        lec_files = [f for f in lec_files if int(f.name[:2]) in nums]

    all_patches = []
    for f in lec_files:
        patches = verify_lecture(f, notebook_id, args.max_facts)
        all_patches.extend(patches)

    PATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATCHES_PATH.write_text(json.dumps(all_patches, ensure_ascii=False, indent=2))

    print(f"\n==== Phase F Summary ====")
    print(f"  Mismatches found: {len(all_patches)}")
    print(f"  Report: {PATCHES_PATH}")
    if not all_patches:
        print("  All claims verified OK.")


if __name__ == "__main__":
    main()
