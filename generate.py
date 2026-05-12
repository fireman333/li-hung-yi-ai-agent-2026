"""Project-local lecture generator: gemini-cli watches YouTube directly.

Adapted from ~/.claude/skills/youtube-playlist-to-textbook/scripts/generate_lecture.py
with one structural change: prompts include the YouTube URL inline so gemini
processes the video natively, instead of relying on a yt-dlp transcript.

Usage:
    python3 generate.py 1                      # single lecture
    python3 generate.py all                    # all lectures
    python3 generate.py 1 --force              # re-run even if done
    python3 generate.py 1 --model gemini-2.5-pro
    python3 generate.py all --concurrent 2     # parallel
"""
import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL_DIR = Path.home() / ".claude" / "skills" / "youtube-playlist-to-textbook"
PLAYLIST_PATH = ROOT / "lectures-raw" / "playlist.json"
STATE_PATH = ROOT / ".skill-state.json"
PROGRESS_PATH = ROOT / ".state" / "progress.json"
PROGRESS_LOCK_PATH = ROOT / ".state" / "progress.json.lock"
PROGRESS_JOURNAL_PATH = ROOT / ".state" / "progress.jsonl"
LECTURES_DIR = ROOT / "lectures"


def count_cjk(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def load_playlist():
    return json.loads(PLAYLIST_PATH.read_text())


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


@contextlib.contextmanager
def _progress_lock():
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fp = open(PROGRESS_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        lock_fp.close()


def load_progress():
    with _progress_lock():
        if PROGRESS_PATH.exists():
            return json.loads(PROGRESS_PATH.read_text())
        return {}


def save_progress(p):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _progress_lock():
        merged = {}
        if PROGRESS_PATH.exists():
            try:
                merged = json.loads(PROGRESS_PATH.read_text())
            except Exception:
                merged = {}
        merged.update(p)
        payload = json.dumps(merged, ensure_ascii=False, indent=2)
        fd, tmp_path = tempfile.mkstemp(prefix=".progress.", suffix=".tmp", dir=str(PROGRESS_PATH.parent))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp_path, PROGRESS_PATH)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        with open(PROGRESS_JOURNAL_PATH, "a") as j:
            ts = int(time.time())
            for k, v in p.items():
                j.write(json.dumps({"ts": ts, "key": k, "entry": v}, ensure_ascii=False) + "\n")


def slug(title: str) -> str:
    s = re.sub(r"^(Lecture|Lec|Video)\s+\d+:?\s*", "", title, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:40] or "untitled"


def lecture_dir_name(num: int, title: str) -> str:
    return f"{num:02d}_{slug(title)}"


def load_persona() -> str:
    state = load_state()
    if state.get("persona_text"):
        persona = state["persona_text"]
    else:
        persona = (SKILL_DIR / "references" / "default_persona.md").read_text()
    if "{CONFUSION_POINTS}" in persona:
        subj = (state.get("subject") or "ml").lower().strip()
        cp_path = SKILL_DIR / "references" / "confusion_points" / f"{subj}.md"
        cp = cp_path.read_text().strip() if cp_path.exists() else ""
        persona = persona.replace("{CONFUSION_POINTS}", cp)
    return persona


def load_sections() -> str:
    state = load_state()
    if state.get("sections"):
        if isinstance(state["sections"], list):
            return "\n".join(f"- {s}" for s in state["sections"])
        return state["sections"]
    return (SKILL_DIR / "references" / "default_sections.md").read_text()


def build_prompt_a(lec: dict) -> str:
    persona = load_persona()
    sections = load_sections()
    return f"""# 系統角色

你是一位為以下讀者撰寫**中文教科書級講義**的作者：

{persona}

# 任務：產出 Lecture {lec['index']:02d} 中文講義的 Part A（§0-§4）

請直接觀看以下 YouTube 影片（gemini 內建支援，請以影片實際視聽內容為主，包括投影片畫面、口說內容、白板手寫、demo 畫面）：

**影片標題**：{lec['title']}
**影片連結**：{lec['url']}
**影片長度**：{lec.get('duration_str', '?')}
**講者**：{lec.get('uploader', '李宏毅 (Hung-yi Lee)')}

## 寫作分配

這是 2-part 講義的 **Part A**，你要寫的是 **§0-§4**（前半部）。後半部（§5-§9）會由另一次請求產出。**請只專心把 §0-§4 寫到最深最完整**，寫到 §4 就停筆。

**字數目標：Part A 中文字 3,500-5,000 字**。寧可深、不可淺。

## 語言規範（嚴格遵守）

1. **繁體中文為主，CS／ML 術語保留英文**：首次出現 `英文（中文）`，後續直接用英文
   - 例：`context（上下文）` → 後續 `context`
   - 例：`reasoning（推理）` → 後續 `reasoning`
2. 通用縮寫 LLM/RAG/CoT/API/CPU/GPU/SQL 直接用，不翻譯
3. 程式碼註解用英文
4. 禁用大陸譯名（不要寫「信息／数据／软件／程序／网络／智能体」，用「資訊／資料／軟體／程式／網路／代理人 (agent)」）
5. **不要反斜線跳脫**（不要寫 `\\.` `\\*` `\\&`，直接寫 `.` `*` `&`）
6. **直接以李宏毅教授的講授內容**為主，引用他講的具體例子、論文、demo

## Part A 必寫結構（§0-§4）

{sections}

※ Part A 只寫 §0-§4，不寫 §5-§9。

## 品質門檻（輸出前自我檢查）

- [ ] §0-§4 全部有內容，沒跳節
- [ ] §4 有至少一個 mermaid 或 ASCII 圖（描繪影片中提到的架構或流程）
- [ ] §4 有至少一段偽程式碼或實際程式片段（如影片有 demo 程式碼）
- [ ] §4 walkthrough 至少 2 組（正常 + 異常／失敗 case）
- [ ] 術語首次出現都有 `英文（中文）` 格式
- [ ] Part A 中文字數 ≥ 3,500
- [ ] 沒有反斜線跳脫
- [ ] 沒有大陸譯名
- [ ] 引用至少 2 個影片中提到的具體 paper / 系統名稱（且只引用影片有講的，不要編造）

---

**開始輸出 Part A**。直接從 `# Lecture {lec['index']:02d}: <主題>` 這一行開始寫，寫完 §4 Deep Dive 就停筆。不要寫 §5，不要寫「待續」、「未完」、「下部分請見...」之類的句子。
"""


def build_prompt_b(lec: dict) -> str:
    persona = load_persona()
    sections = load_sections()
    return f"""# 系統角色

你是一位為以下讀者撰寫**中文教科書級講義**的作者：

{persona}

# 任務：產出 Lecture {lec['index']:02d} 中文講義的 Part B（§5-§9）

請觀看以下 YouTube 影片（gemini 內建支援），以影片實際視聽內容為主：

**影片標題**：{lec['title']}
**影片連結**：{lec['url']}
**影片長度**：{lec.get('duration_str', '?')}

## 寫作分配

這是 2-part 講義的 **Part B**。前半部（§0-§4）已由另一次請求產出，**請只專心把 §5-§9 寫到最深最完整**，不要重複前面的內容。

**字數目標：Part B 中文字 3,500-5,000 字**。

## 語言規範（嚴格遵守）

1. 繁體中文為主，技術術語首次 `英文（中文）`，後續英文
2. 通用縮寫 LLM/RAG/CoT/API/CPU/GPU/SQL 直接用
3. 禁用大陸譯名
4. 不要反斜線跳脫
5. 引用李宏毅教授講過的具體 paper / 系統名稱（不要編造）

## Part B 必寫結構（§5-§9）

{sections}

※ Part B 只寫 §5-§9。不要重寫 `# Lecture XX: ...` 這行（那是 Part A 的起頭）。

## 特別強調：§5 醫學類比品質

§5 是本講義最大差異化價值所在。讀者是台灣醫學生，請選用**醫院／臨床／病歷／檢驗 workflow** 做類比。例如：

- **AI Agent 對 LLM 的關係** ↔ **住院醫師對 attending 的關係**：誰下決策、誰執行、誰收 feedback
- **Context engineering** ↔ **病歷 SOAP / handoff I-PASS**：什麼資訊要 carry forward、什麼可以丟
- **Tool use** ↔ **醫師會診（consult）**：知道什麼時候 call 哪科、怎麼開單、怎麼讀報告
- **Self-correction** ↔ **morbidity & mortality conference**：error 後 retrospective 分析、改 protocol

每組類比必須有：
1. 類比情境描述（150+ 字）
2. 對應關係表（本堂 AI/ML 概念 ↔ 醫院場景元件）
3. ✅ 吻合之處（為何類比有效）
4. ⚠️ 不吻合之處（類比邊界）

**至少 3 組類比，每組 400+ 字，§5 總字數 ≥ 1,500。**

## §6 Q&A 注意

李宏毅的影片是 monologue 為主（不是 lecture hall Q&A）。§6 改寫為：「教授特別強調的觀念釐清」— 抽 6 組他在影片中明確指出「常見誤解 / 容易搞錯 / 學生常問」的點，仍以 Q/A 格式呈現，但 Q 是「讀者可能會問的問題」、A 是「教授在影片中對此的回答／釐清」。

## 品質門檻（輸出前自我檢查）

- [ ] §5-§9 全部有內容
- [ ] §5 至少 3 組醫學類比，每組有對應表 + ✅ + ⚠️
- [ ] §6 至少 6 組 Q&A
- [ ] §7 至少 6 條陷阱
- [ ] §8 正好 10 題，每題有詳細答案解釋
- [ ] §9 至少 2 項延伸閱讀（影片中提到的 paper 或他自己的其他課程）
- [ ] Part B 中文字數 ≥ 3,500
- [ ] 沒有反斜線跳脫，沒有大陸譯名

---

**開始輸出 Part B**。直接從 `## §5. 真實類比` 開始寫（不要重寫 `# Lecture XX` 標題）。寫完 §9 就停筆。
"""


QUOTA_PATTERNS = ("429", "quota", "exhausted", "rate limit", "RESOURCE_EXHAUSTED")
RETRYABLE_PATTERNS = ("AbortError", "ECONNRESET", "socket hang up", "ETIMEDOUT", "timed out")

MAX_RPM = int(os.environ.get("YTT_MAX_RPM", "8"))
_rpm_lock: asyncio.Lock | None = None
_rpm_calls: list[float] = []


async def _throttle_rpm():
    global _rpm_lock
    if _rpm_lock is None:
        _rpm_lock = asyncio.Lock()
    async with _rpm_lock:
        now = time.time()
        cutoff = now - 60.0
        while _rpm_calls and _rpm_calls[0] < cutoff:
            _rpm_calls.pop(0)
        if len(_rpm_calls) >= MAX_RPM:
            wait = _rpm_calls[0] + 60.0 - now + 0.5
            if wait > 0:
                print(f"    [rpm] {len(_rpm_calls)}/{MAX_RPM} → sleep {wait:.1f}s", flush=True)
                await asyncio.sleep(wait)
        _rpm_calls.append(time.time())


async def call_gemini_with_retry(prompt: str, model: str, label: str, timeout: float = 1500.0, max_retries: int = 4) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            return await call_gemini(prompt, model, label, timeout)
        except RuntimeError as e:
            last_err = e
            msg = str(e)
            msg_lower = msg.lower()
            is_quota = any(p.lower() in msg_lower for p in QUOTA_PATTERNS)
            is_generic = any(p in msg for p in RETRYABLE_PATTERNS)
            if not (is_quota or is_generic):
                raise
            if attempt == max_retries - 1:
                raise
            if is_quota:
                wait = min(300, 60 * (2 ** attempt) + random.uniform(0, 15))
                tag = "429/quota"
            else:
                wait = min(120, 15 * (2 ** attempt) + random.uniform(0, 10))
                tag = "transient"
            print(f"    [{label}] {tag} → sleep {wait:.0f}s (attempt {attempt + 1}/{max_retries}): {msg[:120]}", flush=True)
            await asyncio.sleep(wait)
    raise last_err


async def call_gemini(prompt: str, model: str, label: str, timeout: float = 1500.0) -> str:
    await _throttle_rpm()
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        "gemini", "-m", model,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"gemini call timed out after {timeout}s")

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"gemini exited {proc.returncode}: {err}")

    text = stdout.decode("utf-8", errors="ignore")
    dt = time.time() - t0
    print(f"    [{label}] {len(text)} chars, {count_cjk(text)} CJK ({dt:.0f}s)", flush=True)
    return text


FALLBACK_MODEL = "gemini-2.5-pro"
MIN_PART_CJK = 1500


async def _generate_with_model(lec: dict, model: str) -> dict:
    num = lec["index"]
    out_path = LECTURES_DIR / f"{lecture_dir_name(num, lec['title'])}.md"
    t0 = time.time()

    prompt_a = build_prompt_a(lec)
    text_a = await call_gemini_with_retry(prompt_a, model, f"Lec{num:02d}/A[{model}]")
    if count_cjk(text_a) < MIN_PART_CJK:
        return {"ok": False, "err": f"Part A too short ({count_cjk(text_a)} CJK)", "model": model}

    await asyncio.sleep(5)

    prompt_b = build_prompt_b(lec)
    text_b = await call_gemini_with_retry(prompt_b, model, f"Lec{num:02d}/B[{model}]")
    if count_cjk(text_b) < MIN_PART_CJK:
        return {"ok": False, "err": f"Part B too short ({count_cjk(text_b)} CJK)", "model": model}

    combined = text_a.rstrip() + "\n\n" + text_b.lstrip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined)

    dt = time.time() - t0
    cjk_total = count_cjk(combined)
    print(f"[Lec {num:02d}] OK {cjk_total} CJK -> {out_path.name}  ({dt:.0f}s, {model})", flush=True)
    return {
        "ok": True,
        "chars": len(combined),
        "cjk_total": cjk_total,
        "cjk_a": count_cjk(text_a),
        "cjk_b": count_cjk(text_b),
        "path": str(out_path.relative_to(ROOT)),
        "duration_s": round(dt, 1),
        "model": model,
        "ts": int(time.time()),
    }


async def generate_one(lec: dict, model: str) -> dict:
    num = lec["index"]
    print(f"\n[Lec {num:02d}] {lec['title']}", flush=True)
    try:
        res = await _generate_with_model(lec, model)
        if res.get("ok"):
            return res
        err = res.get("err", "")
        if model != FALLBACK_MODEL and "too short" in err:
            print(f"    [Lec {num:02d}] short on {model} ({err}) -> fallback {FALLBACK_MODEL}", flush=True)
            res2 = await _generate_with_model(lec, FALLBACK_MODEL)
            if res2.get("ok"):
                res2["fallback_from"] = model
            return res2
        return res
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "model": model}


async def run(lec_nums, concurrent: int, force: bool, model: str):
    playlist = load_playlist()
    index_map = {v["index"]: v for v in playlist}
    progress = load_progress()

    pending = []
    for n in lec_nums:
        if n not in index_map:
            print(f"  !! Lec {n:02d} not in playlist (skip)")
            continue
        key = f"lec_{n:02d}"
        if not force and progress.get(key, {}).get("ok"):
            print(f"  [skip] Lec {n:02d} already done ({progress[key].get('cjk_total')} CJK)")
            continue
        pending.append(index_map[n])

    if not pending:
        print("Nothing to do.")
        return

    for i in range(0, len(pending), concurrent):
        batch = pending[i:i + concurrent]
        print(f"\n========== Batch: {[l['index'] for l in batch]} ==========")
        coros = [generate_one(l, model) for l in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for lec, r in zip(batch, results):
            key = f"lec_{lec['index']:02d}"
            if isinstance(r, Exception):
                progress[key] = {"ok": False, "err": f"exception: {type(r).__name__}: {r}"}
            else:
                progress[key] = r
        save_progress(progress)
        if i + concurrent < len(pending):
            print("(sleep 30s before next batch)")
            await asyncio.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lectures", nargs="+", help="Lecture numbers (or 'all')")
    ap.add_argument("--concurrent", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    playlist = load_playlist()
    if args.lectures == ["all"]:
        nums = [v["index"] for v in playlist]
    else:
        nums = [int(x) for x in args.lectures]

    asyncio.run(run(nums, args.concurrent, args.force, args.model))


if __name__ == "__main__":
    main()
