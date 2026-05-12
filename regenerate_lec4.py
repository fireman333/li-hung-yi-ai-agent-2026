"""One-off corrective regenerator for Lec 4 (Harness Engineering).

The original generation hallucinated heavily — gemini-2.5-pro associated 'harness'
with LangChain-style structured-output frameworks (Guidance / Outlines / LMQL /
constrained decoding / logit masking). NotebookLM verification confirmed the
actual video content is about the 馬具 (horse-tack) metaphor:

  Harness = the non-LLM scaffolding (rules, tools, workflows) around the LLM
  in an AI Agent system. 3 control mechanisms:
    1. 認知框架 control via agents.md / Cloud.md style human-language rule files
    2. 工具能力邊界 (tool boundary control)
    3. 標準工作流程 (planner / generator / evaluator pattern, from Anthropic Hness Design)
  Frameworks discussed: OpenCloud (codename '小金'), Claude Code, CoWork (cowork.md style)
  Comparison: prompt engineering -> context engineering -> harness engineering
  (each is a superset / generalization of the prior)

This script regenerates Lec 4 with an aggressive anti-hallucination prompt that
pre-seeds these anchor facts and forbids constrained-decoding / logit-masking content.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import (
    call_gemini_with_retry, count_cjk, lecture_dir_name,
    load_persona, load_sections, save_progress, LECTURES_DIR,
)


LEC = {
    "index": 4,
    "title": "Harness Engineering：有時候語言模型不是不夠聰明，只是沒有人類好好引導",
    "url": "https://www.youtube.com/watch?v=R6fZR_9kmIw",
    "duration_str": "1:32:21",
    "uploader": "Hung-yi Lee",
}


GROUND_TRUTH = """
# 影片真實內容 ground-truth（嚴格對照）

李宏毅老師在這支影片中講的「Harness Engineering」**不是** LangChain 風格的
constrained-decoding framework（Guidance / Outlines / LMQL / logit masking 等都
**完全沒有提到**）。請嚴格依照以下從影片驗證過的事實寫講義：

## 1. Harness 的定義（馬具比喻）

- 「**AI 是一匹馬，還有很強大的力量，但是你要駕馭它，你還是需要一些馬具，
  你需要馬鞍，你需要韁繩，那這些馬鞍韁繩就是 Harness**」
- **AI Agent = LLM + Harness**。Harness 是 LLM 周邊「其他」程式的總稱
  （tools、workflows、rule files），過去沒有好名字，現在統稱 harness。
- 中文翻譯：「**駕馭工程**」。

## 2. 三種「駕馭手段」（核心架構）

老師明確列出三個面向（影片中有圖：藍色 = 手段，紅色 = 控制對象）：

(A) **控制認知框架（cognitive framework）**：用人類語言寫成的「規則」放進 prompt。
   - 範例：`agents.md`（OpenCloud 用）、`Cloud.md`（Claude Code 用）、`cowork.md`（CoWork 用）
   - 這些 .md 是「給 AI 看的 README」，每次 agent 啟動時自動讀進 prompt
   - 類比：人類社會的法律 — 不一定 100% 遵守，但建立 baseline

(B) **控制工具能力邊界**：透過給 AI agent 的 tool 加上限制／過濾，控制它能做什麼。

(C) **制定標準工作流程（SOP）**：planner / generator / evaluator pattern。
   - 來源：Anthropic 在 2026 年 3 月發的 *Harness Design* paper（影片有提到）
   - 流程：人類給指令 → planner（拆解成小項目）→ generator（執行）→ evaluator（檢查）
   - 變體：generator + evaluator 開始前先簽 contract（避免做完才發現方向錯）
   - JSON structure 在 CLI/CLI line 中很常見（AI 擅長輸出 JSON）

## 3. 三個 *Engineering 的關係

老師明確說明「prompt engineering ⊂ context engineering ⊂ harness engineering」
（後者是前者的延伸／一般化），但邊界模糊：
- **Prompt Engineering**：早期 LLM 弱，要靠咒語（"step by step"）；現在新模型都會自動推理，咒語越來越沒用
- **Context Engineering**：自動化 prompt engineering，系統會挑合適的 context 組成 prompt
- **Harness Engineering**：上面兩者都是 sub-task；核心關注「LLM 在多輪互動中完成任務」

## 4. 影片提到的具體 framework / 廠商 blog

- **OpenCloud** (暱稱「小金」)：李宏毅自己常用的 agent，配 `agents.md`
- **Claude Code** (Anthropic)：配 `Cloud.md`
- **CoWork**（Google？）：配 `cowork.md`
- **小龍蝦**（Lobster？影片中提到）：另一個 harness 的代稱
- **Anthropic 2024-11 文章**：談 effective harness、long-running agent
- **OpenAI 2026-02 文章**：談 Harness Engineering
- **Anthropic 2026-03 文章**：*Harness Design* — planner/generator/evaluator 出處

## 5. 一個有趣的延伸

- 在 OpenCloud 上跑的 agent 可以「移植」到 Claude Code：把 `agents.md` 改名成 `Cloud.md` 即可
- AI 跟人不一樣的地方：寫 500 字作文對 AI 是一瞬間，對人類是費力 → 所以 AI 擅用的工具跟人不同（如複雜 JSON）
- 老師問 AI（小金）：「你怎麼判斷對方是人還是 AI？」AI 答：「叫他寫 500 字作文，能瞬間寫出來的就是 AI。」

---

# 嚴格禁止事項

下列內容**完全沒有出現在影片中**，請**絕對不要**寫進講義：

- ❌ Constrained decoding / logit masking
- ❌ Guidance / Outlines / LMQL framework
- ❌ JSON Schema / Pydantic / 結構化輸出 framework
- ❌ Function calling、tool use 的低階實作（影片只在概念層面講工具）
- ❌ Token 機率分佈操作 / decoding 演算法（這是另一支 self-correction 影片的主題）
- ❌ 任何訓練／fine-tuning 的細節（影片明確說 harness 跟訓練是兩條不同的強化路徑）
- ❌ 關於 PACS / EMR 細節（除非用作類比，且要明確標註類比邊界）
"""


def build_prompt_a_v2(lec):
    persona = load_persona()
    sections = load_sections()
    return f"""# 系統角色

你是一位為以下讀者撰寫**中文教科書級講義**的作者：

{persona}

# 任務：產出 Lecture {lec['index']:02d} 中文講義的 Part A（§0-§4）

請直接觀看以下 YouTube 影片（gemini 內建支援），以**影片實際視聽內容為唯一依據**：

**影片標題**：{lec['title']}
**影片連結**：{lec['url']}
**影片長度**：{lec.get('duration_str', '?')}
**講者**：李宏毅 (Hung-yi Lee), 國立台灣大學電機系教授

⚠️ 上次生成此講義時 gemini 把「Harness Engineering」誤認成 LangChain 風格的
constrained-decoding framework（Guidance / Outlines / LMQL），這是嚴重錯誤。
請**嚴格依照下方 ground-truth**寫，不要從訓練資料拉外部知識進來。

{GROUND_TRUTH}

---

## 寫作分配

這是 2-part 講義的 **Part A（§0-§4）**。後半部由另一次請求產出。

**字數目標：3,500-5,000 中文字**。寧可深、不可淺。

## 語言規範（嚴格遵守）

1. **繁體中文為主，CS／ML 術語保留英文**：首次出現 `英文（中文）`，後續英文
2. 通用縮寫 LLM/RAG/CoT/API 直接用
3. 程式碼註解用英文
4. 禁用大陸譯名（不要寫「智能體」，用「代理人 (agent)」）
5. **不要反斜線跳脫**

## Part A 必寫結構（§0-§4）

{sections}

## 品質門檻（輸出前自我檢查）

- [ ] §0-§4 全部有內容
- [ ] §3 至少把 harness 的「馬具比喻」清楚講過
- [ ] §4 有 mermaid 圖：人類指令 → planner → generator → evaluator 流程
- [ ] §4 有偽程式碼示範 `agents.md` 風格的 rule file
- [ ] §4 walkthrough 至少 2 組（agents.md 工作流 + planner-evaluator 工作流）
- [ ] **完全不提**：constrained decoding / logit masking / Guidance / Outlines / LMQL / JSON Schema 強制
- [ ] 引用至少 2 個影片提到的 framework：OpenCloud（小金）、Claude Code、CoWork、Anthropic blog
- [ ] Part A 中文字數 ≥ 3,500

---

**開始輸出 Part A**。從 `# Lecture 04: Harness Engineering（駕馭工程）` 開始。寫完 §4 就停。
"""


def build_prompt_b_v2(lec):
    persona = load_persona()
    sections = load_sections()
    return f"""# 系統角色

你是一位為以下讀者撰寫**中文教科書級講義**的作者：

{persona}

# 任務：產出 Lecture {lec['index']:02d} 中文講義的 Part B（§5-§9）

請觀看以下 YouTube 影片（嚴格以影片內容為依據）：

**影片標題**：{lec['title']}
**影片連結**：{lec['url']}

{GROUND_TRUTH}

---

## 寫作分配

這是 Part B（§5-§9），前半部已產出。**字數目標：3,500-5,000 中文字**。

## §5 醫學類比靈感（讀者是台灣醫學生）

請挑下列**至少 3 組**做類比，每組 400+ 字：

- **Harness ↔ 病房 SOP / 護理工作流程 / 醫囑書寫規則**：LLM 像實習醫師（聰明但易犯錯），harness 像護理 SOP / 醫囑系統的 sanity check
- **agents.md ↔ 主治醫師交班的「定期 reminder」**：每次值班開始前要 check 的固定事項（過敏史、隔離 status、DNR）
- **Planner / Generator / Evaluator ↔ Resident 寫 admission note / 主治改 note / 主任 sign-off** 三層 review
- **Tool boundary ↔ 不同層級 resident 能下的 order**：PGY-1 不能下 controlled substance、ICU 才能下 vasopressor — 這是工具能力邊界
- **Cloud.md vs agents.md 換名移植 ↔ 病人從醫學中心轉地區醫院**：medication list 格式不同但內容一樣

## §6 改寫為「教授特別強調的觀念釐清」

李宏毅老師在影片中明確指出哪些**觀念釐清**？例如：
- harness 跟 LLM 的訓練是「兩條獨立強化路徑」
- prompt / context / harness engineering 邊界其實模糊
- agents.md 像法律：不保證 100% 遵守
- AI 擅長的工具跟人不同（500 字作文 case）

## 嚴格禁止（Part B 也適用）

- ❌ 不要寫 constrained decoding、Guidance、Outlines、LMQL
- ❌ 不要寫 JSON Schema / Pydantic 強制輸出
- ❌ §8 自測題不要考 logit masking 等不在影片內的內容

## Part B 必寫結構（§5-§9）

{sections}

## 品質門檻

- [ ] §5 至少 3 組醫學類比，每組有對應表 + ✅ 吻合 + ⚠️ 不吻合
- [ ] §6 至少 6 組 Q&A
- [ ] §7 至少 6 條陷阱
- [ ] §8 正好 10 題
- [ ] §9 至少列出影片中提到的 Anthropic / OpenAI 文章作為延伸閱讀
- [ ] Part B ≥ 3,500 中文字

---

**開始輸出 Part B**。從 `## §5. 真實類比` 開始（不要重寫 `# Lecture 04`）。寫完 §9 就停。
"""


async def main():
    print(f"\n[Lec 04 RETRY with anti-hallucination prompt]")
    t0 = time.time()
    out_path = LECTURES_DIR / f"{lecture_dir_name(LEC['index'], LEC['title'])}.md"

    text_a = await call_gemini_with_retry(
        build_prompt_a_v2(LEC), "gemini-2.5-pro", "Lec04/A[v2]"
    )
    if count_cjk(text_a) < 1500:
        print(f"  Part A too short ({count_cjk(text_a)} CJK), abort")
        return

    await asyncio.sleep(5)

    text_b = await call_gemini_with_retry(
        build_prompt_b_v2(LEC), "gemini-2.5-pro", "Lec04/B[v2]"
    )
    if count_cjk(text_b) < 1500:
        print(f"  Part B too short ({count_cjk(text_b)} CJK), abort")
        return

    combined = text_a.rstrip() + "\n\n" + text_b.lstrip()
    out_path.write_text(combined)

    dt = time.time() - t0
    cjk_total = count_cjk(combined)
    print(f"\n[Lec 04 v2] OK {cjk_total} CJK -> {out_path.name}  ({dt:.0f}s)")

    save_progress({
        "lec_04": {
            "ok": True,
            "chars": len(combined),
            "cjk_total": cjk_total,
            "cjk_a": count_cjk(text_a),
            "cjk_b": count_cjk(text_b),
            "path": str(out_path.relative_to(Path(__file__).resolve().parent)),
            "duration_s": round(dt, 1),
            "model": "gemini-2.5-pro",
            "regenerated": True,
            "ts": int(time.time()),
        }
    })


if __name__ == "__main__":
    asyncio.run(main())
