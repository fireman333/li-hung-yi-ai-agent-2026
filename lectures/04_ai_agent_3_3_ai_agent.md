---
title: "AI Agent's Impact on Work（以學術研究為例）"
lectureNumber: 4
videoUrl: "https://www.youtube.com/watch?v=VqB8zMujdjM"
videoId: VqB8zMujdjM
durationMin: 24
readingMin: 30
uploadedAt: 2026-04-29
tableOfContents:
  minHeadingLevel: 2
  maxHeadingLevel: 3
---
## §0. TL;DR（速覽）

- **一句話總結**：AI 代理人 (agent) 正在重塑知識型工作，本堂課以學術研究為例，探討如何讓 AI 擔任研究助理，從文獻搜尋、點子發想到論文撰寫，以及人類專家在其中不可或缺的監督角色。
- **Key Takeaways**：
    1.  **AI 作為研究加速器**：AI Agent 可自動化處理繁瑣的學術任務（如文獻回顧），將研究者從重複性工作中解放出來，專注於更高層次的思考與創新。
    2.  **工具使用是關鍵**：單純的 Large Language Model (LLM) 無法接觸最新或私有的資料庫。真正的 AI Agent 必須能使用外部工具 (tool)，例如呼叫 `Scopus`、`PubMed` 的 API 進行文獻檢索。
    3.  **批判性驗證的必要性**：AI 會產生幻覺 (hallucination)，給出看似專業但實則錯誤的資訊。人類專家必須扮演最終的守門員，對 AI 產出的所有內容進行嚴格的批判性驗證。
    4.  **從自動化任務到自動化科學**：目前的 AI Agent 主要處理單點任務。未來的終極想像是「自動化科學」，AI 能自主提出假說、設計實驗、甚至操作實驗儀器，但這條路仍充滿挑戰。

- 預計閱讀時間：約 30 分鐘

---

## §1. Motivation（為什麼要這堂課）

在前兩堂課中，我們建立了 AI Agent 的基本概念，理解了它如何透過 `ReAct` 等框架進行 `reasoning（推理）`、`planning（規劃）` 與 `tool usage（工具使用）`。然而，這些抽象概念在真實世界中究竟能帶來多大的顛覆性影響？學術界與產業界對 AI Agent 的熱切期待，究竟是務實的預測，還是又一輪被過度炒作的泡沫？

這堂課將回答這個核心問題。我們將焦點從 Agent 的內部運作機制，轉向其在真實工作場景中的應用與衝擊。而「學術研究」是一個絕佳的切入點。對各位醫學背景的讀者而言，做研究是職涯中不可避免的一環，其流程——從提出問題、回顧文獻、設計實驗、分析數據到撰寫論文——既有結構，又充滿了知識密集型的挑戰。

想像一下當代的醫學研究。根據統計，全球醫學文獻的數量大約每 73 天就翻一倍。一位臨床醫師或研究者，即便只專注於一個極度狹窄的次專科領域，也幾乎不可能讀完所有相關的新發表。傳統的文獻回顧，仰賴研究者手動在 `PubMed`、`Embase` 等資料庫中設定關鍵字、篩選上千篇摘要、再一篇篇精讀，整個過程可能耗費數週甚至數月。這不僅是時間成本，更可能因為遺漏關鍵文獻而導致研究方向走偏。

一個 naïve 的解決方案可能是直接打開 `ChatGPT`，命令它：「給我寫一篇關於『標靶藥物在非小細胞肺癌治療』的文獻回顧」。你會很快發現它的致命缺陷：
1.  **知識陳舊**：通用 LLM 的知識有其 cut-off date，無法得知最近幾個月、甚至去年的最新臨床試驗結果。
2.  **無法存取專業資料庫**：LLM 無法直接連線到 `Scopus` 或 `Web of Science` 這類需要訂閱的學術資料庫，更不用說存取醫院內部 EMR 的數據。
3.  **Hallucination（幻覺）**：它可能會「編造」出不存在的論文、作者，甚至是杜撰臨床試驗的數據，且寫得煞有其事，極具欺騙性。

這些缺陷凸顯了單純的 LLM 無法勝任嚴謹的學術工作。這正是本堂課的主角——能夠使用工具的 AI Agent——發揮作用的地方。一個設計得當的 Research Agent，可以被賦予使用學術搜尋引擎 API 的能力，可以執行程式碼來分析數據，甚至可以讀取 PDF 檔案。它不再是一個封閉的語言模型，而是一個能夠與真實世界資訊系統互動的「代理人」。

本堂課將深入探討，當我們把 AI Agent 放在「研究助理」這個職位上時，會發生什麼事。它如何改變我們的研究 workflow？我們該如何下達指令（`prompting`）才能得到最好的結果？以及最重要的，身為人類專家，我們該如何與這位「非常聰明但有時會一本正經胡說八道」的助理協作，才能在加速研究的同時，確保科學的嚴謹性與正確性？

---

## §2. 背景知識補完（Prerequisites）

在深入探討 AI Agent 如何改變學術研究之前，讓我們先補齊一些必要的背景知識。這些概念是理解本堂課核心內容的基石。

1.  **RAG (Retrieval-Augmented Generation，檢索增強生成)**
    - **嚴謹定義**：一種結合了 `information retrieval（資訊檢索）` 系統與 `language model（語言模型）` 的架構。當模型需要回答問題或生成內容時，它會先從一個外部的知識庫（如 PDF 文件、資料庫、網頁）中檢索相關的資訊片段，然後將這些片段作為額外的 `context（上下文）`，與原始問題一起提供給語言模型，以生成更準確、更具時效性的回答。
    - **白話版**：就像給 LLM 一本書讓它「開卷考試」。當你問 `ChatGPT` 一個它不知道的問題，它只能猜。但有了 RAG，它會先去你指定的資料庫（例如最新的醫學文獻庫）裡「翻書」，找到最相關的幾頁，然後根據這幾頁的內容來回答你。這樣不僅答案更準確，還能告訴你「答案出自第幾頁」（引用來源）。
    - **為何本堂會用到**：學術研究極度仰賴最新且可信的文獻。RAG 是讓 AI Agent 能夠「閱讀」最新論文、存取私有研究資料的關鍵技術。沒有 RAG，Agent 就只是個知識停留在過去的空談家。

2.  **API (Application Programming Interface，應用程式介面)**
    - **嚴謹定義**：一組預先定義的規則、協定和工具，允許不同的軟體應用程式之間相互溝通和交換資料。它定義了可以發出的請求種類、如何發出請求、應使用的資料格式等。
    - **白話版**：想像你去餐廳吃飯，你（應用程式 A）不需要知道廚房（應用程式 B）內部如何運作，你只需要看懂菜單（API 文件），然後告訴服務生（API）你要點什麼（發出請求）。廚房根據訂單做好菜後，服務生再把菜端給你（回傳資料）。API 就是軟體世界裡的「標準化服務生」。
    - **為何本堂會用到**：AI Agent 的「工具使用」能力，本質上就是透過呼叫各種 API 實現的。例如，Agent 想要搜尋文獻，它不是模擬人類去點擊網頁，而是直接呼叫 `PubMed` 或 `Scopus` 提供的 API，以結構化的方式快速獲取資料。

3.  **Systematic Review (系統性文獻回顧)**
    - **嚴謹定義**：一種嚴謹的學術研究方法，旨在針對一個明確定義的研究問題，全面性地搜尋、評估和整合所有相關的實證研究。其過程包含制定周詳的檢索策略、設定清晰的納入與排除標準 (inclusion/exclusion criteria)、評估文獻品質，並對結果進行綜合分析（有時會包含 meta-analysis）。
    - **白話版**：這是醫學研究中「最硬核」的文獻回顧。它不是隨意挑幾篇相關論文來看，而是像做實驗一樣，預先設計好一套「撈資料」的標準作業流程 (SOP)，確保 unbiased 地把所有符合條件的文獻都找出來，然後進行系統性的整理與總結。目標是提供目前關於某個臨床問題最高證據等級的總結。
    - **為何本堂會用到**：AI Agent 自動化文獻回顧的 workflow，很大程度上就是在模仿 Systematic Review 的流程。理解 Systematic Review 的嚴謹步驟，有助於我們設計出更好的 prompt 和 Agent 行為，也讓我們知道該從哪些環節去審核 Agent 的產出是否可靠。

4.  **Prompt Engineering (提示工程)**
    - **嚴謹定義**：設計和優化輸入提示 (prompt) 的藝術與科學，旨在引導 LLM 生成更準確、更相關、更符合期望的輸出。它涵蓋了指令的措辭、提供範例 (few-shot prompting)、設定角色、以及將複雜任務分解為多個簡單步驟等技巧。
    - **白話版**：對 LLM「說話的藝術」。跟它說話不能像跟人一樣隨意，而是要學習一套「咒語」。好的 `prompt` 就像給了 GPS 清晰的目的地和路徑偏好設定，讓它能精準導航；壞的 `prompt` 則像只說了「我想出門走走」，結果可能被帶到任何地方。
    - **為何本堂會用到**：要讓 AI Agent 這個「聰明助理」好好工作，你必須學會當個好「老闆」。Prompt Engineering 就是你下達指令的語言。在本堂課的學術研究情境中，你需要設計出能讓 Agent 理解複雜研究問題、遵循特定搜尋策略、並以固定格式產出報告的 `prompt`。

---

## §3. 核心概念辭典（Core Concepts Glossary）

本堂課將圍繞以下幾個核心概念展開，理解它們的精確含義至關重要。

1.  **AI Agent (AI 代理人)**
    - **嚴謹定義**：一個能感知其環境、自主做出決策、並透過執行動作以達成特定目標的運算實體。在 LLM 的脈絡下，特指以 LLM 為核心大腦，具備規劃 (planning)、記憶 (memory) 和工具使用 (tool usage) 能力的系統。
    - **白話重述**：你可以把它想像成一個軟體機器人。它不只會聊天（像 `ChatGPT`），它還會「做事」。你給它一個目標（例如「幫我找出最近五年治療阿茲海默症的所有新藥」），它會自己規劃步驟（1. 去 `PubMed` 搜尋 2. 篩選臨床試驗文章 3. 整理成表格），並實際去執行這些步驟（呼叫 API、讀取檔案），最後把結果交給你。
    - **常見誤解**：誤以為 AI Agent 就是 `ChatGPT` 加上外掛。真正的 Agent 核心在於其「自主性」和「目標導向」的規劃能力。它不是被動地回應指令，而是會為了達成最終目標而主動地、連續地執行一系列動作。

2.  **Tool Usage (工具使用)**
    - **嚴謹定義**：AI Agent 透過呼叫外部 API 或執行程式碼，來獲取或處理其內部知識所不包含的資訊的能力。這是 Agent 與環境互動、擴展其能力邊界的主要手段。
    - **白話重述**：就像你給了 iPhone 裡的 Siri 使用 App 的權限。當你問天氣，Siri 不是靠自己「通靈」出溫度，而是去呼叫天氣 App (一個工具) 來取得即時資訊。同理，Research Agent 要查文獻，就是去呼叫 `Scopus` 這個「查文獻 App」。
    - **相近概念區辨**：`Tool Usage` vs `RAG`：RAG 是 `Tool Usage` 的一種特例。`RAG` 專指「檢索」這個動作，是去讀取現有資料。而 `Tool Usage` 範圍更廣，還包含執行計算、發送 Email、甚至在未來操作實體機器人等「寫入」或「執行」類型的動作。

3.  **Hallucination (幻覺)**
    - **嚴謹定義**：語言模型生成看似合理、流暢，但實際上與事實不符或缺乏現實基礎的內容的現象。這種現象源於模型本身的生成機制，它是在預測下一個最可能的詞，而非在陳述事實。
    - **白話重述**：LLM 是一個很會「腦補」的學生。當它對某個知識點不確定時，它不會承認「我不知道」，而是會根據自己讀過的大量文本，編造一個最「像」答案的答案。這個編出來的答案可能語法通順、術語專業，但根本就是錯的。在醫學和科學領域，這種幻覺是極度危險的。
    - **常見誤解**：以為 `hallucination` 是模型的一個 bug，可以被輕易修復。事實上，`hallucination` 是目前生成式 AI 的內在屬性。我們能做的是透過 `RAG`、`prompting` 和事實查核來「減輕」它，但無法完全「根除」它。

4.  **Critical Thinking (批判性思維)**
    - **嚴謹定義**：在接收資訊時，不被動全盤接受，而是透過分析、評估、質疑和綜合，來形成自己判斷的思維過程。在人機協作中，指人類專家對 AI 產出內容的準確性、相關性、邏輯連貫性和潛在偏見進行審核與驗證的角色。
    - **白話重述**：就是當 AI 助理交給你一份報告時，你不能直接複製貼上。你要像個資深的主治醫師在 review intern寫的病歷一樣，逐字逐句檢查：這個數據來源可靠嗎？這個推論合理嗎？有沒有漏掉什麼重要的可能性？你才是最終負責的人。
    - **為何重要**：在 AI Agent 時代，`Critical Thinking` 不再只是一個軟實力，而是所有知識工作者必備的核心專業技能。AI 負責「生成」，人類負責「決策」。缺乏批判性思維，就等於把方向盤交給一個有時會產生幻覺的司機。

5.  **Automated Science (自動化科學)**
    - **嚴謹定義**：一個設想中的未來科研範式，其中 AI 系統不僅能協助人類，更能自主地執行整個科學發現的循環：從觀察現象、形成假說、設計並執行實驗、分析結果，到得出結論並迭代下一個研究問題。
    - **白話重述**：終極夢想是打造一個「AI 科學家」。你只要給它一個大方向（例如「找到治癒癌症的方法」），它就會自己去讀文獻、提出新的藥物靶點、在虛擬環境中模擬藥物效果、甚至控制實驗室的自動化機械手臂進行濕實驗 (wet-lab experiment)，然後把實驗報告給你。
    - **目前狀態**：我們離完全的 `Automated Science` 還非常遙遠。目前的 AI Agent 主要在「數位世界」中處理資訊型任務（稱為 `in silico`），例如文獻分析。要讓 AI Agent 能夠可靠地設計並操作「物理世界」的實驗（`in vitro`, `in vivo`），還需要克服大量的工程與安全挑戰。

6.  **OpenEvidence / Elicit / SciSpace**
    - **嚴謹定義**：這些都是現實世界中，專為學術研究設計的 AI 工具或平台，體現了本堂課討論的許多概念。
    - **白話重述**：
        - `OpenEvidence`: 專攻醫學領域，能針對臨床問題進行大規模文獻檢索與證據合成，產出類似 Systematic Review 的報告。
        - `Elicit`: 一個 AI 研究助理，其特色在於能將你的研究問題轉化為可執行的任務，並從大量論文中提取結構化資料，例如找出論文中的主要發現、實驗設計、研究對象等。
        - `SciSpace`: 提供了一個整合性的研究平台，能讓你上傳 PDF 論文後，透過 AI 進行問答、摘要和解釋，加速你理解論文的速度。
    - **為何重要**：這些工具不再是理論，而是已經可以實際使用的產品。它們是 AI Agent 在學術研究領域應用的具體實例，讓我們可以親身體驗這項技術的潛力與局限。

---

## §4. System / Paper Deep Dive

為了具體理解 AI Agent 如何執行學術研究，我們來設計一個假想的「自動化文獻回顧代理人」（Automated Literature Review Agent）系統。這個系統的設計靈感，源自於課堂中提到的 `OpenEvidence`、`Elicit` 等工具背後的運作邏輯。

### 4.1 Architecture

這個 Agent 的核心是一個以 LLM 為基礎的控制器，它能協調多個專用工具來完成一個複雜的文獻回顧任務。其工作流程可以用下面的圖來表示：

```mermaid
graph TD
    A[User: 輸入研究問題] --> B{Agent Core (LLM Controller)};
    B --> C{1. 任務規劃};
    C --> D[1a. 分解問題];
    C --> E[1b. 選擇工具序列];

    subgraph "工具執行循環 (Tool Execution Loop)"
        E --> F{2. 工具選擇};
        F -- "需要搜尋" --> G["Search Tool (Scopus/PubMed API)"];
        F -- "需要讀取" --> H[PDF Reader Tool];
        F -- "需要分析" --> I[Code Interpreter];
        F -- "任務完成" --> K{3. 結果合成};
    end

    G -- "論文列表" --> B;
    H -- "論文內容" --> B;
    I -- "分析結果" --> B;
    
    B -- "根據工具回傳更新計畫" --> C;

    K -- "生成摘要報告" --> L[最終報告];
    L --> M[User: 批判性驗證];
```

**架構說明**：
1.  **User Input**：研究者輸入一個自然語言的研究問題，例如：「比較 SGLT2 inhibitors 和 GLP-1 agonists 對於第二型糖尿病合併心衰竭患者的治療效果」。
2.  **Agent Core (LLM Controller)**：這是系統的大腦。它接收到問題後，首先進行 **任務規劃 (Planning)**。
3.  **任務規劃 (Planning)**：
    - **分解問題**：將複雜問題拆解成多個子問題，例如：「SGLT2i 的相關試驗有哪些？」、「GLP-1a 的相關試驗有哪些？」、「有沒有頭對頭比較 (head-to-head) 的研究？」。
    - **選擇工具序列**：規劃出一個執行的步驟，例如 `[Search(SGLT2i), Search(GLP-1a), Read(papers), Synthesize]`。
4.  **工具執行循環 (Tool Execution Loop)**：Agent 按照計畫，一步步呼叫工具。
    - **Search Tool**：呼叫學術資料庫的 API，使用 LLM 生成的關鍵字（如 `SGLT2 inhibitors AND heart failure`）進行搜尋，取回論文列表。
    - **PDF Reader Tool**：當需要精讀某篇論文時，Agent 會使用此工具讀取論文的 PDF 檔案，將其轉換為純文字。
    - **Code Interpreter**：如果需要進行數據分析或統計，Agent 可以生成 Python 程式碼並在此環境中執行。
5.  **結果合成 (Synthesis)**：當所有資訊蒐集完畢，Agent Core 會整合所有工具的產出、對話歷史和原始問題，生成一份結構化的摘要報告。
6.  **批判性驗證 (Human Verification)**：最終產出的報告必須由人類專家進行嚴格審查，確認其準確性、完整性與結論的合理性。

### 4.2 關鍵演算法

Agent 的核心決策邏輯可以用一段類似 ReAct 框架的偽程式碼來表示。這段程式碼模擬了 Agent 如何進行「思考-行動-觀察」的循環。

```python
# Pseudocode for the research agent's reasoning loop

def research_agent(initial_question: str):
    # Initialize memory with the user's question
    memory = f"Objective: Answer the question '{initial_question}'"
    
    # Maximum number of steps to prevent infinite loops
    for _ in range(MAX_STEPS):
        # THOUGHT step: The LLM thinks about what to do next
        prompt = f"""
        {memory}
        
        You have access to the following tools:
        - search_scopus(query: str) -> list_of_papers
        - read_pdf(url: str) -> paper_content
        - finish(answer: str) -> final_response

        Based on the objective and the history, what is your next action?
        Think step-by-step.
        """
        
        thought_process = llm.generate(prompt) # e.g., "I need to find relevant papers first."
        
        # ACTION step: The LLM decides which tool to use and with what arguments
        action_json = llm.generate_action(thought_process) # e.g., {"tool": "search_scopus", "query": "SGLT2i AND heart failure"}
        
        if action_json['tool'] == 'finish':
            return action_json['answer']

        # OBSERVATION step: Execute the chosen tool and get the result
        tool_output = execute_tool(action_json['tool'], action_json['query'])
        
        # Update memory with the result of the action
        memory += f"\nObservation: I used {action_json['tool']} and got the following result: {tool_output}"

    return "Agent could not finish the task within the step limit."

```

**演算法解釋**：
- 這個循環的核心是 `memory`，它記錄了到目前為止的所有步驟和觀察結果。
- 在每一次循環中，Agent 首先進行 **`Thought`**，讓 LLM 檢視 `memory` 並思考下一步該做什麼。這是一個內部的、不出聲的獨白。
- 接著，它做出 **`Action`** 的決定，選擇一個工具並提供參數。這個決策被格式化為 JSON，以便程式解析。
- 系統執行這個 `Action`，並將結果（**`Observation`**）回傳。
- 這個新的 `Observation` 被附加到 `memory` 的尾端，構成下一次循環的輸入。
- 這個過程不斷重複，直到 Agent 認為它已經收集到足夠的資訊，可以回答最初的問題，此時它會呼叫 `finish` 工具，結束任務。

### 4.3 關鍵 data structure

在文獻回顧的任務中，一個關鍵的資料結構是最終產出的「證據表」(Evidence Table)。Agent 需要將非結構化的論文內容，轉換成這種結構化的表格，以利於人類專家快速比較與評估。

| Paper (Link to PDF) | Year | Study Design | Population (N) | Intervention | Comparator | Key Outcomes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EMPEROR-Reduced | 2020 | RCT | 3730 | Empagliflozin | Placebo | **Primary**: CV death or HHF. **Result**: HR 0.75 (0.65-0.86), p<0.001 |
| DAPA-HF | 2019 | RCT | 4744 | Dapagliflozin | Placebo | **Primary**: Worsening HF or CV death. **Result**: HR 0.74 (0.65-0.85), p<0.001 |
| ... | ... | ... | ... | ... | ... | ... |

*RCT: Randomized Controlled Trial, HHF: Hospitalization for Heart Failure, HR: Hazard Ratio*

這個表格清晰地展示了每篇研究的核心元素，極大地方便了後續的 `meta-analysis` 或臨床決策。AI Agent 的價值就在於能自動化地從數十篇 PDF 中填寫這個表格。

### 4.4 Walkthrough

#### 情境一：正常運作 (Happy Path)

1.  **User**：「請總結 `dapagliflozin` 用於 `HFrEF` (Heart Failure with reduced Ejection Fraction) 的主要臨床試驗證據。」
2.  **Agent (Thought)**：「Okay，我需要搜尋關於 `dapagliflozin` 和 `HFrEF` 的臨床試驗。我應該使用 `search_scopus` 工具。」
3.  **Agent (Action)**：`search_scopus(query="dapagliflozin AND HFrEF AND randomized controlled trial")`
4.  **Observation**：API 回傳了 15 篇論文，其中一篇標題為 "Dapagliflozin in Patients with Heart Failure and Reduced Ejection Fraction" (DAPA-HF) 看起來最相關。
5.  **Agent (Thought)**：「DAPA-HF 試驗是關鍵。我需要讀取它的內容來提取 PICO (Population, Intervention, Comparator, Outcome)。我將使用 `read_pdf` 工具。」
6.  **Agent (Action)**：`read_pdf(url="<link_to_dapa_hf_pdf>")`
7.  **Observation**：工具回傳了 DAPA-HF 論文的全文純文字。
8.  **Agent (Thought)**：「現在我有了全文，我需要從中找出研究設計、樣本數、主要終點和結果。我將仔細閱讀摘要和方法學部分，然後填寫證據表。」
9.  **Agent (Action)**：`finish(answer="...[生成包含 DAPA-HF 摘要的 Markdown 表格]...")`
10. **Final Output**：產出一份精確的摘要表格，正確列出 DAPA-HF 的 HR 為 0.74，p value < 0.001。人類專家審核後確認無誤。

#### 情境二：異常／失敗案例 (Failure Case)

1.  **User**：「`Dapagliflozin` 和 `Empagliflozin` 哪個對 `HFrEF` 比較好？」 (這是一個需要頭對頭比較的問題)
2.  **Agent (Thought)**：「我需要找 `dapagliflozin` vs `empagliflozin` 的頭對頭比較試驗。」
3.  **Agent (Action)**：`search_scopus(query="dapagliflozin AND empagliflozin AND HFrEF AND head-to-head")`
4.  **Observation**：API 回傳 0 篇論文。（因為現實中當時沒有直接比較兩者的大型試驗）
5.  **Agent (Thought - **錯誤的推理**)**：「沒有頭對頭研究。沒關係，我可以分別看它們各自的試驗，然後比較它們的 Hazard Ratio。DAPA-HF 的 HR 是 0.74，EMPEROR-Reduced 的 HR 是 0.75。看起來 0.74 比 0.75 好一點。」
6.  **Agent (Action - **產生幻覺**)**：`finish(answer="根據 DAPA-HF (HR=0.74) 和 EMPEROR-Reduced (HR=0.75) 的數據，Dapagliflozin 的效果略優於 Empagliflozin。")`
7.  **Critical Failure**：這個結論是**極度錯誤且危險的**。在沒有直接頭對頭比較的情況下，跨試驗比較 (cross-trial comparison) 是臨床研究的大忌，因為兩個試驗的病人族群、研究設計、事件率都可能不同。AI Agent 犯了這個典型的統計學錯誤。
8.  **Human Verification**：幸好，人類專家立刻識破了這個謬誤。他駁回了 Agent 的結論，並指示 Agent 應該回覆：「目前缺乏直接比較兩者療效的頭對頭臨床試驗，因此無法判定何者較優。」

這個失敗案例完美地凸顯了人類專家在 `loop` 中不可或缺的 `Critical Thinking` 角色。AI 擅長快速執行和整合，但缺乏領域的 `common sense` 和對研究方法學的深刻理解。

## §5. 真實類比（★ 讀者背景特化）

這一節是本講義的核心價值所在，我們將用醫學院與臨床工作的情境，來類比 AI Agent 在學術研究中的工作模式。這些類比旨在幫助你建立直覺，但務必留意每個類比的極限。

### 類比一：AI 研究 Agent 如同「跑研究 block 的第一年住院醫師（R1）」

想像一位剛結束 PGY（畢業後一般醫學訓練）、對臨床充滿熱情但對研究方法尚在摸索的 R1，被指派到一個研究型的科別輪訓。指導他的主治醫師（Attending Physician）就是研究計畫的主持人（Principal Investigator, PI）。

**類比情境描述**：
PI 交給 R1 一個研究題目，例如：「分析本院過去五年，特定抗生素使用與 Acute Kidney Injury（AKI）的關聯性」。PI 不會手把手教 R1 怎麼做，只會給出大方向和最終目標（例如：年底前投稿到某個等級的期刊）。這位 R1 就必須自己動手，規劃時程、查文獻、申請 IRB（人體試驗委員會）許可、跟資訊室要資料、用統計軟體分析，最後撰寫成論文初稿。過程中，他會定期跟 PI 報告進度，PI 會給予 feedback，可能會質疑他的統計方法、或要求他補充更多文獻。R1 必須根據這些 feedback 反覆修改，直到產出 PI 滿意的結果。這個過程，就非常像一個 AI Agent 在執行一個複雜的學術研究任務。

| AI Agent 概念 | 臨床研究類比：R1 住院醫師 |
| :--- | :--- |
| **使用者 (User)** | **主治醫師 / PI (Attending / Principal Investigator)** |
| **AI Agent** | **第一年住院醫師 (R1)** |
| **LLM 核心模型** | **R1 的醫學基礎知識** (醫學院六年所學) |
| **Prompt (指令)** | **PI 交代的初步研究方向** ("去研究一下抗生素跟 AKI 的關係") |
| **Agent 的規劃能力 (Planning)** | **R1 撰寫的研究計畫書 (Proposal)** (包含文獻回顧、收案條件、統計方法) |
| **工具使用 (Tool Use)** | **R1 使用各種研究工具** (PubMed 查文獻、醫院 HIS/EMR 撈資料、SPSS/R 跑統計) |
| **長期記憶 (Long-term Memory)** | **R1 的個人筆記 / Zotero 文獻庫** (紀錄讀過的 paper 和 PI 的指示) |
| **自我修正 (Self-correction)** | **R1 根據 PI 的 feedback 修改論文** (M&M conference 後的流程改善) |
| **最終產出 (Final Output)** | **論文初稿 (Manuscript draft)** |

**✅ 吻合之處（為何類比有效）**
這個類比最貼切的地方在於「任務拆解」與「工具使用」的自主性。PI 給的是一個高層次的目標，而 R1 必須自主地將其分解成一系列可執行步驟：先去 PubMed 查前人做了什麼（`search` tool），然後設計關鍵字去 EMR 撈病人名單（`database` tool），接著把資料匯出成 CSV 檔用 R 跑統計模型（`code_interpreter` tool），最後把結果整理成圖表和文字（`text_generator` tool）。這完美對應了 AI Agent 的 `plan-and-execute` 循環。此外，PI 的 feedback 機制也類似於 Agent 的 `self-correction` 迴圈，都是在一個外部監督下，對自己的產出進行迭代式改進。

**⚠️ 不吻合之處（類比邊界，避免誤導）**
最大的差異在於「學習與成長」。R1 在跑研究的過程中，其內在的醫學知識和研究能力會真正地「成長」，他會從錯誤中學習，下次遇到類似問題能做得更好。目前的 AI Agent，其核心 LLM 是靜態的，它在一次任務中的「學習」僅限於該任務的 context window 內，任務結束後，除非進行 fine-tuning，否則模型本身不會因為犯了錯而變得更「聰明」。此外，R1 具備真正的臨床判斷（clinical judgment）和倫理考量，例如他會質疑資料的隱私問題；而 AI Agent 目前仍缺乏這種高層次的抽象推理與價值判斷能力。

### 類比二：多 Agent 協作如同「癌症多專科團隊會議（Tumor Board）」

單一 AI Agent 就像一個 R1，能力有限。但當我們把多個專門化的 Agent 組合起來，就形成了一個強大的研究團隊，這就好比醫院裡為了複雜癌症病人召開的 Tumor Board。

**類比情境描述**：
一位新診斷的複雜胰臟癌病人，主治醫師（通常是腫瘤內科）會將此案提到 Tumor Board。與會者包括：腫瘤內科醫師、外科醫師、放射科醫師、病理科醫師、放射腫瘤科醫師、個案管理師等。會議中，放射科醫師會展示影像，報告腫瘤侵犯範圍（`Data Analyst Agent`）；病理科醫師會呈現切片染色結果，判斷癌症分型（`another Data Analyst Agent`）；腫瘤內科醫師會根據最新的臨床試驗，提出化療或標靶藥物建議（`Literature Search Agent`）；外科醫師評估手術切除的可能性（`Feasibility Assessment Agent`）。整個會議由主席（`Orchestrator Agent`）引導，確保大家聚焦，並在最後形成一個整合所有專家意見的共識治療計畫，再由住院醫師或個管師記錄下來（`Writer Agent`）。

| Multi-Agent 概念 | 臨床類比：Tumor Board |
| :--- | :--- |
| **協調者 Agent (Orchestrator)** | **會議主席 / 主治醫師** |
| **文獻檢索 Agent** | **臨床藥師 / 專科護理師** (負責查找最新指引與試驗) |
| **數據分析 Agent** | **放射科醫師 / 病理科醫師** (解讀影像與病理報告) |
| **寫作 Agent (Writer)** | **記錄會議的住院醫師 / 個管師** |
| **Agent 間的通訊 (Communication)** | **會議中的口頭報告與討論** |
| **共享工作空間 (Shared Workspace)** | **投影幕上的 PACS 影像與病歷摘要** |
| **最終協作產出** | **多專科共識治療計畫 (Consensus treatment plan)** |

**✅ 吻合之處（為何類比有效）**
這個類比完美詮釋了「分工合作 (Division of Labor)」的核心精神。沒有任何一位醫師能獨自處理這個複雜個案，每個人都貢獻自己最專業的部份。這對應了 Multi-Agent 系統的設計哲學：與其打造一個巨大的、什麼都會但都不精的「萬能 Agent」，不如設計多個小而美的「專家 Agent」（例如一個專門讀 PDF、一個專門跑 code、一個專門寫作），再由一個「協調者 Agent」來統籌調度。Tumor Board 的會議流程——報告、討論、整合、決議——也酷似 CAMEL (Communicative Agents for Machine Learning) 這類框架中，Agent 之間輪流「發言」、貢獻想法、最終達成目標的模式。

**⚠️ 不吻合之處（類比邊界，避免誤導）**
目前 Agent 之間的「通訊」遠比 Tumor Board 的真人互動來得原始。Agent 之間通常是傳遞結構化的資料（如 JSON），而非進行真正的語意層面的辯論、協商或說服。放射科醫師可能會和外科醫師為了「是否能切乾淨」而有激烈辯論，這種基於經驗、直覺和風險評估的動態互動，是當前 Agent 系統還無法複製的。此外，Tumor Board 的決策涉及沉重的倫理和病人價值觀考量，而 Agent 的「決策」純粹是基於邏輯和任務成功率。

### 類比三：Agent 的 Hallucination 如同「醫學生的『瞎掰』式鑑別診斷」

AI Agent，尤其是其核心的 LLM，有時會產生「幻覺 (Hallucination)」。這在醫學情境中，就像一個知識不夠扎實的醫學生，在被老師問到鑑別診斷時，為了不被問倒而開始「瞎掰」。

**類比情境描述**：
在晨會上，老師報告一個主訴為「胸痛」的病人，問一位 Clerk（見習醫學生）：「請列出五個可能的鑑別診斷。」這位 Clerk 可能只想到了心肌梗塞和胃食道逆流，但為了湊滿五個，他開始瞎掰一些聽起來很專業但其實跟主訴關聯性很低的病名，例如「Acinetobacter 菌血症」或「Vitamin B12 缺乏」。這些答案並非完全錯誤的「醫學名詞」，但它們出現在這個 context 下，就顯得非常不合理，缺乏臨床邏輯。這就是一種「合理的廢話」，也是 LLM Hallucination 的常見形式。

| AI Agent 概念 | 臨床類比：醫學生瞎掰 |
| :--- | :--- |
| **語言模型 (LLM)** | **醫學生的腦內知識庫** |
| **Prompt** | **老師的提問** ("胸痛的鑑別診斷？") |
| **Hallucination (幻覺)** | **瞎掰的、不合邏輯的鑑別診斷** |
| **看似流暢但錯誤的輸出** | **聽起來很專業但完全不對勁的回答** |
| **Fact-checking (事實查核)** | **老師或資深醫師的糾正** ("你說 B12 缺乏，它跟胸痛的機轉是什麼？") |
| **Grounding (基於事實的生成)** | **基於病史、PE、EKG 等真實數據進行診斷** |

**✅ 吻合之 '處（為何類比有效）**
此類比精準地抓住了 Hallucination 的一個核心特徵：「看似合理，實則不然 (Plausible but non-factual)」。LLM 生成的幻覺文本，語法通常是流暢的，甚至引用的「文獻」格式也可能很逼真，就像醫學生能說出完整的醫學名詞一樣。問題出在內容與現實脫節。醫學生瞎掰的原因是他知識網路中的連結是脆弱或錯誤的；LLM 產生幻覺的根本原因，也是它在機率模型上找到了一條看似通順、但實際上沒有事實根據的路徑。老師透過追問「你的證據是什麼？」來進行事實查核，這也對應了 RAG (Retrieval-Augmented Generation) 系統中，要求 Agent 的每一句話都必須 `grounded` 在檢索到的真實文獻上。

**⚠️ 不吻合之處（類比邊界，避免誤導）**
醫學生瞎掰通常帶有「意圖」（不想丟臉），而 LLM 的幻覺是其模型內在機率分佈的副產物，沒有主觀意圖。更重要的是，一個有經驗的醫師能輕易識破醫學生的瞎掰，因為他們腦中有一個強大的世界模型（包括生理、病理、藥理知識）。而 AI Agent 在判斷另一個 Agent 的產出是否為幻覺時，若沒有可靠的外部工具（如真實的資料庫或 API）進行驗證，它自己也可能被「騙過」，因為它們都基於相似的底層模型。

## §6. 課堂 Q&A 精華

李宏毅教授的影片是精心設計的 monologue，其中穿插了許多他預期學生會感到困惑的觀念。我們將這些釐清的重點，整理成模擬的 Q&A 形式，幫助你掌握關鍵。

**Q1**: 聽起來 AI Agent 好像只是一個會自己上網查資料、然後把結果丟給 ChatGPT 的腳本 (script)，跟我們自己手動操作有什麼本質上的不同？
**A**: 這是一個非常好的問題，也是很多人初期的誤解。關鍵的不同在於「自主性」和「動態決策」。一個簡單的 script 是寫死的，它只能按照你預設的步驟執行，例如「1. 搜尋 Google、2. 抓取第一個連結、3. 摘要內容」。如果第一個連結是錯的，或需要登入才能看，script 就會失敗。但一個真正的 AI Agent 擁有「規劃 (planning)」和「自我修正 (self-correction)」的能力。它會根據初步的搜尋結果，動態決定下一步要做什麼。如果發現第一個連結無效，它會自主決定「放棄此路徑，改試第二個連結」或「更換關鍵字重新搜尋」。它更像一個懂得變通的實習生，而不是一個只會執行命令的機器人。

**Q2**: 影片中提到 Agent 可以做學術研究，這是否意味著研究人員未來會被取代？
**A**: 教授強調，現階段更準確的看法是「增強 (augmentation)」而非「取代 (replacement)」。AI Agent 的角色是一個極其強大的「研究助理」。它可以將研究人員從繁瑣、重複性的工作中解放出來，例如初步的文獻篩選、資料清理、程式碼撰寫等。這讓研究人員能更專注於提出創新的研究假設、設計實驗、以及解讀結果背後的科學意義這些更高層次的創造性工作。正如計算機沒有取代數學家，而是讓他們能處理更複雜的問題一樣，AI Agent 將會是頂尖研究人員的「能力放大器」。

**Q3**: Agent 的「規劃能力」聽起來很神奇，它到底是如何「思考」並制定出計畫的？
**A**: 這並不是神秘的魔法，而是基於 LLM 強大的語言理解和推理能力。像 ReAct (Reason + Act) 這類框架，就是讓 LLM 進行一種「內心獨白 (inner monologue)」。當接到任務時，Agent 會自問自答：「我現在的目標是什麼？」「為了達成這個目標，我手邊有哪些工具？」「根據現狀，我下一步最該做的是什麼？」。LLM 會生成一段包含「思考 (Thought)」、「行動 (Action)」、「觀察 (Observation)」的文字。這個「思考」過程就是它的規劃。例如，它會想：「嗯，我需要找一篇論文，所以我應該使用 `semantic_scholar_search` 這個工具。」這個決策過程雖然不完全等同於人類的意識，但在功能上實現了動態的任務拆解與規劃。

**Q4**: 如果 AI Agent 在研究過程中產生了 Hallucination（幻覺），例如引用了一篇不存在的論文，我們要如何發現並防止？
**A**: 這是目前 AI Agent 應用於嚴肅學術工作的最大挑戰之一。教授提到的解法是「Grounding (落地/溯源)」。一個設計良好的研究 Agent，其產出的每一句話，都必須能連結回一個真實的來源。例如，當它摘要一篇論文時，不僅要給出摘要，還要附上原始論文的 DOI 和段落索引。當它從資料庫中提取數據時，要能提供確切的查詢指令和結果。這就像寫論文時，每一個論點都要有 reference 支持一樣。未來的 Agent 框架會越來越強調「可驗證性 (verifiability)」，讓使用者可以輕易地追溯 Agent 的每一步思考路徑和資訊來源，從而揪出幻覺。

**Q5**: 當 Agent 使用外部工具（如程式碼直譯器）時，如果工具執行出錯，Agent 會怎麼辦？它會卡住嗎？
**A**: 一個成熟的 Agent 不會輕易卡住，這就是它與簡單 script 的另一個關鍵區別。當它嘗試執行一段 Python code 來畫圖，但 code 卻報錯了 (exception)，這個錯誤訊息會被當成「觀察 (Observation)」回傳給 Agent。Agent 的 LLM 核心會「讀懂」這個錯誤訊息，例如 "ImportError: No module named 'seaborn'"。接著，在下一個「思考 (Thought)」步驟中，它會推理：「喔，看起來是少了 `seaborn`這個套件。」然後它會生成一個新的「行動 (Action)」：「執行 `pip install seaborn`」。這就是一個完整的「從錯誤中恢復 (error recovery)」的循環，也是 Agent 魯棒性 (robustness) 的關鍵。

**Q6**: 讓 AI Agent 擁有寫程式和執行程式的能力，聽起來有點危險，會不會有安全問題？
**A**: 絕對有，而且這是整個領域最重視的問題之一。讓一個 AI 模型擁有自主執行程式碼的能力，無異於給了它一把操作電腦的鑰匙。因此，「沙盒 (Sandbox)」環境是不可或缺的。任何由 Agent 生成並執行的程式碼，都必須在一個與主系統嚴格隔離的環境中運行。這個沙盒環境應該有嚴格的限制：不能存取本地檔案系統（除非明確授權）、不能建立任意的網路連線、CPU 和記憶體使用量也受到嚴格控管。這確保了即使 Agent 因為 prompt injection 或其他原因「失控」，試圖執行惡意程式碼（如 `rm -rf /`），它也只能在沙盒這個「模擬的小房間」裡搞破壞，無法影響到使用者真實的作業系統。

**最常見誤解 Top 3**
1.  **誤解**: AI Agent = 自動化腳本。**釐清**: Agent 核心是動態規劃與修正，而非固定的 if-then 邏輯。
2.  **誤解**: Agent 會很快取代所有知識工作者。**釐清**: Agent 是「增強」工具，將人類從繁瑣工作中解放，專注於創造與決策。
3.  **誤解**: Agent 的思考是無法解釋的黑盒子。**釐清**: 透過 Chain-of-Thought 和日誌，Agent 的每一步推理和決策都是可以被追蹤和審計的。

## §7. 常見陷阱與考點（What Engineers Actually Get Wrong）

即使理解了核心概念，在實際建構或使用 AI Agent 時，仍然有很多容易踩進的坑。以下是教授在演講中暗示或業界公認的常見陷阱。

**陷阱一：過度信任 Agent 的文獻綜述**
- **為何會掉進去**：Agent 可以在幾分鐘內生成一份看似完美的文獻回顧 (literature review)，引用數十篇論文，格式工整，誘使研究人員直接採納。
- **正確做法**：將 Agent 的產出視為「初稿」而非「終稿」。**務必**手動抽查其中 10-20% 的關鍵引用，回到原始 PDF 確認 Agent 的摘要沒有曲解原意、斷章取義，或產生 subtle 的幻覺。尤其要注意 Agent 對研究方法和結果數值的總結是否精確。
- **實例**：要求 Agent 總結關於某藥物副作用的文獻，它可能回報「多數研究顯示無顯著副作用」，但實際上它可能忽略了幾篇指出在特定族群（如腎功能不佳者）有嚴重副作用的關鍵論文。

**陷阱二：忽略 Agent 的 Token 成本**
- **為何會掉進去**：Agent 的思考過程 (Chain-of-Thought) 和工具使用日誌會產生大量的 token。一次複雜的任務，其 context 可能會累積到數十萬甚至上百萬 token，導致 API 費用遠超預期。
- **正確做法**：設計 Agent 時必須有「成本意識」。使用更小的模型來執行分類、路由等簡單任務；為 Agent 的思考迴圈設定最大步數限制；定期修剪 (prune) context，只保留最重要的資訊；使用 streaming 模式來監控 token 消耗。
- **實例**：一個研究 Agent 在網路上漫無目的地「衝浪」，從一個連結跳到另一個連結，試圖找到答案。如果沒有設定停止條件，它的 context 會像滾雪球一樣越來越大，一小時後你可能會收到一張昂貴的 API 帳單。

**陷阱三：在不安全的環境中執行 Agent 生成的程式碼**
- **為何會掉進去**：為了方便，開發者有時會直接在自己的本機上執行 Agent 生成的 Python 程式碼，而沒有建立適當的沙盒。
- **正確做法**：**永遠**在嚴格隔離的 Docker container 或類似的沙盒環境中執行 Agent 的程式碼。這個環境應該沒有網路存取權限（或只有白名單）、沒有檔案系統存取權限，並對運算資源設限。
- **實例**：使用者要求 Agent「幫我整理下載資料夾」，一個惡意的 prompt 可能會誘使 Agent 生成並執行 `rm -rf ~/Downloads` 的指令，如果沒有沙盒，後果不堪設想。

**陷阱四：對 Agent 的「常識」有不切實際的期望**
- **為何會掉進去**：因為 LLM 在很多方面表現得像人類，我們下意識地會假設它擁有和我們一樣的常識 (common sense) 和領域知識 (domain knowledge)。
- **正確做法**：明確地在 prompt 中提供所有必要的 context 和背景知識，即使它對你來說是「常識」。把 Agent 當成一個非常聰明、博學，但完全沒有你的專業領域經驗的實習生。
- **實例**：你告訴一個研究 Agent「分析這份『流感』數據」，並期望它知道要按北半球/南半球、年份、病毒株亞型進行分層分析。但如果你不把這些具體的分析維度寫進 prompt，它很可能只會給出一個籠統的總體統計。

**陷阱五：設計的工具 (Tools) 過於複雜或過於簡單**
- **為何會掉進去**：設計過於簡單的工具（如只會回傳 `True`/`False` 的 API）能給 Agent 的資訊太少；設計過於複雜的工具（需要十幾個參數的函數）則會讓 Agent 難以學會如何正確使用。
- **正確做法**：設計工具時要站在 Agent 的角度思考。每個工具應該功能單一、目的明確 (single-purpose)。函數的命名要直觀（如 `search_papers_by_keyword`），參數要少且有意義，並且一定要提供清晰的 docstring，解釋這個工具做什麼、需要什麼輸入、回傳什麼輸出。
- **實例**：設計一個 `run_statistical_analysis` 工具，它需要 `data`、`model_type`、`variables`、`covariates`、`p_value_threshold` 等多個參數。Agent 很可能會搞錯參數的用法。更好的設計是將其拆分為 `run_t_test(group1, group2)`、`run_linear_regression(x, y)` 等更具體的工具。

**陷阱六：陷入「規劃-執行」的死迴圈**
- **為何會掉進去**：當 Agent 遇到一個它無法解決的問題時，可能會陷入一個不斷嘗試同樣失敗方法的迴圈。例如，不斷嘗試用同一個錯誤的關鍵字去搜尋。
- **正確做法**：在 Agent 的設計中加入「偵錯」和「狀態追蹤」機制。例如，如果 Agent 連續三次使用同一個工具得到同樣的錯誤結果，就應該強制它進入一個「反思 (reflection)」步驟，要求它評估目前的策略是否可行，並考慮從更高層次改變計畫，甚至向使用者求助。

## §8. 自測題（正好 10 題，附摺疊答案）

**1. (概念題)** 下列何者最能描述 AI Agent 與傳統自動化腳本 (script) 的核心區別？
A. AI Agent 使用更先進的程式語言。
B. AI Agent 能夠動態規劃、自我修正，並適應非預期的情況。
C. AI Agent 的執行速度遠快於傳統腳本。
D. AI Agent 必須在雲端運行，而腳本可以在本地運行。

<details><summary>展開答案</summary>

**答案：B**

**解釋**：AI Agent 的核心價值在於其「自主性」。傳統腳本遵循的是寫死的、固定的指令流程，遇到預期外的狀況（例如網頁改版、API 報錯）就會失敗。而 AI Agent 則像一個能獨立思考的助理，它會根據當前的情境和目標，動態地制定計畫 (Planning)，並在行動失敗時分析原因，嘗試用不同的方法來達成目標 (Self-correction)。選項 A、C、D 都不是本質區別。

</details>

**2. (概念題)** 在 ReAct (Reason + Act) 框架中，「Thought」步驟的主要作用是什麼？
A. 執行一個具體的工具（如 API call）。
B. 記錄從工具得到的外部觀察結果。
C. 進行內心獨白，分析當前狀況並決定下一步行動。
D. 生成最終給使用者的答案。

<details><summary>展開答案</summary>

**答案：C**

**解釋**：「Thought」是 ReAct 框架的精髓，是 Agent 「推理」能力的體現。在這個步驟中，LLM 會像一個偵探一樣自言自語，分析它已經擁有的資訊（來自之前的 Observation），評估距離目標還有多遠，然後決定下一步最合理的「行動 (Action)」是什麼。選項 A 是 Action 步驟，選項 B 是 Observation 步驟，選項 D 是任務完成時的最終輸出。

</details>

**3. (情境題)** 你要求一個研究 Agent 找出「Ozempic 對於第二型糖尿病患心血管事件的影響」。根據 §5 的類比，這個 Agent 的第一個合理行動 (Action) 應該是什麼？
A. 執行 `pip install tensorflow` 來準備建立模型。
B. 使用 `pubmed_search` 工具，關鍵字設為 "Ozempic cardiovascular events type 2 diabetes"。
C. 撰寫一份完整的 IRB 申請書。
D. 生成一份關於 Ozempic 藥理機轉的詳細報告。

<details><summary>展開答案</summary>

**答案：B**

**解釋**：如同一個 R1 接到研究題目後，第一步一定是先做文獻回顧，了解目前已有的研究。因此，Agent 最合理的起手式是使用文獻搜尋工具來探索現有的知識。選項 A、C、D 都是在對現有文獻有一定了解後才可能進行的下游步驟。直接建立模型或寫 IRB 太過草率，而直接寫報告則可能遺漏最重要的臨床試驗證據。

</details>

**4. (情境題)** 你的 AI Agent 在執行程式碼時，收到了 `FileNotFoundError` 的錯誤訊息。在一個設計良好的 Agent 中，這個錯誤訊息會被視為什麼，並觸發什麼反應？
A. 被視為任務徹底失敗，Agent 會停止運作。
B. 被 Agent 忽略，並嘗試執行下一步的程式碼。
C. 被視為一個「Observation」，Agent 會在下一個「Thought」步驟中分析此錯誤，並可能嘗試生成一個修正路徑的行動。
D. 被視為使用者輸入錯誤，Agent 會要求使用者提供正確的檔案路徑。

<details><summary>展開答案</summary>

**答案：C**

**解釋**：這是 Agent 魯棒性 (robustness) 的關鍵。外部工具的錯誤回饋，對於 Agent 來說是寶貴的「觀察」資訊。Agent 會將這個錯誤訊息納入其 context，並在下一步的推理中考慮它。一個好的 Agent 會思考：「喔，檔案不存在。可能是我搞錯了路徑，或是這個檔案需要先被生成。」然後它會採取行動來修正這個問題，而不是直接放棄或把問題丟回給使用者。

</details>

**5. (Debug 題)** 一個 Agent 的日誌顯示它連續五次執行 `search("latest cancer treatment")`，並且每次都得到幾乎相同的搜尋結果。這最可能暗示了 Agent 設計中的什麼缺陷？
A. LLM 的溫度 (temperature) 設為 0。
B. Agent 缺乏有效的「狀態追蹤」和「反思」機制。
C. `search` 工具本身有 bug。
D. Agent 的 context window 太小。

<details><summary>展開答案</summary>

**答案：B**

**解釋**：這種行為被稱為「卡在迴圈 (stuck in a loop)」。這表明 Agent 缺乏一個機制來意識到它正在重複做無用功。一個設計良好的 Agent 應該能夠追蹤自己的歷史行動，當它發現自己連續多次採取同樣的行動卻沒有進展時，就應該觸發一個「反思 (reflection)」或「策略改變」的程序，例如嘗試用不同的關鍵字搜尋，或者向使用者求助。

</details>

**6. (概念題)** 為什麼在讓 AI Agent 執行程式碼時，「沙盒 (Sandbox)」是一個絕對必要的安全措施？
A. 為了讓程式碼執行得更快。
B. 為了防止 Agent 寫出有語法錯誤的程式碼。
C. 為了隔離 Agent 的執行環境，防止它對主機系統造成非預期的、甚至惡意的修改。
D. 為了將程式碼自動翻譯成多種語言。

<details><summary>展開答案</summary>

**答案：C**

**解釋**：安全是 Agent 系統的頭等大事。讓一個 LLM 擁有自主執行程式碼的能力，就必須假設它可能被惡意利用（例如透過 prompt injection）來執行有害操作（如刪除檔案、竊取資料）。沙盒創建了一個與主機系統隔離的牢籠，Agent 所有的操作都被限制在這個牢籠內，從而保護了使用者的電腦安全。

</details>

**7. (情境題)** 你正在設計一個幫助醫師查詢臨床指引的 Agent。下列哪一個工具 (Tool) 設計是最好的？
A. `query(text)`：一個能回答任何問題的通用工具。
B. `get_guideline(disease_name, year, country)`：一個參數清晰、目標明確的工具。
C. `run_sql_on_guideline_database(sql_query)`：一個功能強大但要求 Agent 具備 SQL 知識的工具。
D. `download_pdf(url)`：一個過於底層的工具。

<details><summary>展開答案</summary>

**答案：B**

**解釋**：好的工具設計應該遵循「高內聚、低耦合」原則。工具的功能應該專一且明確。選項 B 的 `get_guideline` 命名清晰，參數 `disease_name`, `year`, `country` 對 LLM 來說非常直觀，容易學習使用。選項 A 太模糊，Agent 不知道它能做什麼。選項 C 要求 Agent 自己寫 SQL，這太困難且容易出錯。選項 D 太底層，Agent 需要先找到 URL 才能用，增加了不必要的步驟。

</details>

**8. (Debug 題)** 一個研究 Agent 在其報告中寫道：「根據 Smith 等人於 2023 年在《自然》期刊發表的論文，證實藥物 X 可以完全治癒阿茲海默症。」但你卻在網路上找不到這篇論文。這最可能是哪一種 Agent 常見的失敗模式？
A. 工具執行錯誤 (Tool Execution Error)
B. 幻覺 (Hallucination)
C. 任務規劃失敗 (Planning Failure)
D. 上下文長度超限 (Context Length Exceeded)

<details><summary>展開答案</summary>

**答案：B**

**解釋**：這是典型的幻覺，而且是學術應用中最危險的一種。LLM 基於其訓練數據中的模式，生成了一段看起來非常逼真、格式完全正確的引文，但其內容（作者、年份、期刊、結論）卻是憑空捏造的。這凸顯了對 Agent 產出進行嚴格事實查核 (fact-checking) 的必要性。

</details>

**9. (概念題)** 根據 §5 的「Tumor Board」類比，一個由多個專家 Agent 組成的系統，其主要優勢是什麼？
A. 執行速度比單一 Agent 更快。
B. 總體 Token 消耗更少。
C. 允許「分工合作」，讓每個 Agent 專注於其擅長的子任務，從而解決更複雜的問題。
D. 可以完全不需要人類監督。

<details><summary>展開答案</summary>

**答案：C**

**解釋**：多 Agent 系統的核心思想是「分工 (Division of Labor)」，就像人類的專家團隊一樣。與其試圖訓練一個什麼都會的巨大模型，不如組合多個小而精的專家 Agent（一個專門搜尋、一個專門分析數據、一個專門寫作），並由一個協調者來調度它們。這種模組化的方法讓系統更容易擴展、除錯，並且能處理單一 Agent 無法勝任的複雜任務。

</details>

**10. (情境題)** 你希望 Agent 幫你分析一份病患滿意度調查的 CSV 檔案。在你的 prompt 中，除了提供檔案路徑外，提供下列哪一項資訊對 Agent 的成功最為關鍵？
A. 你的電腦型號和作業系統版本。
B. 對 CSV 檔案中每一個欄位的詳細描述（例如，"column_5: overall_satisfaction, scale 1-5, 5 is best"）。
C. 你希望最終報告使用的字體和顏色。
D. 你過去發表過的所有論文列表。

<details><summary>展開答案</summary>

**答案：B**

**解釋**：這對應到「不要對 Agent 的常識有不切實際的期望」這個陷阱。你不能假設 Agent 能自動「看懂」你的資料欄位代表什麼意義。提供一個清晰的「資料字典 (data dictionary)」是至關重要的。這會幫助 Agent 正確地理解資料、選擇合適的統計方法，並生成有意義的分析結果。缺少這份詮釋，Agent 很可能只會回傳一些基本的、不具洞見的描述性統計。

</details>

## §9. 延伸資源

- **本堂對應核心 Paper**：
    - **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao, et al., 2022): 這是提出 "Thought, Action, Observation" 循環的開創性論文，是理解現代 Agent 內部運作邏輯的必讀經典。
    - **A Survey on Large Language Model based Autonomous Agents** (Wang, et al., 2023): 一篇非常全面的綜述論文，整理了 Agent 的架構、關鍵組件、應用和挑戰，適合想對整個領域有鳥瞰式理解的讀者。

- **推薦延伸閱讀**：
    - **Lilian Weng 的部落格文章: "LLM-powered Autonomous Agents"**: 業界公認的經典入門文章，用非常清晰的架構圖和解釋，說明了 Agent 的核心組件（Planning, Memory, Tool Use）。(網址: https://lilianweng.github.io/posts/2023-06-23-agent/)
    - **李宏毅教授相關課程: "Advanced NLP - Prompt Engineering & P-tuning"**: 如果你對 Agent 如何與 LLM 互動的底層技術感興趣，了解 Prompt Engineering 的各種技巧會非常有幫助。

- **下一堂預告**：
    我們已經了解了單一 Agent 如何規劃、行動與學習，也探討了它對學術研究的衝擊。但是，當一個 Agent 不夠用時，我們要如何組織一個「Agent 團隊」？下一堂課，我們將深入探討 **Multi-Agent Systems**，學習如何讓多個 Agent 彼此溝通、協作甚至競爭，以解決更宏大的挑戰。
