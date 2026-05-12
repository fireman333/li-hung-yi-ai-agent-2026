# 李宏毅 AI Agent 2026 課程筆記

> 國立臺灣大學 **李宏毅老師** 2026 年《AI Agent / Harness Engineering / Self-Correction》系列課程之 **AI 自動生成學習筆記**。
>
> 線上閱讀：<https://tony85314.github.io/li-hung-yi-ai-agent-2026/>
>
> 原始播放清單：<https://www.youtube.com/playlist?list=PLJV_el3uVTsMOzDAUQ0cj9oLqEnqTSBkl>

## 這是什麼

把 YouTube 講座透過 `gemini-2.5-pro` 整理成繁中、教科書級講義 markdown，再由 Astro Starlight 包成可全文搜尋的網站，部署到 GitHub Pages。

**內容來源 © 李宏毅老師**，本筆記僅供個人複習，請以 YouTube 原片為準。

詳細的講義導讀、學習路徑、三種讀法請見 [LECTURES_INDEX.md](./LECTURES_INDEX.md)。

## 加新講義 SOP

### 路徑 A — 教授上傳新影片到原 playlist

```bash
# 1. 重抓 playlist metadata（如果有新影片 / 順序改變）
bash ~/.claude/skills/youtube-playlist-to-textbook/scripts/fetch_playlist.sh \
  "https://www.youtube.com/playlist?list=PLJV_el3uVTsMOzDAUQ0cj9oLqEnqTSBkl"

# 2. 增量生成（skill incremental mode 自動只跑新集數）
python3 ~/.claude/skills/youtube-playlist-to-textbook/scripts/generate_lecture.py all --concurrent 2

# 3. 補 frontmatter（idempotent，舊講義不會被改）
node scripts/migrate-frontmatter.mjs

# 4. push → CI 自動部署
git add lectures/ lectures-raw/playlist.json
git commit -m "Add Lecture NN: <title>"
git push
```

> **⚠ 注意**：如果新影片插在中間（不是接在最後），會導致既有講義編號 shift。需要手動 rename `lectures/NN_*.md` + 改 `.state/progress.json` 的 `lec_NN` key + 改 `path` 欄位。本 repo 已有先例（2026-05-12 處理過 5→10 集擴充），看 git log 參考。

### 路徑 B — 手寫一篇講義

在 `lectures/` 新增 `NN_slug.md`，最頂端帶這樣的 frontmatter：

```yaml
---
title: 講義標題
lectureNumber: 11
videoUrl: https://www.youtube.com/watch?v=XXXXXXXXXXX
durationMin: 60
readingMin: 25
uploadedAt: 2026-06-01
---
```

然後 `git push` 即可（無需跑 migration）。

## 本機 dev

```bash
cd site
npm install
npx playwright install chromium   # mermaid build-time renderer
npm run dev                       # http://localhost:4321/li-hung-yi-ai-agent-2026/
```

`npm run build && npm run preview` 跑 production build 驗證。

## 部署

`push` 到 `main` 自動觸發 `.github/workflows/deploy.yml` → GitHub Pages，約 2-3 分鐘後生效。
首次部署前要去 GitHub repo **Settings → Pages → Source** 選 **GitHub Actions**（一次性）。

## Repo 結構

```
.
├── lectures/                 ← 講義 markdown (single source of truth)
│   ├── index.mdx             ← 首頁，渲染 LectureIndex 卡片牆
│   └── NN_slug.md            ← 一篇講義一檔，frontmatter 帶 lectureNumber
├── lectures-raw/             ← yt-dlp 抓的 playlist metadata + transcripts
├── scripts/
│   ├── migrate-frontmatter.mjs   ← idempotent: 補缺的 frontmatter
│   └── sync-pdfs.mjs             ← build 前 copy build/*.pdf → site/public/pdf/
├── site/                     ← Astro Starlight app
│   ├── astro.config.mjs
│   ├── src/
│   │   ├── content.config.ts ← glob loader, 直讀 ../lectures
│   │   ├── components/
│   │   │   ├── LectureIndex.astro
│   │   │   └── SiteFooter.astro
│   │   └── styles/global.css
│   └── public/pdf/           ← postbuild 同步進來的 PDF（gitignored）
├── .github/workflows/deploy.yml
├── generate.py / regenerate_lec4.py  ← 早期 skill copy；新 pipeline 走 ~/.claude/skills/
└── LECTURES_INDEX.md         ← 課程內容導讀（skill Stage 7 產出，含 mermaid 學習路徑圖）
```

## 致謝

- **李宏毅 老師** — 原始講座內容（國立臺灣大學電機工程學系）
- **Google Gemini** — 講義內容生成（`gemini-2.5-pro`）
- **Anthropic Claude Code** + `youtube-playlist-to-textbook` skill — pipeline orchestration
- **Astro Starlight** — 文件站架構
