#!/usr/bin/env node
// Idempotent migration: add YAML frontmatter to lectures/*.md files generated
// by youtube-playlist-to-textbook skill. Skips files that already have
// frontmatter. Safe to re-run after each `generate_lecture.py` invocation.

import { readdir, readFile, writeFile, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..');
const LECTURES_DIR = join(REPO_ROOT, 'lectures');
const PLAYLIST_PATH = join(REPO_ROOT, 'lectures-raw', 'playlist.json');
const PROGRESS_PATH = join(REPO_ROOT, '.state', 'progress.json');

// Try to load playlist + progress to enrich frontmatter
async function loadJsonSafe(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch {
    return null;
  }
}

const playlist = (await loadJsonSafe(PLAYLIST_PATH)) ?? [];
const progress = (await loadJsonSafe(PROGRESS_PATH)) ?? {};

const playlistByIndex = new Map(playlist.map((p) => [p.index, p]));

function yamlEscape(s) {
  if (s == null) return '';
  if (typeof s !== 'string') return String(s);
  if (/[:#&*!|>'"%@`]/.test(s) || s.includes('\n') || /^\s|\s$/.test(s)) {
    return JSON.stringify(s);
  }
  return s;
}

function parseTopMetadata(body) {
  // Look for h1 + bullet block at the very top of the file.
  const lines = body.split('\n');
  let i = 0;
  while (i < lines.length && lines[i].trim() === '') i++;
  if (!lines[i] || !lines[i].startsWith('# ')) return null;

  const h1 = lines[i].replace(/^#\s+/, '').trim();
  i++;
  while (i < lines.length && lines[i].trim() === '') i++;

  const bullets = [];
  let bulletEnd = i;
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.startsWith('- ') || ln.startsWith('* ')) {
      bullets.push(ln);
      bulletEnd = i + 1;
      i++;
    } else if (ln.trim() === '' && bullets.length === 0) {
      i++;
    } else {
      break;
    }
  }

  // Extract metadata from bullets
  const meta = {};
  for (const b of bullets) {
    const m = b.match(/^[-*]\s+([^:：]+)[:：]\s*(.+)$/);
    if (!m) continue;
    const key = m[1].trim().toLowerCase();
    const val = m[2].trim();
    if (key === 'video' || key === 'youtube') meta.videoUrl = val;
    else if (key.includes('duration') || key.includes('時長') || key.includes('分鐘')) {
      const n = parseInt(val.match(/\d+/)?.[0] ?? '', 10);
      if (Number.isFinite(n)) meta.durationMin = n;
    } else if (key.includes('預計閱讀') || key.includes('reading')) {
      const n = parseInt(val.match(/\d+/)?.[0] ?? '', 10);
      if (Number.isFinite(n)) meta.readingMin = n;
    }
  }

  return { h1, meta, bulletEnd };
}

function deriveLectureNumber(filename) {
  const m = filename.match(/^(\d+)_/);
  return m ? parseInt(m[1], 10) : null;
}

function extractVideoId(url) {
  if (!url) return null;
  const m = url.match(/[?&]v=([\w-]+)/);
  return m ? m[1] : null;
}

async function migrateFile(filename) {
  const fullpath = join(LECTURES_DIR, filename);
  const src = await readFile(fullpath, 'utf8');

  if (src.startsWith('---\n')) {
    return { filename, status: 'skip-has-frontmatter' };
  }

  const lectureNumber = deriveLectureNumber(filename);
  if (lectureNumber == null) {
    return { filename, status: 'skip-no-lecture-number' };
  }

  const parsed = parseTopMetadata(src);
  if (!parsed) {
    return { filename, status: 'skip-no-h1' };
  }

  const { h1, meta, bulletEnd } = parsed;

  // Enrich from playlist.json if available
  const playlistEntry = playlistByIndex.get(lectureNumber);
  if (playlistEntry) {
    if (!meta.videoUrl) meta.videoUrl = playlistEntry.url;
    // Prefer playlist duration (always known to be in seconds from yt-dlp)
    if (playlistEntry.duration && Number.isFinite(playlistEntry.duration)) {
      meta.durationMin = Math.round(playlistEntry.duration / 60);
    }
  }

  // Sanity-fix: if duration looks like seconds (e.g. 2979), convert to minutes.
  if (meta.durationMin && meta.durationMin > 240) {
    meta.durationMin = Math.round(meta.durationMin / 60);
  }

  // uploadedAt = file birthtime (fallback mtime)
  let uploadedAt;
  try {
    const st = await stat(fullpath);
    const t = st.birthtime || st.mtime;
    uploadedAt = t.toISOString().slice(0, 10);
  } catch {}

  // Clean title: drop "Lecture NN:" prefix if present, use playlist title if better
  let title = h1.replace(/^Lecture\s+\d+:\s*/i, '').trim();
  if (playlistEntry?.title && title.length < playlistEntry.title.length * 0.4) {
    title = playlistEntry.title;
  }

  // Reading time heuristic: ~400 CJK chars / min for casual reading
  if (!meta.readingMin) {
    const cjkCount = (src.match(/[一-鿿]/g) || []).length;
    if (cjkCount > 0) {
      meta.readingMin = Math.max(5, Math.round(cjkCount / 400));
    }
  }

  const videoId = extractVideoId(meta.videoUrl);

  const fm = [
    '---',
    `title: ${yamlEscape(title)}`,
    `lectureNumber: ${lectureNumber}`,
    meta.videoUrl ? `videoUrl: ${yamlEscape(meta.videoUrl)}` : null,
    videoId ? `videoId: ${yamlEscape(videoId)}` : null,
    meta.durationMin ? `durationMin: ${meta.durationMin}` : null,
    meta.readingMin ? `readingMin: ${meta.readingMin}` : null,
    uploadedAt ? `uploadedAt: ${uploadedAt}` : null,
    `tableOfContents:`,
    `  minHeadingLevel: 2`,
    `  maxHeadingLevel: 3`,
    '---',
    '',
  ]
    .filter((line) => line !== null)
    .join('\n');

  const lines = src.split('\n');
  let body = lines.slice(bulletEnd).join('\n').replace(/^\n+/, '');
  // Strip a leading horizontal-rule line (a stray `---` that was a section divider
  // between the bullet metadata and the first heading) — it would render confusingly
  // right after the frontmatter close.
  body = body.replace(/^---\s*\n+/, '');
  const newContent = fm + body;
  await writeFile(fullpath, newContent, 'utf8');
  return { filename, status: 'migrated', title, lectureNumber };
}

const files = (await readdir(LECTURES_DIR))
  .filter((f) => /^\d+_.+\.md$/.test(f))
  .sort();

console.log(`Found ${files.length} lecture .md files in lectures/`);

const results = [];
for (const f of files) {
  results.push(await migrateFile(f));
}

const stats = results.reduce((acc, r) => {
  acc[r.status] = (acc[r.status] ?? 0) + 1;
  return acc;
}, {});
console.log('Result:', stats);
for (const r of results) {
  console.log(`  [${r.status}] ${r.filename}${r.title ? ` — ${r.title}` : ''}`);
}
