#!/usr/bin/env node
// Copy publishable build artifacts (PDFs) into site/public/ so Astro emits them
// as downloadable static assets. Runs before `astro build`.

import { readdir, copyFile, mkdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..');
const BUILD_DIR = join(REPO_ROOT, 'build');
const TARGET_DIR = join(REPO_ROOT, 'site', 'public', 'pdf');

if (!existsSync(BUILD_DIR)) {
  console.log(`[sync-pdfs] No build/ folder — skipping (this is fine on a fresh checkout).`);
  process.exit(0);
}

await mkdir(TARGET_DIR, { recursive: true });

const entries = await readdir(BUILD_DIR);
let copied = 0;
for (const name of entries) {
  if (!name.toLowerCase().endsWith('.pdf')) continue;
  const src = join(BUILD_DIR, name);
  const st = await stat(src);
  if (!st.isFile()) continue;
  const dst = join(TARGET_DIR, name);
  await copyFile(src, dst);
  console.log(`[sync-pdfs] ${name} → site/public/pdf/`);
  copied++;
}
console.log(`[sync-pdfs] Copied ${copied} PDF(s).`);
