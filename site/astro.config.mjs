// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeMermaid from 'rehype-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LECTURES_DIR = resolve(__dirname, '../lectures');

/**
 * Generate Starlight sidebar items by reading lectures/*.md at build time.
 *
 * Starlight's `autogenerate: { directory: '.' }` only walks the physical
 * `src/content/docs/` tree. Since our docs collection uses a glob loader
 * pointing at `../lectures/`, autogenerate yields an empty list.
 *
 * This function reads the same lectures/ directory at config-load time and
 * emits explicit `{ label, slug }` items, preserving "drop new .md → sidebar
 * auto-updates" extensibility without a custom plugin.
 */
function loadLectureSidebar() {
  const files = readdirSync(LECTURES_DIR)
    .filter((f) => /^\d+_.+\.md$/.test(f))
    .sort();

  return files.map((f) => {
    const slug = f.replace(/\.md$/, '');
    const content = readFileSync(resolve(LECTURES_DIR, f), 'utf8');

    // Lightweight YAML frontmatter parsing — only title + lectureNumber.
    // Avoid adding gray-matter as a dependency for this small need.
    const titleMatch = content.match(/^title:\s*["']?(.+?)["']?\s*$/m);
    const numMatch = content.match(/^lectureNumber:\s*(\d+)\s*$/m);

    const num = numMatch ? numMatch[1].padStart(2, '0') : '??';
    let title = titleMatch ? titleMatch[1].trim() : slug;
    // Truncate long titles to keep sidebar narrow.
    if (title.length > 28) title = title.slice(0, 27) + '…';

    return {
      label: `Lec ${num} · ${title}`,
      slug,
    };
  });
}

// https://astro.build/config
export default defineConfig({
  site: 'https://fireman333.github.io',
  base: '/li-hung-yi-ai-agent-2026',
  integrations: [
    starlight({
      title: '李宏毅 AI Agent 2026 課程筆記',
      description: '李宏毅老師 2026 年 AI Agent / Harness Engineering / Self-Correction 系列課程之 AI 自動生成學習筆記',
      logo: {
        src: './src/assets/logo.svg',
        alt: '課程筆記',
      },
      // Single-locale site (zh-TW) served at the URL root. Using the literal
      // `root` key (not the locale code as key) tells Starlight not to prefix
      // routes with /zh-tw/, so the homepage logo / nav links land on `/` not
      // `/zh-tw/` (avoids 404 on logo click).
      defaultLocale: 'root',
      locales: { root: { label: '繁體中文', lang: 'zh-TW' } },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/fireman333/li-hung-yi-ai-agent-2026',
        },
        {
          icon: 'youtube',
          label: 'YouTube Playlist',
          href: 'https://www.youtube.com/playlist?list=PLJV_el3uVTsMOzDAUQ0cj9oLqEnqTSBkl',
        },
      ],
      sidebar: [
        {
          label: '所有講義',
          items: loadLectureSidebar(),
        },
      ],
      customCss: ['./src/styles/global.css'],
      pagefind: true,
      lastUpdated: true,
      components: {
        Footer: './src/components/SiteFooter.astro',
      },
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeMermaid, { strategy: 'pre-mermaid' }], rehypeKatex],
  },
});
