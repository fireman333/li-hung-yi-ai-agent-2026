// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeMermaid from 'rehype-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://fireman333.github.io',
  base: '/li-hung-yi-ai-agent-2026',
  integrations: [
    starlight({
      title: '李宏毅 AI Agent 2026 課程筆記',
      description: '李宏毅老師 2026 年 AI Agent / Harness Engineering / Self-Correction 系列課程之 AI 自動生成學習筆記',
      defaultLocale: 'zh-tw',
      locales: { 'zh-tw': { label: '繁體中文', lang: 'zh-TW' } },
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
          items: [{ autogenerate: { directory: '.' } }],
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
