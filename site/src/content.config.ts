import { defineCollection, z } from 'astro:content';
import { docsSchema } from '@astrojs/starlight/schema';
import { glob } from 'astro/loaders';

const lectureSchema = z.object({
  lectureNumber: z.number().int().min(1).optional(),
  videoUrl: z.string().url().optional(),
  videoId: z.string().optional(),
  durationMin: z.number().int().optional(),
  readingMin: z.number().int().optional(),
  uploadedAt: z.coerce.date().optional(),
  seriesTitle: z.string().optional(),
  pdfUrl: z.string().optional(),
  summaryImage: z.string().optional(),
  tags: z.array(z.string()).default([]),
});

export const collections = {
  docs: defineCollection({
    loader: glob({ pattern: '*.{md,mdx}', base: '../lectures' }),
    schema: docsSchema({ extend: lectureSchema }),
  }),
};
