// @ts-check
import casoonPages from '@casoon/pages-theme';
import { defineConfig } from 'astro/config';

// Project page: https://casoon.github.io/typedlm/ — `base` is the GitHub Pages path.
export default defineConfig({
  site: 'https://casoon.github.io/typedlm',
  base: '/typedlm/',
  integrations: [
    casoonPages({
      name: 'TypedLM',
      description: 'Typed, testable LLM programs for Rust: typed contracts, validated answers, evaluation.',
      repo: 'casoon/typedlm',
      license: 'MIT',
      docsGroups: {
        'getting-started': 'Getting started',
        guides: 'Guides',
        concepts: 'Concepts',
        project: 'Project',
      },
    }),
  ],
});
