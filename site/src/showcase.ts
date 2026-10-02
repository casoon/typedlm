import { ansiToHtml } from '@casoon/pages-theme/ansi';
import type { ShowcaseExample } from '@casoon/pages-theme/showcase';

// Input: the example sources from the crate. Output: recorded runs (see
// examples/recorded/README.md). Model answers cannot be produced at build time.
const sources = import.meta.glob<string>('../../crates/typedlm/examples/*.rs', {
  query: '?raw',
  import: 'default',
  eager: true,
});
const recorded = import.meta.glob<string>('../../examples/recorded/*.txt', {
  query: '?raw',
  import: 'default',
  eager: true,
});

const entries = [
  {
    slug: 'classification',
    title: 'Classification',
    tags: ['enums', 'run', 'execute'],
    description:
      'A support ticket goes in, two enums come out. Recorded with qwen3:32b on Ollama.',
  },
  {
    slug: 'extraction',
    title: 'Invoice extraction',
    tags: ['nested types', 'Option', 'validate'],
    description:
      'Nested line items, amounts in cents and a domain rule: the items must add up to the net total. Recorded with qwen3:32b on Ollama.',
  },
  {
    slug: 'evaluation',
    title: 'Evaluation report',
    tags: ['dataset', 'ExactMatch', 'confidence interval'],
    description:
      'Eight labelled tickets, partially labelled. The interval shows how little eight examples say. Recorded with qwen3:32b on Ollama.',
  },
];

export const examples: ShowcaseExample[] = entries.map((entry) => {
  const source = sources[`../../crates/typedlm/examples/${entry.slug}.rs`];
  const output = recorded[`../../examples/recorded/${entry.slug}.txt`];
  return {
    ...entry,
    file: `crates/typedlm/examples/${entry.slug}.rs`,
    input: { code: source, lang: 'rust' },
    output: { html: ansiToHtml(output), kind: 'terminal' },
  };
});
