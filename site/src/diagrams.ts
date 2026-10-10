import { renderChart } from '@casoon/chartlet';

type Spec = Record<string, unknown>;

// Diagrams as chartlet specifications in site/diagrams/, rendered to HTML at build time.
const specs = import.meta.glob<Spec>('../diagrams/*.json', { import: 'default', eager: true });

/**
 * Each diagram is rendered twice — with chartlet's light and its dark theme — and CSS shows the
 * one matching the page. Overriding a few colour variables is not enough: flow charts fill their
 * nodes from role colours, and light text on a light node is unreadable.
 */
const SURFACE_STYLE = `<style>.cl-surface{padding:16px;border-radius:10px;margin-block:1.5rem}.cl-surface details{overflow-x:auto}.cl-light{background:#fff;color:#172033;color-scheme:light}.cl-dark{display:none;background:#0e131c;color:#e8edf6;color-scheme:dark}:root[data-theme=dark] .cl-light{display:none}:root[data-theme=dark] .cl-dark{display:block}@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .cl-light{display:none}:root:not([data-theme=light]) .cl-dark{display:block}}</style>`;

/** A diagram from site/diagrams/<name>.json. */
export function diagram(name: string): string {
  const spec = specs[`../diagrams/${name}.json`];
  if (!spec) throw new Error(`unknown diagram: ${name}`);
  return render(spec, name);
}

/** A diagram from a specification computed during the build. */
export function render(spec: Spec, name: string): string {
  const variant = (theme: 'light' | 'dark') => {
    const { content, warnings } = renderChart(
      { ...spec, theme },
      { format: 'html', table: 'details', idPrefix: `${name}-${theme}` },
    );
    for (const warning of warnings) console.warn(`[diagram ${name}, ${theme}] ${warning.trim()}`);
    return `<div class="cl-surface cl-${theme}">${content}</div>`;
  };
  return `${SURFACE_STYLE}${variant('light')}${variant('dark')}`;
}

/**
 * 95 % Wilson score interval, as typedlm's `Report` computes it for 0/1 scores
 * (crates/typedlm/src/eval.rs, `confidence_interval`).
 */
export function wilson(correct: number, n: number): [number, number] {
  const z = 1.96;
  const p = correct / n;
  const z2 = z * z;
  const center = (p + z2 / (2 * n)) / (1 + z2 / n);
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

/** How the interval of an 87.5 % exact-match score narrows as the dataset grows. */
export function intervalBySize(): string {
  const [low8, high8] = wilson(7, 8);
  // The recorded report (examples/recorded/evaluation.txt) shows 52.9 % – 97.8 % for 7 of 8.
  if (low8.toFixed(3) !== '0.529' || high8.toFixed(3) !== '0.978') {
    throw new Error(`Wilson interval differs from typedlm's report: ${low8} – ${high8}`);
  }
  const pct = (x: number) => Math.round(x * 1000) / 10;
  const ranges = [8, 16, 40, 80, 200, 400].map((n) => {
    const [low, high] = wilson(n * 0.875, n);
    return { label: `${n} examples`, low: pct(low), high: pct(high), mid: 87.5 };
  });
  return render(
    {
      schemaVersion: 1,
      type: 'rangebar',
      title: 'An 87.5 % score and its 95 % confidence interval',
      description:
        'The same exact-match score of 87.5 % on datasets of different size. With 8 examples the true score may lie anywhere between about 53 % and 98 %; with 400 examples the interval is about 6 points wide.',
      source: 'Wilson score interval, as computed by typedlm::eval::Report',
      orientation: 'horizontal',
      valueAxis: { title: 'Exact match, %', min: 40, max: 100 },
      showValues: true,
      ranges,
    },
    'interval-by-size',
  );
}
