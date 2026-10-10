import Diagram from './components/Diagram.astro';
import IntervalBySize from './components/IntervalBySize.astro';

// Components for docs/ MDX, which cannot import anything itself.
export const components = { Diagram, IntervalBySize };
