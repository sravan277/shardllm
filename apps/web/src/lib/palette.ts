/** Fixed palette so a device keeps one colour across every chart. No chart
 *  library: these values are read straight into SVG `stroke`/`background`. */

export const USAGE_PALETTE = ["#2dd4bf", "#f5b544", "#60a5fa", "#f472b6", "#a78bfa", "#34d399", "#fb7185", "#facc15"];

export const UNATTRIBUTED_COLOR = "#5b6b7c";

/** Colour for `key`, assigned by its position in `order` so the mapping is
 *  stable for a given set of devices (no reshuffle when a value changes). */
export function usageColorFor(key: string, order: string[]): string {
  const i = order.indexOf(key);
  return USAGE_PALETTE[(i < 0 ? 0 : i) % USAGE_PALETTE.length];
}

/** Stable display order: donut slices first, then any plan-only device. */
export function colorOrder(...groups: string[][]): string[] {
  return [...new Set(groups.flat().filter((k) => k.length > 0))];
}