/** Pure-SVG donut (no chart library): tokens_out share per device. */

import { memo } from "react";
import { usageColorFor } from "../lib/palette";

export type DonutSlice = { key: string; label: string; tokens: number };

type Arc = { slice: DonutSlice; len: number; offset: number };

/** Arc geometry computed off-render, so nothing is mutated while rendering. */
function computeArcs(slices: DonutSlice[], circumference: number, total: number): Arc[] {
  const out: Arc[] = [];
  let acc = 0;
  for (const slice of slices) {
    const len = total > 0 ? (slice.tokens / total) * circumference : 0;
    out.push({ slice, len, offset: acc });
    acc += len;
  }
  return out;
}

export const UsageDonut = memo(function UsageDonut({
  slices,
  order,
  total,
}: {
  slices: DonutSlice[];
  order: string[];
  total: number;
}) {
  const R = 70;
  const C = 2 * Math.PI * R;
  const arcs = computeArcs(slices, C, total);
  return (
    <svg
      className="donut"
      width="180"
      height="180"
      viewBox="0 0 180 180"
      role="img"
      aria-label={`tokens out share, ${total} total`}
    >
      <circle cx="90" cy="90" r={R} fill="none" stroke="#1e2530" strokeWidth="28" />
      {arcs.map(({ slice, len, offset }) => {
        if (len <= 0) return null;
        const gap = slices.length > 1 ? 2 : 0;
        const drawn = Math.max(0, len - gap);
        return (
          <circle
            key={slice.key}
            cx="90"
            cy="90"
            r={R}
            fill="none"
            stroke={usageColorFor(slice.key, order)}
            strokeWidth="28"
            strokeDasharray={`${drawn} ${C - drawn}`}
            strokeDashoffset={-offset}
            transform="rotate(-90 90 90)"
          >
            <title>{`${slice.label}: ${slice.tokens} tokens`}</title>
          </circle>
        );
      })}
      <text x="90" y="86" textAnchor="middle" className="donut-total">
        {total}
      </text>
      <text x="90" y="104" textAnchor="middle" className="donut-sub">
        tokens out
      </text>
    </svg>
  );
});