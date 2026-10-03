/** Labeled resource bar. Only ever rendered for a real sample: callers pass a
 *  number they already know is non-null and 0–100. */

import { memo } from "react";

export const UsageBar = memo(function UsageBar({ label, pct, tone }: { label: string; pct: number; tone?: "load" }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="meter">
      <div className="mono">
        {label} {clamped.toFixed(0)}%
      </div>
      <div className="bar" aria-label={`${label} ${clamped.toFixed(0)} percent`}>
        <i className={tone === "load" ? "load" : undefined} style={{ width: `${clamped}%` }} />
      </div>
    </div>
  );
});