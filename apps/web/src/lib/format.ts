/** Small pure formatters shared by every tab. */

import type { LayerRun } from "./layers";

/** `9f2c…c41a` — enough to tell two rows apart, short enough for a card. */
export function shortId(id: string): string {
  if (!id) return "unknown";
  return id.length > 20 ? `${id.slice(0, 9)}…${id.slice(-4)}` : id;
}

/** Parse a coordinator timestamp. SQLite UTC text has no zone marker, so a
 *  bare `YYYY-MM-DDTHH:MM:SS[.fff]` is read as UTC rather than local. */
function parseTimestamp(value: string): number | null {
  let text = value.trim();
  if (!text) return null;
  if (!/(Z|[+-]\d{2}:?\d{2})$/.test(text)) text += "Z";
  const t = Date.parse(text);
  return Number.isFinite(t) ? t : null;
}

/** "just now", "45s ago", "3 min ago", "2 h ago", or null when unparseable. */
export function relativeTime(value: string | null, now: number = Date.now()): string | null {
  if (!value) return null;
  const t = parseTimestamp(value);
  if (t === null) return null;
  const secs = Math.round((now - t) / 1000);
  if (secs < 0) return "just now";
  if (secs < 10) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.floor(hours / 24);
  return days < 30 ? `${days} d ago` : null;
}

/** "4 h 12 min" style uptime, or null when unknown. */
export function formatDuration(seconds: number | undefined): string | null {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return null;
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h} h ${m} min`;
  if (m > 0) return `${m} min ${s} s`;
  return `${s} s`;
}

/** `0–27` style run label with its layer count. */
export function describeRun(name: string, run: LayerRun): string {
  return `${name}: layers ${run.start}–${run.end}, ${run.count} layer${run.count === 1 ? "" : "s"}`;
}