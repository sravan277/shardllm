/** Thin LAN HTTP helpers: base-URL handling plus one honest error type.
 *
 * Errors keep the server's own `error` string when the coordinator sent one
 * (e.g. DELETE /v1/devices/{self} -> 400 "cannot delete self coordinator"),
 * so a screen can show the cause instead of a bare status code.
 */

import type { Model } from "./types";

/** localStorage key for the one-tap coordinator URL. */
export const BASE_KEY = "dllm:base";

export const LOOPBACK_BASE = "http://127.0.0.1:8080";

/** Default coordinator: the host that served this page, else loopback.
 *
 * `npm run dev` serves the page from Vite (:5173) while `dllm serve` is a
 * different origin, so dev mode must fall back to loopback instead.
 */
export function defaultBase(): string {
  try {
    if (import.meta.env.DEV) return LOOPBACK_BASE;
    if (window.location.protocol.startsWith("http") && window.location.port !== "5173") {
      return window.location.origin;
    }
  } catch {
    /* non-browser context: fall through to loopback */
  }
  return LOOPBACK_BASE;
}

/** Trim and drop a trailing slash so `base + path` never doubles up. */
export function normalizeBase(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

/** Validate a coordinator URL. Returns the normalized value or a cause + fix. */
export function validateBase(raw: string): { base: string } | { error: string } {
  const trimmed = raw.trim();
  if (!trimmed) {
    return { error: "Enter the coordinator address — it looks like http://192.168.1.10:8080." };
  }
  if (!/^https?:\/\//i.test(trimmed)) {
    return { error: "Start the address with http:// or https:// — a bare IP or host name is not enough." };
  }
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return { error: "That address cannot be read — check it for typos, then paste it again." };
  }
  if (!url.hostname) {
    return { error: "That address has no host — include the machine name or IP, for example http://192.168.1.10:8080." };
  }
  if (url.pathname !== "/" && url.pathname !== "") {
    return { error: "Use the address only, without a path — the API paths are added for you." };
  }
  if (url.search || url.hash) {
    return { error: "Drop the ?query and #fragment — the address alone is enough." };
  }
  return { base: normalizeBase(url.origin) };
}

export function readStoredBase(): string {
  try {
    const raw = window.localStorage.getItem(BASE_KEY);
    if (raw) {
      const v = validateBase(raw);
      if ("base" in v) return v.base;
    }
  } catch {
    /* storage blocked: fall back to the default */
  }
  return defaultBase();
}

export function storeBase(base: string): void {
  try {
    window.localStorage.setItem(BASE_KEY, base);
  } catch {
    /* storage blocked: the connection still works for this tab */
  }
}

/** An HTTP failure with the coordinator's own message when it sent one. */
export class ApiError extends Error {
  readonly status: number;
  readonly path: string;

  constructor(status: number, path: string, serverMessage: string | null) {
    super(serverMessage ?? `HTTP ${status} on ${path}`);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
  }
}

/** Read an error body without ever throwing (JSON, text, or empty). */
async function readErrorMessage(res: Response): Promise<string | null> {
  try {
    const text = await res.text();
    if (!text) return null;
    try {
      const parsed = JSON.parse(text) as unknown;
      if (typeof parsed === "object" && parsed !== null) {
        const err = (parsed as Record<string, unknown>)["error"];
        if (typeof err === "string" && err.trim()) return err.trim();
      }
    } catch {
      /* not JSON: fall through to the raw text */
    }
    return text.slice(0, 200);
  } catch {
    return null;
  }
}

/** `GET base + path`, JSON in / out. Throws {@link ApiError} on any non-2xx. */
export async function jget<T>(base: string, path: string): Promise<T> {
  const res = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!res.ok) throw new ApiError(res.status, path, await readErrorMessage(res));
  return (await res.json()) as T;
}

/** `GET` that keeps the status code instead of throwing on 403/404. */
export async function jgetStatus(base: string, path: string): Promise<{ status: number; json: unknown }> {
  const res = await fetch(`${base}${path}`, { cache: "no-store" });
  let json: unknown = null;
  try {
    json = await res.json();
  } catch {
    /* empty or non-JSON body: callers treat it as "nothing usable" */
  }
  return { status: res.status, json };
}

/** `POST` / `DELETE` with an optional JSON body. Throws {@link ApiError}. */
export async function jwrite(base: string, path: string, method: "POST" | "DELETE"): Promise<unknown> {
  const res = await fetch(`${base}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new ApiError(res.status, path, await readErrorMessage(res));
  try {
    return (await res.json()) as unknown;
  } catch {
    return null;
  }
}

/** `GET /api/models` narrowed to the fields the cards render. */
export async function fetchModels(base: string): Promise<Model[]> {
  const raw = await jget<unknown>(base, "/api/models");
  if (typeof raw !== "object" || raw === null) return [];
  const list = (raw as Record<string, unknown>)["models"];
  if (!Array.isArray(list)) return [];
  return list.filter((m): m is Model => typeof m === "object" && m !== null && typeof (m as Model).id === "string");
}