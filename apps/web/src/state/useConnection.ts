/** One coordinator connection for the whole app: the URL, its health, and
 *  the single click that re-establishes everything hanging off it.
 *
 *  - The URL is persisted, so a reload lands back on the same coordinator.
 *  - A health ping runs every 5 s; `generation` is bumped on every manual
 *    reconnect and every accepted URL change, which is how tabs re-fetch and
 *    the chat re-attaches its EventSource without a reload.
 */

import { useCallback, useEffect, useState } from "react";
import { readStoredBase, storeBase, validateBase, jget } from "../api/http";
import type { Health } from "../api/types";

export type Connection = {
  /** Committed, normalized coordinator URL every request uses. */
  base: string;
  /** Editable copy shown in the URL box. */
  draft: string;
  setDraft: (v: string) => void;
  health: Health;
  /** cause + fix, or null while the URL is fine. */
  error: string | null;
/** Apply the draft: validate, persist, re-ping, bump `generation`.
   *  Returns the new coordinator URL when it actually changed, else null. */
  commit: () => string | null;
  /** Single-click reconnect: re-ping + re-attach, keeping the stored cursors. */
  reconnect: () => void;
  /** Bumped on reconnect / URL change; consumers re-fetch and re-attach. */
  generation: number;
};

export function useConnection(): Connection {
  const [base, setBase] = useState<string>(readStoredBase);
  const [draft, setDraftState] = useState<string>(base);
  const [health, setHealth] = useState<Health>("unknown");
  const [error, setError] = useState<string | null>(null);
  const [generation, setGeneration] = useState(0);

  const ping = useCallback(async () => {
    try {
      await jget(base, "/api/health");
      setHealth("live");
    } catch {
      setHealth("down");
    }
  }, [base]);

  useEffect(() => {
    // Kick the first ping from a timer so the page paints "checking" first
    // and the coordinator switch never blocks the commit that triggered it.
    const kick = window.setTimeout(() => void ping(), 0);
    const timer = window.setInterval(() => void ping(), 5000);
    return () => {
      window.clearTimeout(kick);
      window.clearInterval(timer);
    };
  }, [ping, generation]);

  const setDraft = useCallback((v: string) => {
    setDraftState(v);
    setError(null);
  }, []);

  const commit = useCallback((): string | null => {
    const result = validateBase(draft);
    if ("error" in result) {
      setError(result.error);
      return null;
    }
    setError(null);
    setDraftState(result.base);
    storeBase(result.base);
    setHealth("unknown");
    if (result.base === base) return null;
    setBase(result.base);
    setGeneration((g) => g + 1);
    // The caller needs the new URL now: `setBase` has not re-rendered yet, so
    // anything it calls still closes over the previous coordinator.
    return result.base;
  }, [base, draft]);

  const reconnect = useCallback(() => {
    setHealth("unknown");
    setError(null);
    void ping();
    setGeneration((g) => g + 1);
  }, [ping]);

  return { base, draft, setDraft, health, error, commit, reconnect, generation };
}