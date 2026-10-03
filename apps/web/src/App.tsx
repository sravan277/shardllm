/** App shell: tab switch, coordinator connection, and the notice strip.
 *
 * All five tab bodies live in ./tabs and all data fetching lives in ./state, so
 * this file stays a routing + connection surface. That split is also what
 * keeps a token update from re-rendering the model list, the device registry
 * and the plan: those components are unmounted while chat is open.
 */

import { useCallback, useEffect, useState } from "react";
import { useConnection } from "./state/useConnection";
import { useChat } from "./state/useChat";
import { useDevices } from "./state/useDevices";
import { useDistribution } from "./state/useDistribution";
import { useUsageTab } from "./state/useUsageTab";
import { ChatTab } from "./tabs/ChatTab";
import { DevicesTab } from "./tabs/DevicesTab";
import { DistributionTab } from "./tabs/DistributionTab";
import { ModelsTab } from "./tabs/ModelsTab";
import { UsageTab } from "./tabs/UsageTab";
import { jget } from "./api/http";
import { parseNode } from "./api/parse";
import type { NodeInfo } from "./api/types";
import { shortId } from "./lib/format";

const TABS = ["chat", "devices", "distribution", "models", "usage"] as const;
type Tab = (typeof TABS)[number];

const TAB_LABEL: Record<Tab, string> = {
  chat: "Chat",
  devices: "Devices",
  distribution: "Distribution",
  models: "Models",
  usage: "Usage",
};

function initialTab(): Tab {
  try {
    const t = new URLSearchParams(window.location.search).get("tab");
    return TABS.includes(t as Tab) ? (t as Tab) : "chat";
  } catch {
    return "chat";
  }
}

export default function App() {
  const conn = useConnection();
  const [tab, setTab] = useState<Tab>(initialTab);
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [chatDraft, setChatDraft] = useState("");

  const reachable = conn.health === "live";
  const chat = useChat(conn.base, conn.generation, reachable);

  // Tab data hooks are always called (React rules) but only fetch while their
  // tab is the visible one, so the shell keeps a stable hook order.
  const devices = useDevices(conn.base, tab === "devices", conn.generation);
  const distribution = useDistribution(conn.base, tab === "distribution", conn.generation);
  const usageTab = useUsageTab(conn.base, tab === "usage", conn.generation);

  useEffect(() => {
    let live = true;
    jget<unknown>(conn.base, "/api/node")
      .then((raw) => {
        if (live) setNode(parseNode(raw));
      })
      .catch(() => {
        if (live) setNode(null);
      });
    return () => {
      live = false;
    };
  }, [conn.base, conn.generation]);

  // Keep ?tab= deep links working without a router.
  useEffect(() => {
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("tab", tab);
      window.history.replaceState(null, "", u.toString());
    } catch {
      /* non-URL context: skip */
    }
  }, [tab]);

  const runModel = useCallback((modelId: string) => {
    setChatDraft(`Run ${modelId}: hello mesh`);
    setTab("chat");
  }, []);

  /** One submit = validate, persist, and drop the old coordinator's chat.
   *  `commit` hands back the new URL because the state update has not landed
   *  yet, so `forgetCoordinator` would otherwise read the previous one. */
  const forgetCoordinator = chat.forgetCoordinator;
  const applyCoordinator = useCallback(() => {
    const next = conn.commit();
    if (next) forgetCoordinator(next);
  }, [conn, forgetCoordinator]);

  const pill = connectionPill(conn.health, chat.streamPhase);
  const identity = identityParts(node, conn.base);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden />
          <div>
            <h1>DLLM Mesh</h1>
            <p>Local LAN inference, with no cloud in the loop</p>
            {identity.length > 0 ? (
              <p className="mono identity">
                {identity.map((part) => (
                  <span key={part}>{part}</span>
                ))}
              </p>
            ) : null}
          </div>
        </div>

        <nav className="nav" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t}
              className={tab === t ? "on" : ""}
              onClick={() => setTab(t)}
              aria-current={tab === t ? "page" : undefined}
            >
              {TAB_LABEL[t]}
            </button>
          ))}
        </nav>

        <form
          className="conn"
          onSubmit={(e) => {
            e.preventDefault();
            applyCoordinator();
          }}
        >
          <input
            value={conn.draft}
            onChange={(e) => conn.setDraft(e.target.value)}
            spellCheck={false}
            aria-label="Coordinator address"
            aria-invalid={conn.error ? true : undefined}
            placeholder="http://127.0.0.1:8080"
          />
          <button type="submit" className="ghost">
            Connect
          </button>
          <button
            type="button"
            className={`pill-button ${pill.tone}`}
            onClick={conn.reconnect}
            title="Reconnect now: re-check the coordinator and resume the stream"
          >
            {pill.text}
          </button>
        </form>
      </header>

      {conn.error ? <p className="notice">{conn.error}</p> : null}
      {chat.notice ? <p className="notice">{chat.notice}</p> : null}

      {tab === "chat" ? <ChatTab chat={chat} draft={chatDraft} onDraft={setChatDraft} /> : null}
      {tab === "devices" ? <DevicesTab devices={devices} base={conn.base} /> : null}
      {tab === "distribution" ? <DistributionTab data={distribution} /> : null}
      {tab === "models" ? <ModelsTab base={conn.base} onRun={runModel} generation={conn.generation} /> : null}
      {tab === "usage" ? <UsageTab data={usageTab} /> : null}

      <footer>Protocol dllm/1, KV block 16, checkpoints every 64 to 128 tokens, the cloud never sees chat.</footer>
    </div>
  );
}

type Pill = { text: string; tone: "live" | "warn" | "down" };

/** One state for the clickable pill. A dead EventSource shows up here as
 *  "reconnecting" rather than hiding inside a notice string. */
function connectionPill(health: "unknown" | "live" | "down", phase: "idle" | "live" | "reconnecting"): Pill {
  if (health === "down") return { text: "no coordinator", tone: "down" };
  if (phase === "reconnecting") return { text: "reconnecting", tone: "warn" };
  if (health === "live") return { text: "coordinator live", tone: "live" };
  return { text: "checking", tone: "warn" };
}

/** Header chips, built only from what the coordinator actually returned.
 *  Rendered as separate elements: no `a · b · c` meta string. */
function identityParts(node: NodeInfo | null, base: string): string[] {
  let where = "";
  try {
    const host = new URL(base).hostname;
    where = host === "127.0.0.1" || host === "localhost" || host === "::1" ? "on this machine" : `on ${host}`;
  } catch {
    /* no usable base yet */
  }
  return [node ? `node ${shortId(node.node_id)}` : "", node?.version ? `version ${node.version}` : "", where].filter(
    (p) => p.length > 0,
  );
}