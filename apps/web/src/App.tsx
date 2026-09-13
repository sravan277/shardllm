import { useCallback, useEffect, useRef, useState } from "react";

type Msg = { role: "you" | "mesh"; text: string };
type Stage = { name: string; layers: string; ms: number; hot: boolean };
type Model = { id: string; quant?: string; params?: string; size_mb?: number; stages?: { range: string }[] };
type Peer = { host: string; port: number; name?: string };

const FALLBACK_STAGES: Stage[] = [
  { name: "Stage A", layers: "layers 0–8", ms: 0, hot: false },
  { name: "Stage B", layers: "layers 9–18", ms: 0, hot: false },
  { name: "Stage C", layers: "layers 19–27", ms: 0, hot: false },
];

async function jget<T>(base: string, path: string): Promise<T> {
  const r = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return (await r.json()) as T;
}

export default function App() {
  const [base, setBase] = useState("http://127.0.0.1:8080");
  const [health, setHealth] = useState<"unknown" | "live" | "down">("unknown");
  const [tab, setTab] = useState<"chat" | "models" | "devices">("chat");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [session, setSession] = useState<string | null>(null);
  const [lastEvent, setLastEvent] = useState<string>("");
  const [models, setModels] = useState<Model[]>([]);
  const [stages, setStages] = useState<Stage[]>(FALLBACK_STAGES);
  const [peers, setPeers] = useState<Peer[]>([]);
  const [notice, setNotice] = useState("");
  const esRef = useRef<EventSource | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  const ping = useCallback(async () => {
    try {
      await jget(base, "/api/health");
      setHealth("live");
    } catch {
      setHealth("down");
    }
  }, [base]);

  useEffect(() => {
    ping();
    const t = setInterval(ping, 5000);
    return () => clearInterval(t);
  }, [ping]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [msgs]);

  const attach = useCallback(
    (id: string, resume: string) => {
      esRef.current?.close();
      const url = `${base}/v1/sessions/${id}/events${resume ? `?last_event=${encodeURIComponent(resume)}` : ""}`;
      const es = new EventSource(url);
      es.addEventListener("token", (e) => {
        const ev = e as MessageEvent;
        if (ev.lastEventId) setLastEvent(ev.lastEventId);
        try {
          const d = JSON.parse(ev.data) as { text?: string; pos?: number };
          if (d.text) {
            setMsgs((m) => {
              const n = [...m];
              const last = n[n.length - 1];
              if (last?.role === "mesh" && !busy) n.push({ role: "mesh", text: d.text! });
              else if (last?.role === "mesh") last.text += d.text!;
              else n.push({ role: "mesh", text: d.text! });
              return n;
            });
          }
        } catch {
          /* plain-text chunk */
          setMsgs((m) => [...m, { role: "mesh", text: ev.data }]);
        }
      });
      es.addEventListener("commit", () => setBusy(false));
      es.addEventListener("status", (e) => {
        const ev = e as MessageEvent;
        setNotice(ev.data);
      });
      es.onerror = () => setNotice("Stream interrupted — will resume from last event on next send.");
      esRef.current = es;
    },
    [base, busy],
  );

  useEffect(() => () => esRef.current?.close(), []);

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMsgs((m) => [...m, { role: "you", text }]);
    setBusy(true);
    setNotice("");
    try {
      let id = session;
      if (!id) {
        const s = await (await fetch(`${base}/v1/sessions`, { method: "POST" })).json();
        id = (s.id ?? s.session_id) as string;
        setSession(id);
      }
      attach(id!, lastEvent);
      // Light the pipeline strip while generating.
      setStages((st) => st.map((s, i) => ({ ...s, ms: 8 + i * 5, hot: true })));
      await fetch(`${base}/v1/sessions/${id}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      setStages((st) => st.map((s) => ({ ...s, hot: false })));
    } catch (err) {
      setBusy(false);
      setNotice(`Send failed: ${err}. Is dllm serve running at ${base}?`);
      setStages((st) => st.map((s) => ({ ...s, hot: false })));
    }
  };

  const loadModels = useCallback(async () => {
    try {
      const d = await jget<{ models: Model[] }>(base, "/api/models");
      setModels(d.models ?? []);
    } catch {
      setNotice("Model catalog unreachable — start dllm serve first.");
    }
  }, [base]);

  const loadPeers = useCallback(async () => {
    try {
      const d = await jget<{ devices: Peer[] }>(base, "/v1/devices");
      setPeers(d.devices ?? []);
    } catch {
      setPeers([]);
    }
  }, [base]);

  useEffect(() => {
    if (tab === "models") loadModels();
    if (tab === "devices") loadPeers();
  }, [tab, loadModels, loadPeers]);

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="mark" aria-hidden />
          <div>
            <h1>DLLM Mesh</h1>
            <p>Local LAN inference · no cloud in the loop</p>
          </div>
        </div>
        <div className="conn">
          <input value={base} onChange={(e) => setBase(e.target.value)} spellCheck={false} aria-label="Coordinator URL" />
          <span className={`pill ${health}`}>{health === "live" ? "coordinator live" : health === "down" ? "no coordinator" : "checking…"}</span>
        </div>
      </header>

      <nav className="tabs">
        {(["chat", "models", "devices"] as const).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      {notice && <div className="notice">{notice}</div>}

      {tab === "chat" && (
        <main className="chatgrid">
          <section className="thread">
            <div className="log" ref={logRef}>
              {msgs.length === 0 && (
                <div className="empty">
                  <h2>Talk to your own mesh</h2>
                  <p>Start dllm serve, then send a message. Tokens stream here over local SSE — nothing leaves the LAN.</p>
                  {session && <p className="mono">session {session} · resume from {lastEvent || "start"}</p>}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`bubble ${m.role}`}>
                  {m.text}
                </div>
              ))}
              {busy && <div className="typing">meshing…</div>}
            </div>
            <div className="composer">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                placeholder="Message the mesh…"
                aria-label="Message"
              />
              <button onClick={send} disabled={busy || !input.trim()}>
                Send
              </button>
            </div>
          </section>
          <aside className="pipe">
            <h2>Pipeline</h2>
            <div className="stages">
              {stages.map((s, i) => (
                <div key={s.name}>
                  <div className={`stage ${s.hot ? "hot" : ""}`}>
                    <strong>{s.name}</strong>
                    <span>{s.layers}</span>
                    <span className="ms">{s.ms ? `${s.ms} ms` : "idle"}</span>
                  </div>
                  {i < stages.length - 1 && <div className={`link ${s.hot ? "hot" : ""}`} aria-hidden />}
                </div>
              ))}
            </div>
            <p className="hint">Activations hop stage to stage. Only token IDs come back.</p>
          </aside>
        </main>
      )}

      {tab === "models" && (
        <main className="cards">
          {models.length === 0 && <p className="hint">No catalog yet — start the coordinator, or pull a model with dllm pull.</p>}
          {models.map((m) => (
            <div key={m.id} className="card">
              <h2>{m.id}</h2>
              <p className="hint">
                {[m.quant, m.params, m.size_mb ? `${m.size_mb} MB` : ""].filter(Boolean).join(" · ") || "curated GGUF"}
              </p>
              <div className="shards">{(m.stages ?? []).map((s) => <span key={s.range}>{s.range}</span>)}</div>
              <button
                onClick={async () => {
                  setTab("chat");
                  setInput(`Run ${m.id}: hello mesh`);
                }}
              >
                Run in chat
              </button>
            </div>
          ))}
        </main>
      )}

      {tab === "devices" && (
        <main className="cards">
          <div className="card">
            <h2>Paired devices</h2>
            {peers.length === 0 ? (
              <p className="hint">Nothing paired yet. Pair from Android (QR / code / nearby) or approve below once discovery lands.</p>
            ) : (
              peers.map((p) => (
                <p key={`${p.host}:${p.port}`} className="mono">
                  {p.name ?? "device"} · {p.host}:{p.port}
                </p>
              ))
            )}
          </div>
          <div className="card">
            <h2>Pair this browser</h2>
            <p className="hint">Browsers attach by session — paste a session id to resume its stream from your last event.</p>
            <div className="row">
              <input placeholder="session id" id="sessbox" />
              <button
                onClick={() => {
                  const el = document.getElementById("sessbox") as HTMLInputElement | null;
                  if (el?.value.trim()) {
                    setSession(el.value.trim());
                    attach(el.value.trim(), lastEvent);
                    setTab("chat");
                  }
                }}
              >
                Attach
              </button>
            </div>
          </div>
        </main>
      )}

      <footer>proto dllm/1 · KV block 16 · checkpoints every 64–128 tokens · cloud never sees chat</footer>
    </div>
  );
}
