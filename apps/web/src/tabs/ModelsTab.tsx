/** Models tab: the catalog the coordinator is serving. */

import { useEffect, useState } from "react";
import { fetchModels } from "../api/http";
import type { Model } from "../api/types";
import { EmptyState } from "../components/EmptyState";

export function ModelsTab({
  base,
  onRun,
  generation,
}: {
  base: string;
  onRun: (modelId: string) => void;
  generation: number;
}) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const kick = window.setTimeout(() => {
      setLoading(true);
      fetchModels(base)
        .then((list) => {
          if (!live) return;
          setModels(list);
          setError(null);
        })
        .catch(() => {
          if (!live) return;
          setModels([]);
          setError("Could not read the catalog. Check that dllm serve is running at the address in the top bar.");
        })
        .finally(() => {
          if (live) setLoading(false);
        });
    }, 0);
    return () => {
      live = false;
      window.clearTimeout(kick);
    };
  }, [base, generation]);

  if (loading && models.length === 0) {
    return (
      <main className="cards">
        <p className="hint">Reading the model catalog…</p>
      </main>
    );
  }

  return (
    <main className="cards">
      {models.length === 0 ? (
        <EmptyState
          title={error ? "No catalog to show" : "No models installed"}
          body={error ?? "Pull a model with dllm pull on the coordinator, then refresh this tab."}
        />
      ) : (
        models.map((m) => (
          <article className="card" key={m.id}>
            <h2>{m.id}</h2>
            <p className="hint meta-chips">
              {[m.quant, m.params, m.size_mb ? `${m.size_mb} MB` : ""].filter(Boolean).length > 0 ? (
                [m.quant, m.params, m.size_mb ? `${m.size_mb} MB` : ""]
                  .filter(Boolean)
                  .map((bit) => <span key={bit}>{bit}</span>)
              ) : (
                <span>curated GGUF</span>
              )}
            </p>
            {m.stages && m.stages.length > 0 ? (
              <div className="shards">
                {m.stages.map((s) => (
                  <span key={s.range}>{s.range}</span>
                ))}
              </div>
            ) : (
              <p className="hint">No shard ranges reported for this model.</p>
            )}
            <button onClick={() => onRun(m.id)}>Run in chat</button>
          </article>
        ))
      )}
    </main>
  );
}