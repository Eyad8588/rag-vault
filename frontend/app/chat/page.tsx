"use client";

import { useRef, useState } from "react";
import Shell from "@/components/Shell";
import { TagBadge, btn, input } from "@/components/ui";
import { api, type Citation } from "@/lib/api";

type Turn = { q: string; answer?: string; citations?: Citation[]; error?: string; ms?: number };

function AnswerText({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="whitespace-pre-wrap leading-relaxed">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        return m ? (
          <button
            key={i}
            onClick={() => onCite(Number(m[1]))}
            className="mx-0.5 rounded bg-indigo-500/20 px-1.5 py-0.5 align-baseline text-xs font-semibold text-indigo-300 hover:bg-indigo-500/40"
          >
            {m[1]}
          </button>
        ) : (
          <span key={i}>{p}</span>
        );
      })}
    </p>
  );
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    const question = q.trim();
    if (!question) return;
    setQ("");
    setBusy(true);
    const idx = turns.length;
    setTurns((t) => [...t, { q: question }]);
    const t0 = performance.now();
    try {
      const res = await api<{ answer: string; citations: Citation[] }>("/query", {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      setTurns((t) => t.map((x, i) => (i === idx ? { ...x, ...res, ms: performance.now() - t0 } : x)));
    } catch (err) {
      setTurns((t) => t.map((x, i) => (i === idx ? { ...x, error: String(err) } : x)));
    } finally {
      setBusy(false);
      setTimeout(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }

  return (
    <Shell>
      <div className="mx-auto flex max-w-3xl flex-col">
        {turns.length === 0 && (
          <div className="mb-8 rounded-xl border border-white/10 bg-white/[0.03] p-6 text-sm text-zinc-400">
            <p className="mb-2 text-base text-zinc-200">Ask about the documents you&apos;re allowed to see.</p>
            Retrieval is filtered by your clearance <em>before</em> search runs, so restricted documents can&apos;t leak
            into answers. Every answer cites its sources; click a number to see the passage.
          </div>
        )}
        <div className="space-y-8">
          {turns.map((t, i) => (
            <div key={i} className="space-y-3">
              <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-sm bg-indigo-500/90 px-4 py-2.5 text-white">{t.q}</div>
              <div className="rounded-2xl rounded-bl-sm border border-white/10 bg-white/[0.03] px-5 py-4">
                {t.error ? (
                  <p className="text-rose-400">{t.error}</p>
                ) : t.answer === undefined ? (
                  <p className="animate-pulse text-zinc-400">Searching and thinking…</p>
                ) : (
                  <>
                    <AnswerText text={t.answer} onCite={(n) => setOpen(`${i}:${n}`)} />
                    {t.citations && t.citations.length > 0 && (
                      <div className="mt-4 space-y-2 border-t border-white/10 pt-3">
                        <p className="text-xs uppercase tracking-wide text-zinc-500">
                          Sources {t.ms ? `· ${(t.ms / 1000).toFixed(1)}s` : ""}
                        </p>
                        {t.citations.map((c) => {
                          const key = `${i}:${c.n}`;
                          return (
                            <div key={key} className="rounded-lg border border-white/10">
                              <button
                                className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-white/5"
                                onClick={() => setOpen(open === key ? null : key)}
                              >
                                <span className="rounded bg-indigo-500/20 px-1.5 text-xs font-semibold text-indigo-300">{c.n}</span>
                                <span className="truncate">{c.title}</span>
                                <span className="text-xs text-zinc-500">chunk {c.chunk_index}</span>
                                <span className="ml-auto flex items-center gap-2">
                                  <span className="text-xs text-zinc-500">{c.via.join(" + ")}</span>
                                  <TagBadge tag={c.access_tag} />
                                </span>
                              </button>
                              {open === key && (
                                <p className="whitespace-pre-wrap border-t border-white/10 px-3 py-2 text-sm text-zinc-300">{c.text}</p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
        <div ref={bottom} />
        <form onSubmit={ask} className="sticky bottom-4 mt-8 flex gap-2 rounded-xl border border-white/10 bg-zinc-900/95 p-2 backdrop-blur">
          <input className={`${input} border-0 bg-transparent`} placeholder="Ask a question…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
          <button className={btn} disabled={busy || !q.trim()}>
            Ask
          </button>
        </form>
      </div>
    </Shell>
  );
}
