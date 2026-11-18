"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import Shell from "@/components/Shell";
import { TagBadge, btnGhost, input } from "@/components/ui";
import { api, type Tag } from "@/lib/api";
import { TYPE_COLORS, type GLink, type GNode } from "@/lib/graph";

const ForceGraph = dynamic(() => import("@/components/ForceGraph"), { ssr: false });

type Detail = { id: string; title: string; chunks: { index: number; text: string }[] };

export default function GraphPage() {
  const [data, setData] = useState<{ nodes: GNode[]; links: GLink[] } | null>(null);
  const [sel, setSel] = useState<GNode | null>(null);
  const [filter, setFilter] = useState("");
  const [doc, setDoc] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<{ nodes: GNode[]; links: GLink[] }>("/graph").then(setData).catch((e) => setError(String(e)));
  }, []);

  const shown = useMemo(() => {
    if (!data) return null;
    if (!filter.trim()) return data;
    const f = filter.toLowerCase();
    const keep = new Set(data.nodes.filter((n) => n.name.toLowerCase().includes(f) || n.docs.some((d) => d.title.toLowerCase().includes(f))).map((n) => n.id));
    const id = (x: number | GNode) => (typeof x === "object" ? x.id : x);
    return { nodes: data.nodes.filter((n) => keep.has(n.id)), links: data.links.filter((l) => keep.has(id(l.source)) && keep.has(id(l.target))) };
  }, [data, filter]);

  const relations = useMemo(() => {
    if (!sel || !data) return [];
    const id = (x: number | GNode) => (typeof x === "object" ? x.id : x);
    const name = (x: number | GNode) => data.nodes.find((n) => n.id === id(x))?.name ?? "?";
    return data.links.filter((l) => id(l.source) === sel.id || id(l.target) === sel.id).map((l) => `${name(l.source)} → ${l.relation} → ${name(l.target)}`);
  }, [sel, data]);

  return (
    <Shell>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">Knowledge graph</h1>
        <span className="text-sm text-zinc-500">{data ? `${data.nodes.length} entities · ${data.links.length} relations (only from documents you can see)` : "loading…"}</span>
        <input className={`${input} ml-auto max-w-xs`} placeholder="Filter entities or documents…" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>
      <div className="mb-3 flex flex-wrap gap-3 text-xs text-zinc-400">
        {Object.entries(TYPE_COLORS).map(([t, c]) => (
          <span key={t} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: c }} />
            {t}
          </span>
        ))}
      </div>
      {error && <p className="text-rose-400">{error}</p>}
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <div className="h-[70vh] min-h-[420px] overflow-hidden rounded-xl border border-white/10">
          {shown && shown.nodes.length > 0 ? (
            <ForceGraph data={shown} selected={sel?.id ?? null} onSelect={(n) => { setSel(n); setDoc(null); }} />
          ) : (
            <div className="grid h-full place-items-center text-sm text-zinc-500">{data ? "No entities yet: upload documents first." : ""}</div>
          )}
        </div>
        <aside className="space-y-4">
          {!sel && <p className="rounded-xl border border-white/10 p-4 text-sm text-zinc-400">Click a node to see the documents that mention it.</p>}
          {sel && (
            <div className="rounded-xl border border-white/10 p-4">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full" style={{ background: TYPE_COLORS[sel.type] }} />
                <h2 className="font-semibold">{sel.name}</h2>
                <span className="text-xs text-zinc-500">{sel.type}</span>
              </div>
              <p className="mt-3 text-xs uppercase tracking-wide text-zinc-500">Documents</p>
              <ul className="mt-1 space-y-1">
                {sel.docs.map((d) => (
                  <li key={d.id}>
                    <button className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-white/5" onClick={async () => setDoc(await api<Detail>(`/documents/${d.id}`))}>
                      <span className="truncate">{d.title}</span>
                      <span className="ml-auto"><TagBadge tag={d.access_tag as Tag} /></span>
                    </button>
                  </li>
                ))}
              </ul>
              {relations.length > 0 && (
                <>
                  <p className="mt-3 text-xs uppercase tracking-wide text-zinc-500">Relations</p>
                  <ul className="mt-1 space-y-1 text-sm text-zinc-300">
                    {relations.slice(0, 20).map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </>
              )}
            </div>
          )}
          {doc && (
            <div className="rounded-xl border border-white/10 p-4">
              <div className="mb-2 flex items-center">
                <h3 className="font-semibold">{doc.title}</h3>
                <button className={`${btnGhost} ml-auto`} onClick={() => setDoc(null)}>Close</button>
              </div>
              <div className="max-h-80 space-y-2 overflow-y-auto text-sm text-zinc-300">
                {doc.chunks.map((c) => (
                  <p key={c.index} className={`whitespace-pre-wrap rounded-md p-2 ${sel && c.text.toLowerCase().includes(sel.name.toLowerCase()) ? "bg-indigo-500/10" : ""}`}>{c.text}</p>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>
    </Shell>
  );
}
