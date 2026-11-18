"use client";

import { useCallback, useEffect, useState } from "react";
import Shell, { useUser } from "@/components/Shell";
import { Card, PageTitle, TagBadge, btn, btnDanger, btnGhost } from "@/components/ui";
import { api, type Doc, type Tag } from "@/lib/api";

type Detail = Doc & { chunks: { index: number; text: string }[] };

export default function Documents() {
  const user = useUser();
  const [docs, setDocs] = useState<Doc[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [tag, setTag] = useState<"" | Tag>("");
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [detail, setDetail] = useState<Detail | null>(null);

  const load = useCallback(async () => {
    try {
      setDocs(await api<Doc[]>("/documents"));
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while any document is still being graphed.
  useEffect(() => {
    if (!docs.some((d) => d.status === "indexing" || d.status === "graphing")) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [docs, load]);

  async function uploadAll(e: React.FormEvent) {
    e.preventDefault();
    if (!files) return;
    setBusy(true);
    for (const f of Array.from(files)) {
      const fd = new FormData();
      fd.append("file", f);
      if (tag) fd.append("access_tag", tag);
      try {
        const d = await api<Doc>("/documents", { method: "POST", body: fd });
        setLog((l) => [`✓ ${f.name}: ${d.topic} · ${d.sensitivity} → ${d.access_tag} (${d.n_chunks} chunks)`, ...l]);
      } catch (err) {
        setLog((l) => [`✗ ${f.name}: ${err}`, ...l]);
      }
      await load();
    }
    setBusy(false);
    setFiles(null);
    (e.target as HTMLFormElement).reset();
  }

  async function remove(d: Doc) {
    if (!confirm(`Remove "${d.title}"? Its chunks and graph nodes are deleted too.`)) return;
    await api(`/documents/${d.id}`, { method: "DELETE" });
    if (detail?.id === d.id) setDetail(null);
    load();
  }

  async function retag(d: Doc, t: Tag) {
    await api(`/documents/${d.id}`, { method: "PATCH", body: JSON.stringify({ access_tag: t }) });
    load();
  }

  const canAdd = user?.role === "editor" || user?.role === "admin";
  const isAdmin = user?.role === "admin";

  return (
    <Shell>
      <PageTitle title="Documents" subtitle="Only documents your clearance allows are listed." />
      {canAdd && (
        <Card className="mb-6">
          <form onSubmit={uploadAll} className="flex flex-wrap items-end gap-4">
            <label className="flex-1 text-sm">
              <span className="mb-1 block text-zinc-400">Upload PDF, Markdown or text</span>
              <input
                type="file"
                multiple
                accept=".pdf,.md,.markdown,.txt"
                onChange={(e) => setFiles(e.target.files)}
                className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-white/10 file:px-3 file:py-1.5 file:text-zinc-100"
              />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-zinc-400">Access tag</span>
              <select value={tag} onChange={(e) => setTag(e.target.value as Tag | "")} className="rounded-lg border border-white/15 bg-black/30 px-3 py-2">
                <option value="">Auto (from classifier)</option>
                <option value="public">public</option>
                <option value="internal">internal</option>
                <option value="restricted">restricted</option>
              </select>
            </label>
            <button className={btn} disabled={busy || !files?.length}>
              {busy ? "Indexing…" : "Upload"}
            </button>
          </form>
          {log.length > 0 && (
            <ul className="mt-4 space-y-1 font-mono text-xs text-zinc-400">
              {log.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          )}
        </Card>
      )}
      {error && <p className="mb-4 text-rose-400">{error}</p>}
      <div className="overflow-x-auto rounded-xl border border-white/10">
        <table className="w-full text-sm">
          <thead className="bg-white/[0.04] text-left text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-3">Title</th>
              <th className="px-4 py-3">Topic</th>
              <th className="px-4 py-3">Sensitivity</th>
              <th className="px-4 py-3">Access</th>
              <th className="px-4 py-3">Chunks</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id} className="border-t border-white/5 hover:bg-white/[0.02]">
                <td className="px-4 py-3">
                  <button className="text-left font-medium hover:text-indigo-300" onClick={async () => setDetail(await api<Detail>(`/documents/${d.id}`))}>
                    {d.title}
                  </button>
                  <div className="text-xs text-zinc-500">
                    {d.filename} · by {d.uploaded_by}
                  </div>
                </td>
                <td className="px-4 py-3 text-zinc-300">
                  {d.topic} <span className="text-xs text-zinc-500">{d.topic_conf ? `${Math.round(d.topic_conf * 100)}%` : ""}</span>
                </td>
                <td className="px-4 py-3 text-zinc-300">
                  {d.sensitivity} <span className="text-xs text-zinc-500">{d.sens_conf ? `${Math.round(d.sens_conf * 100)}%` : ""}</span>
                </td>
                <td className="px-4 py-3">
                  {isAdmin ? (
                    <select
                      value={d.access_tag}
                      onChange={(e) => retag(d, e.target.value as Tag)}
                      className="rounded-md border border-white/15 bg-black/30 px-2 py-1 text-xs"
                    >
                      <option>public</option>
                      <option>internal</option>
                      <option>restricted</option>
                    </select>
                  ) : (
                    <TagBadge tag={d.access_tag} />
                  )}
                  <div className="mt-1 text-xs text-zinc-500">{d.tag_source}</div>
                </td>
                <td className="px-4 py-3 text-zinc-300">{d.n_chunks}</td>
                <td className="px-4 py-3">
                  <span className={d.status === "ready" ? "text-emerald-400" : d.status === "error" ? "text-rose-400" : "animate-pulse text-amber-300"}>
                    {d.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-right">
                  {isAdmin && (
                    <button className={btnDanger} onClick={() => remove(d)}>
                      Remove
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {docs.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-10 text-center text-zinc-500">
                  No documents visible to you yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {detail && (
        <Card className="mt-6">
          <div className="mb-3 flex items-center gap-3">
            <h2 className="text-lg font-semibold">{detail.title}</h2>
            <TagBadge tag={detail.access_tag} />
            <button className={`${btnGhost} ml-auto`} onClick={() => setDetail(null)}>
              Close
            </button>
          </div>
          <div className="max-h-[480px] space-y-3 overflow-y-auto">
            {detail.chunks.map((c) => (
              <div key={c.index} className="rounded-lg border border-white/10 p-3 text-sm text-zinc-300">
                <div className="mb-1 text-xs text-zinc-500">chunk {c.index}</div>
                <p className="whitespace-pre-wrap">{c.text}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </Shell>
  );
}
