"use client";

import { useCallback, useEffect, useState } from "react";
import Shell from "@/components/Shell";
import { PageTitle, btnGhost } from "@/components/ui";
import { api } from "@/lib/api";

type Entry = { id: number; ts: number; username: string | null; role: string | null; action: string; target: string | null; details: Record<string, unknown> };

const ACTIONS = ["", "query", "add_doc", "remove_doc", "retag_doc", "login", "login_failed", "create_user", "update_user", "delete_user"];
const COLOR: Record<string, string> = {
  query: "text-sky-300",
  add_doc: "text-emerald-300",
  remove_doc: "text-rose-300",
  retag_doc: "text-amber-300",
  login_failed: "text-rose-400",
};

function summary(e: Entry): string {
  const d = e.details as Record<string, unknown>;
  switch (e.action) {
    case "query":
      return `"${d.question}" → cited ${(d.cited_docs as string[])?.length ?? 0} doc(s), tags [${(d.allowed_tags as string[])?.join(", ")}], ${d.ms} ms`;
    case "add_doc":
      return `${d.title} (${d.filename}) · ${d.topic} / ${d.sensitivity} → ${d.access_tag} (${d.tag_source}), ${d.chunks} chunks`;
    case "remove_doc":
      return `${d.title} (${d.filename}), ${d.chunks} chunks`;
    case "retag_doc":
      return `${d.from} → ${d.to}`;
    default:
      return Object.keys(d).length ? JSON.stringify(d) : "";
  }
}

export default function Audit() {
  const [rows, setRows] = useState<Entry[]>([]);
  const [action, setAction] = useState("");
  const [tick, setTick] = useState(0);
  const load = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    api<Entry[]>(`/admin/audit?limit=500${action ? `&action=${action}` : ""}`).then(setRows).catch(() => setRows([]));
  }, [action, tick]);

  return (
    <Shell roles={["admin"]}>
      <PageTitle title="Audit log" subtitle="Every login, query, document add/remove and user change." />
      <div className="mb-4 flex items-center gap-3">
        <select className="rounded-lg border border-white/15 bg-black/30 px-3 py-2 text-sm" value={action} onChange={(e) => setAction(e.target.value)}>
          {ACTIONS.map((a) => <option key={a} value={a}>{a || "all actions"}</option>)}
        </select>
        <button className={btnGhost} onClick={load}>Refresh</button>
      </div>
      <div className="overflow-x-auto rounded-xl border border-white/10">
        <table className="w-full text-sm">
          <thead className="bg-white/[0.04] text-left text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-3">Time</th>
              <th className="px-4 py-3">User</th>
              <th className="px-4 py-3">Action</th>
              <th className="px-4 py-3">Target</th>
              <th className="px-4 py-3">Details</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr key={e.id} className="border-t border-white/5 align-top">
                <td className="whitespace-nowrap px-4 py-2 text-zinc-400">{new Date(e.ts * 1000).toLocaleString()}</td>
                <td className="px-4 py-2">{e.username} <span className="text-xs text-zinc-500">{e.role}</span></td>
                <td className={`px-4 py-2 font-mono text-xs ${COLOR[e.action] ?? "text-zinc-300"}`}>{e.action}</td>
                <td className="px-4 py-2 font-mono text-xs text-zinc-400">{e.target}</td>
                <td className="px-4 py-2 text-zinc-300">{summary(e)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}
