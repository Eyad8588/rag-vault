"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import Shell, { useUser } from "@/components/Shell";
import { Card, PageTitle, btn, btnDanger, input } from "@/components/ui";
import { api, type Role, type Tag, type User } from "@/lib/api";

const TAGS: Tag[] = ["public", "internal", "restricted"];
const ROLES: Role[] = ["reader", "editor", "admin"];

export default function Admin() {
  const me = useUser();
  const [users, setUsers] = useState<User[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ username: "", password: "", role: "reader" as Role });

  const load = useCallback(async () => {
    try {
      setUsers(await api<User[]>("/admin/users"));
    } catch (e) {
      setError(String(e));
    }
  }, []);
  useEffect(() => {
    api<User[]>("/admin/users").then(setUsers).catch((e) => setError(String(e)));
  }, []);

  async function patch(u: User, body: Partial<User>) {
    setError(null);
    try {
      await api(`/admin/users/${u.username}`, { method: "PATCH", body: JSON.stringify(body) });
    } catch (e) {
      setError(String(e));
    }
    load();
  }

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api("/admin/users", { method: "POST", body: JSON.stringify(form) });
      setForm({ username: "", password: "", role: "reader" });
      load();
    } catch (err) {
      setError(String(err));
    }
  }

  async function remove(u: User) {
    if (!confirm(`Delete user ${u.username}?`)) return;
    await api(`/admin/users/${u.username}`, { method: "DELETE" });
    load();
  }

  return (
    <Shell roles={["admin"]}>
      <PageTitle title="Admin" subtitle="Manage users, roles and clearances. Remove documents from the Documents page." />
      <div className="mb-4 flex gap-3 text-sm">
        <Link href="/documents" className="text-indigo-300 hover:underline">→ Manage documents</Link>
        <Link href="/audit" className="text-indigo-300 hover:underline">→ Audit log</Link>
      </div>
      {error && <p className="mb-4 text-sm text-rose-400">{error}</p>}
      <Card className="mb-6">
        <h2 className="mb-3 font-semibold">Add user</h2>
        <form onSubmit={create} className="flex flex-wrap gap-3">
          <input className={`${input} max-w-[200px]`} placeholder="username" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
          <input className={`${input} max-w-[200px]`} type="password" placeholder="password (4+ chars)" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          <select className="rounded-lg border border-white/15 bg-black/30 px-3 text-sm" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
            {ROLES.map((r) => <option key={r}>{r}</option>)}
          </select>
          <button className={btn} disabled={!form.username || form.password.length < 4}>Create</button>
        </form>
      </Card>
      <div className="overflow-x-auto rounded-xl border border-white/10">
        <table className="w-full text-sm">
          <thead className="bg-white/[0.04] text-left text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-3">User</th>
              <th className="px-4 py-3">Role</th>
              <th className="px-4 py-3">Clearances (tags they can read)</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.username} className="border-t border-white/5">
                <td className="px-4 py-3 font-medium">{u.username}</td>
                <td className="px-4 py-3">
                  <select className="rounded-md border border-white/15 bg-black/30 px-2 py-1" value={u.role} disabled={u.username === me?.username} onChange={(e) => patch(u, { role: e.target.value as Role })}>
                    {ROLES.map((r) => <option key={r}>{r}</option>)}
                  </select>
                </td>
                <td className="px-4 py-3">
                  <div className="flex gap-3">
                    {TAGS.map((t) => (
                      <label key={t} className="flex items-center gap-1.5">
                        <input
                          type="checkbox"
                          checked={u.role === "admin" || u.clearances.includes(t)}
                          disabled={u.role === "admin"}
                          onChange={(e) => patch(u, { clearances: e.target.checked ? [...u.clearances, t] : u.clearances.filter((x) => x !== t) })}
                        />
                        {t}
                      </label>
                    ))}
                    {u.role === "admin" && <span className="text-xs text-zinc-500">(admins see all)</span>}
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  {u.username !== me?.username && <button className={btnDanger} onClick={() => remove(u)}>Delete</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}
