"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, saveSession, type User } from "@/lib/api";
import { btn, input } from "@/components/ui";

const DEMO = [
  { name: "reader", desc: "asks questions; sees public + internal" },
  { name: "editor", desc: "adds documents; sees everything" },
  { name: "admin", desc: "removes docs, manages users, audit log" },
];

export default function Login() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function login(u: string, p: string) {
    setBusy(true);
    setError(null);
    try {
      const res = await api<{ token: string; user: User }>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username: u, password: p }),
      });
      saveSession(res.token, res.user);
      router.replace("/chat");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-screen place-items-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-xl bg-indigo-500 text-lg font-bold">RV</div>
          <h1 className="text-2xl font-semibold">RAG Vault</h1>
          <p className="mt-1 text-sm text-zinc-400">Permission-aware answers from your documents, on local models.</p>
        </div>
        <form
          className="space-y-3 rounded-xl border border-white/10 bg-white/[0.03] p-6"
          onSubmit={(e) => {
            e.preventDefault();
            login(username, password);
          }}
        >
          <input className={input} placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
          <input className={input} type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error && <p className="text-sm text-rose-400">{error}</p>}
          <button className={`${btn} w-full`} disabled={busy || !username || !password}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <div className="mt-6">
          <p className="mb-2 text-xs uppercase tracking-wide text-zinc-500">Demo accounts (password = username)</p>
          <div className="grid gap-2">
            {DEMO.map((d) => (
              <button
                key={d.name}
                onClick={() => login(d.name, d.name)}
                disabled={busy}
                className="flex items-center justify-between rounded-lg border border-white/10 px-4 py-2.5 text-left text-sm hover:bg-white/5"
              >
                <span className="font-medium">{d.name}</span>
                <span className="text-zinc-400">{d.desc}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
