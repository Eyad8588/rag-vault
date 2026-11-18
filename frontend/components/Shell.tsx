"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useSyncExternalStore } from "react";
import { clearSession, getUserRaw, subscribeSession, type Role, type User } from "@/lib/api";
import { TagBadge } from "./ui";

const NAV: { href: string; label: string; roles: Role[] }[] = [
  { href: "/chat", label: "Chat", roles: ["reader", "editor", "admin"] },
  { href: "/documents", label: "Documents", roles: ["reader", "editor", "admin"] },
  { href: "/graph", label: "Graph", roles: ["reader", "editor", "admin"] },
  { href: "/admin", label: "Admin", roles: ["admin"] },
  { href: "/audit", label: "Audit log", roles: ["admin"] },
];

const noop = () => () => {};

/** The signed-in user from localStorage (null during SSR/hydration). */
export function useUser(): User | null {
  const raw = useSyncExternalStore(subscribeSession, getUserRaw, () => null);
  return useMemo(() => (raw ? (JSON.parse(raw) as User) : null), [raw]);
}

function useHydrated(): boolean {
  return useSyncExternalStore(noop, () => true, () => false);
}

export default function Shell({ children, roles }: { children: React.ReactNode; roles?: Role[] }) {
  const router = useRouter();
  const path = usePathname();
  const user = useUser();
  const hydrated = useHydrated();

  useEffect(() => {
    if (hydrated && !user) router.replace("/login");
  }, [hydrated, user, router]);

  if (!hydrated || !user) return null;
  const allowed = !roles || roles.includes(user.role);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-white/10 bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <Link href="/chat" className="flex items-center gap-2 font-semibold">
            <span className="grid h-7 w-7 place-items-center rounded-md bg-indigo-500 text-sm">RV</span>
            RAG Vault
          </Link>
          <nav className="flex flex-wrap gap-1 text-sm">
            {NAV.filter((n) => n.roles.includes(user.role)).map((n) => (
              <Link
                key={n.href}
                href={n.href}
                className={`rounded-md px-3 py-1.5 ${path.startsWith(n.href) ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white"}`}
              >
                {n.label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="text-zinc-300">
              {user.username} <span className="text-zinc-500">· {user.role}</span>
            </span>
            <span className="hidden gap-1 sm:flex">
              {(user.role === "admin" ? (["public", "internal", "restricted"] as const) : user.clearances).map((t) => (
                <TagBadge key={t} tag={t} />
              ))}
            </span>
            <button
              className="text-zinc-400 hover:text-white"
              onClick={() => {
                clearSession();
                router.replace("/login");
              }}
            >
              Sign out
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
        {allowed ? children : <p className="text-zinc-400">Your role ({user.role}) can&apos;t open this page.</p>}
      </main>
    </div>
  );
}
