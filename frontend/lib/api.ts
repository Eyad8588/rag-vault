"use client";

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8710";

export type Role = "reader" | "editor" | "admin";
export type Tag = "public" | "internal" | "restricted";
export type User = { username: string; role: Role; clearances: Tag[] };

export type Doc = {
  id: string;
  filename: string;
  title: string;
  topic: string | null;
  topic_conf: number | null;
  sensitivity: string | null;
  sens_conf: number | null;
  access_tag: Tag;
  tag_source: "classifier" | "manual";
  uploaded_by: string;
  created_at: number;
  n_chunks: number;
  status: "indexing" | "graphing" | "ready" | "error";
  error: string | null;
};

export type Citation = {
  n: number;
  doc_id: string;
  title: string;
  filename: string;
  chunk_index: number;
  access_tag: Tag;
  text: string;
  score: number;
  via: string[];
};

const TOKEN_KEY = "ragvault.token";
const USER_KEY = "ragvault.user";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  } catch {
    return null;
  }
}

const listeners = new Set<() => void>();

export function subscribeSession(cb: () => void) {
  listeners.add(cb);
  window.addEventListener("storage", cb);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", cb);
  };
}

export function getUserRaw(): string | null {
  try {
    return localStorage.getItem(USER_KEY);
  } catch {
    return null;
  }
}

export function saveSession(token: string, user: User) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  listeners.forEach((l) => l());
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  listeners.forEach((l) => l());
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const res = await fetch(`${API}${path}`, { ...init, headers });
  // Expired/invalid token: clearing the session makes <Shell> redirect to /login.
  if (res.status === 401 && !path.startsWith("/auth/login")) clearSession();
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {}
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export const TAG_STYLE: Record<Tag, string> = {
  public: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
  internal: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  restricted: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
};
