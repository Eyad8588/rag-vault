import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Vault",
  description: "Permission-aware RAG with a knowledge graph, running on local models",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full bg-zinc-950 text-zinc-100">{children}</body>
    </html>
  );
}
