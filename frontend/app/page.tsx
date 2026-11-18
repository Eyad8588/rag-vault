"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { getUser } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace(getUser() ? "/chat" : "/login");
  }, [router]);
  return null;
}
