"use client";

import { RecFeed } from "@/components/rec-feed";
import { useUser } from "@/lib/user";

export default function Home() {
  const { userId, demoUsers } = useUser();
  const demo = demoUsers.find((u) => u.user_id === userId);
  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Recommended for you</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {userId === null
            ? "Loading…"
            : demo
              ? `${demo.label}. Recently: ${demo.recent.join(" · ") || "nothing yet"}`
              : `User ${userId}`}
        </p>
      </div>
      <RecFeed surface="home" title="Your feed" />
    </main>
  );
}
