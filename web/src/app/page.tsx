"use client";

import { AskBar } from "@/components/ask-bar";
import { Hero } from "@/components/hero";
import { RecFeed } from "@/components/rec-feed";
import { UserPills } from "@/components/user-pills";

export default function Home() {
  return (
    <main className="flex flex-col">
      <RecFeed
        surface="home"
        title="Recommended for you"
        // Above the feed: its first items floating over the wordmark, the search pill, and who
        // is browsing.
        header={({ items, onOpen }) => (
          <div className="mb-16 flex flex-col">
            <Hero items={items} onOpen={onOpen} />
            <div className="flex flex-col items-center gap-6">
              <h1 aria-label="RecLoop" className="text-[84px] leading-[0.9] font-bold tracking-[-0.075em] text-brand max-md:text-6xl">
                recloop
              </h1>
              <AskBar size="lg" />
              <UserPills />
            </div>
          </div>
        )}
      />
    </main>
  );
}
