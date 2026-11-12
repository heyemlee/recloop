"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { Switch } from "@/components/ui/switch";
import { useUser } from "@/lib/user";

import { AskBar } from "./ask-bar";

/** The small wordmark of the inner pages; the home page shows the large one in its hero. */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={className ?? "text-[28px] leading-none font-bold tracking-[-0.075em] text-brand"}>recloop</span>
  );
}

/** Who is browsing (any dataset user id) and the debug switch. */
function Controls() {
  const { userId, setUserId, debug, setDebug } = useUser();
  const [custom, setCustom] = useState("");
  return (
    <div className="ml-auto flex items-center gap-2">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (/^\d+$/.test(custom)) setUserId(Number(custom));
          setCustom("");
        }}
      >
        <input
          className="h-11 w-28 rounded-full border bg-card px-4 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring"
          inputMode="numeric"
          placeholder={userId === null ? "user id" : `user ${userId}`}
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          aria-label="Switch to user id"
        />
      </form>
      <label className="flex h-11 cursor-pointer items-center gap-2.5 rounded-full border bg-card pr-3 pl-4 text-sm">
        Debug
        <Switch checked={debug} onCheckedChange={(v) => setDebug(Boolean(v))} aria-label="Debug panel" />
      </label>
    </div>
  );
}

/**
 * The top row. Home: the controls only (its wordmark and search are in the hero). Item pages:
 * wordmark, search, controls. The assistant: wordmark and controls. The consoles (advertiser,
 * admin) do not depend on who is browsing: wordmark only.
 */
export function TopBar() {
  const pathname = usePathname();
  const home = pathname === "/";
  const store = home || pathname.startsWith("/items") || pathname.startsWith("/chat");
  return (
    <header className="flex min-h-14 flex-wrap items-center gap-4">
      {!home && (
        <Link href="/" aria-label="RecLoop home">
          <Wordmark />
        </Link>
      )}
      {pathname.startsWith("/items") && <AskBar className="mx-auto max-md:order-last" />}
      {store && <Controls />}
    </header>
  );
}
