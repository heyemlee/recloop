"use client";

import Link from "next/link";
import { useState } from "react";

import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useUser } from "@/lib/user";

export function Header() {
  const { userId, setUserId, demoUsers, debug, setDebug } = useUser();
  const [custom, setCustom] = useState("");
  const isDemo = demoUsers.some((u) => u.user_id === userId);

  return (
    <header className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="flex items-baseline gap-2">
          <span className="text-lg font-semibold tracking-tight">RecLoop</span>
          <span className="hidden text-sm text-muted-foreground sm:inline">office supplies</span>
        </Link>
        <Link href="/chat" className="text-sm text-muted-foreground hover:text-foreground">
          Assistant
        </Link>
        <Link href="/advertiser" className="text-sm text-muted-foreground hover:text-foreground">
          Advertiser
        </Link>
        <div className="ml-auto flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-muted-foreground">Browsing as</span>
            <select
              data-testid="user-select"
              className="h-8 max-w-72 rounded-lg border bg-background px-2 text-sm"
              value={isDemo ? String(userId) : ""}
              onChange={(e) => e.target.value && setUserId(Number(e.target.value))}
            >
              {!isDemo && <option value="">{userId === null ? "…" : `User ${userId}`}</option>}
              {demoUsers.map((u) => (
                <option key={u.user_id} value={u.user_id}>
                  {u.label} ({u.history_len} items)
                </option>
              ))}
            </select>
          </label>
          <form
            className="flex items-center gap-1"
            onSubmit={(e) => {
              e.preventDefault();
              if (/^\d+$/.test(custom)) setUserId(Number(custom));
              setCustom("");
            }}
          >
            <Input
              className="h-8 w-28"
              inputMode="numeric"
              placeholder="any user id"
              value={custom}
              onChange={(e) => setCustom(e.target.value)}
              aria-label="Switch to user id"
            />
          </form>
          <label className="flex items-center gap-2">
            <Switch checked={debug} onCheckedChange={(v) => setDebug(Boolean(v))} aria-label="Debug panel" />
            <span className="text-muted-foreground">Debug</span>
          </label>
        </div>
      </div>
    </header>
  );
}
