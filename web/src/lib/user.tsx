"use client";

// Who is browsing: a dataset user picked in the header (persisted in localStorage, or set with
// ?user=<id> in the URL), plus the debug-panel toggle. Both live in a small external store
// (localStorage with an in-memory fallback) read through useSyncExternalStore.
import { createContext, useCallback, useContext, useEffect, useState, useSyncExternalStore } from "react";

import { getDemoUsers } from "./api";
import type { DemoUser } from "./types";

type UserState = {
  userId: number | null;
  setUserId: (id: number) => void;
  demoUsers: DemoUser[];
  debug: boolean;
  setDebug: (on: boolean) => void;
};

const Ctx = createContext<UserState | null>(null);
const USER_KEY = "recloop.user_id";
const DEBUG_KEY = "recloop.debug";

const memory = new Map<string, string>();
const listeners = new Set<() => void>();

function read(key: string): string | null {
  try {
    const v = window.localStorage.getItem(key);
    if (v !== null) return v;
  } catch {
    // storage disabled: fall back to memory
  }
  return memory.get(key) ?? null;
}

function write(key: string, value: string) {
  memory.set(key, value);
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // private mode: the choice lasts for this page only
  }
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  window.addEventListener("storage", cb);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", cb);
  };
}

function readUser(): number | null {
  const v = read(USER_KEY);
  return v !== null && /^\d+$/.test(v) ? Number(v) : null;
}

export function UserProvider({ children }: { children: React.ReactNode }) {
  const userId = useSyncExternalStore(subscribe, readUser, () => null);
  const debugRaw = useSyncExternalStore(subscribe, () => read(DEBUG_KEY), () => null);
  const debug = debugRaw === null ? true : debugRaw === "1";
  const [demoUsers, setDemoUsers] = useState<DemoUser[]>([]);

  const setUserId = useCallback((id: number) => write(USER_KEY, String(id)), []);
  const setDebug = useCallback((on: boolean) => write(DEBUG_KEY, on ? "1" : "0"), []);

  useEffect(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("user");
    if (fromUrl !== null && /^\d+$/.test(fromUrl)) write(USER_KEY, fromUrl);
    getDemoUsers()
      .then((list) => {
        setDemoUsers(list);
        if (readUser() === null && list.length > 0) write(USER_KEY, String(list[0].user_id));
      })
      .catch(() => {
        if (readUser() === null) write(USER_KEY, "0");
      });
  }, []);

  return (
    <Ctx.Provider value={{ userId, setUserId, demoUsers, debug, setDebug }}>{children}</Ctx.Provider>
  );
}

export function useUser(): UserState {
  const v = useContext(Ctx);
  if (v === null) throw new Error("useUser outside UserProvider");
  return v;
}
