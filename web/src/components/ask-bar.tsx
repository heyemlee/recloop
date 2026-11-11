"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { cn } from "@/lib/utils";

import { ArrowRightIcon } from "./icons";

/**
 * The search pill of the store pages. There is no keyword search: what is typed goes to the
 * shopping assistant (/chat?q=...), which answers it as its first message.
 */
export function AskBar({ size = "md", className }: { size?: "md" | "lg"; className?: string }) {
  const router = useRouter();
  const [text, setText] = useState("");
  // Built without an assistant (the Kubernetes image): nothing to ask.
  if (process.env.NEXT_PUBLIC_CHAT_BACKEND === "off") return null;
  return (
    <form
      role="search"
      className={cn(
        "flex w-full items-center gap-3 rounded-full border border-black/10 bg-card shadow-float",
        size === "lg" ? "h-16 max-w-[600px] py-2 pr-2 pl-7" : "h-14 max-w-[520px] py-1 pr-1 pl-6",
        className,
      )}
      onSubmit={(e) => {
        e.preventDefault();
        const q = text.trim();
        if (q) router.push(`/chat?q=${encodeURIComponent(q)}`);
      }}
    >
      <input
        data-testid="ask-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        maxLength={2000}
        placeholder="What are you shopping for today?"
        aria-label="Ask the shopping assistant"
        className="min-w-0 flex-1 bg-transparent text-base tracking-[-0.031em] outline-none placeholder:text-muted-foreground"
      />
      <button
        type="submit"
        aria-label="Ask the assistant"
        className="flex size-12 shrink-0 items-center justify-center rounded-full bg-brand text-white shadow-brand transition-colors hover:bg-brand/90"
      >
        <ArrowRightIcon className="size-5" />
      </button>
    </form>
  );
}
