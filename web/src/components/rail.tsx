"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

import { ChatIcon, GridIcon, HomeIcon, LoopMark, TagIcon } from "./icons";

const LINKS = [
  { href: "/", label: "Home", icon: HomeIcon, match: (p: string) => p === "/" || p.startsWith("/items") },
  { href: "/chat", label: "Shopping assistant", icon: ChatIcon, match: (p: string) => p.startsWith("/chat") },
  { href: "/advertiser", label: "Advertiser console", icon: TagIcon, match: (p: string) => p.startsWith("/advertiser") },
  // The admin page is internal: its link shows only where the web app runs with RECLOOP_ADMIN=1.
  ...(process.env.NEXT_PUBLIC_ADMIN === "on"
    ? [{ href: "/admin", label: "Admin", icon: GridIcon, match: (p: string) => p.startsWith("/admin") }]
    : []),
];

/** The navigation rail: the mark on top, icon links in the middle of the window's height. */
export function Rail() {
  const pathname = usePathname();
  return (
    <div className="flex shrink-0 items-center justify-between md:sticky md:top-5 md:h-[calc(100dvh-2.5rem)] md:w-12 md:flex-col md:justify-start">
      <Link href="/" aria-label="RecLoop home" className="flex size-11 items-center justify-center text-brand">
        <LoopMark />
      </Link>
      <nav aria-label="Main" className="flex gap-4 md:my-auto md:flex-col md:gap-5 md:pb-11">
        {LINKS.map(({ href, label, icon: Icon, match }) => {
          const active = match(pathname);
          return (
            <Link
              key={href}
              href={href}
              aria-label={label}
              title={label}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex size-12 items-center justify-center rounded-full transition-colors hover:text-foreground",
                active ? "text-foreground" : "text-[#787574]",
              )}
            >
              <Icon />
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
