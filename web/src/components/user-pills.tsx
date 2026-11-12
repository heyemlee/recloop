"use client";

import { useUser } from "@/lib/user";
import { cn } from "@/lib/utils";

import { CategoryIcon, UserIcon } from "./icons";

const PILL =
  "flex h-10 shrink-0 items-center gap-2 rounded-full border py-0 pr-4 pl-1 text-[15px] tracking-[-0.031em] transition-colors";

/**
 * A category's name as a pill: its first term, without the generic last word ("Writing &
 * Correction Supplies" -> "Writing", "Desk Accessories & Workspace Organizers" -> "Desk",
 * "Printer Ink & Toner" -> "Printer Ink"). The full name is in the pill's tooltip.
 */
function shortName(category: string): string {
  const first = category.split(/,| & /)[0].trim();
  return first.replace(/ (Products|Supplies|Accessories)$/, "") || category;
}

/**
 * Whose feed this is: one pill per demo user, named after the category the user bought most.
 * A user chosen by id (the top bar) who is not one of them gets a pill of their own.
 */
export function UserPills() {
  const { userId, setUserId, demoUsers } = useUser();
  const isDemo = demoUsers.some((u) => u.user_id === userId);
  return (
    <div
      data-testid="user-pills"
      className="flex max-w-full items-center gap-2 max-md:overflow-x-auto max-md:pb-2 md:flex-wrap md:justify-center"
    >
      <span className="mr-1 shrink-0 text-sm text-muted-foreground">Browsing as</span>
      {!isDemo && userId !== null && (
        <span className={cn(PILL, "border-foreground bg-foreground text-background")}>
          <span className="flex size-8 items-center justify-center rounded-full bg-white/20">
            <UserIcon />
          </span>
          User {userId}
        </span>
      )}
      {demoUsers.map((u) => {
        const active = u.user_id === userId;
        return (
          <button
            key={u.user_id}
            type="button"
            aria-pressed={active}
            title={`${u.category ?? "New user"} · user ${u.user_id} · ${u.history_len} items in history`}
            onClick={() => setUserId(u.user_id)}
            className={cn(
              PILL,
              active ? "border-foreground bg-foreground text-background" : "bg-card shadow-pill hover:bg-muted",
            )}
          >
            <span className={cn("flex size-8 items-center justify-center rounded-full", active ? "bg-white/20" : "bg-muted")}>
              <CategoryIcon category={u.category} />
            </span>
            <span className="max-w-40 truncate">{u.category ? shortName(u.category) : "New user"}</span>
          </button>
        );
      })}
    </div>
  );
}
