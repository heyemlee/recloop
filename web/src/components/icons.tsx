// The app's icons, drawn inline: filled glyphs for the navigation rail, line icons elsewhere.
type IconProps = { className?: string };

function Filled({ className, d, evenodd }: IconProps & { d: string; evenodd?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" fillRule={evenodd ? "evenodd" : undefined} aria-hidden="true" className={className ?? "size-6"}>
      <path d={d} />
    </svg>
  );
}

function Line({ className, d, width = 2 }: IconProps & { d: string; width?: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={width}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className ?? "size-4"}
    >
      <path d={d} />
    </svg>
  );
}

export const HomeIcon = (p: IconProps) => <Filled {...p} d="M12 3l9 7.5V21a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V10.5z" />;
export const ChatIcon = (p: IconProps) => (
  <Filled {...p} d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-5 4v-4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z" />
);
export const TagIcon = (p: IconProps) => (
  <Filled
    {...p}
    evenodd
    d="M3 3h8.6l9.1 9.1a2 2 0 0 1 0 2.8l-5.8 5.8a2 2 0 0 1-2.8 0L3 11.6zM7.5 6a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"
  />
);
export const GridIcon = (p: IconProps) => <Filled {...p} d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" />;

export const ArrowRightIcon = (p: IconProps) => <Line {...p} width={2.2} d="M5 12h14M12 5l7 7-7 7" />;
export const ArrowUpIcon = (p: IconProps) => <Line {...p} width={2.2} d="M12 19V5M5 12l7-7 7 7" />;
export const PlusIcon = (p: IconProps) => <Line {...p} d="M12 5v14M5 12h14" />;
export const CheckCircleIcon = (p: IconProps) => <Line {...p} width={2.2} d="M22 11.08V12a10 10 0 1 1-5.93-9.14M22 4L12 14.01l-3-3" />;
export const HelpCircleIcon = (p: IconProps) => (
  <Line {...p} width={2.2} d="M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0zM9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3M12 17h.01" />
);
export const UserIcon = (p: IconProps) => <Line {...p} d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0z" />;

/** The mark: an open loop that ends in a dot. */
export function LoopMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className ?? "size-7"}>
      <path d="M19.4 14.2A8 8 0 1 1 15.6 4.9" stroke="currentColor" strokeWidth="3.6" strokeLinecap="round" />
      <circle cx="19.2" cy="8.6" r="2.4" fill="currentColor" />
    </svg>
  );
}

// The demo users are named after the category they bought most (ml/data/seed_redis.py); a
// category without an icon of its own gets the tag.
const CATEGORY_PATH: Record<string, string> = {
  "Writing & Correction Supplies": "M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z",
  Paper: "M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM13 2v7h7",
  "Printer Ink & Toner":
    "M6 9V2h12v7M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2M6 14h12v8H6z",
  "Desk Accessories & Workspace Organizers":
    "M22 12h-6l-2 3h-4l-2-3H2M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",
  "Tape, Adhesives & Fasteners":
    "M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48",
  "Envelopes, Mailers & Shipping Supplies":
    "M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2zM22 6l-10 7L2 6",
  "Filing Products": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
};
const TAG_PATH = "M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82zM7 7h.01";

export function CategoryIcon({ category, className }: IconProps & { category: string | null }) {
  if (category === null) return <UserIcon className={className} />;
  return <Line className={className} d={CATEGORY_PATH[category] ?? TAG_PATH} />;
}
