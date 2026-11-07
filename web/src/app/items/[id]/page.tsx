import { Suspense } from "react";

import { Skeleton } from "@/components/ui/skeleton";

import { ItemView } from "./item-view";

// The item id is only known at request time (useParams in ItemView), so the view streams in
// behind a Suspense boundary after the static shell.
export default function ItemPage() {
  return (
    <Suspense
      fallback={
        <main className="mx-auto w-full max-w-7xl px-4 py-6">
          <Skeleton className="h-80 w-full rounded-xl" />
        </main>
      }
    >
      <ItemView />
    </Suspense>
  );
}
