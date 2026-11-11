import { Suspense } from "react";

import { Skeleton } from "@/components/ui/skeleton";

import { ItemView } from "./item-view";

// The item id is only known at request time (useParams in ItemView), so the view streams in
// behind a Suspense boundary after the static shell.
export default function ItemPage() {
  return (
    <Suspense
      fallback={
        <main className="mt-9">
          <Skeleton className="h-96 w-full rounded-[28px]" />
        </main>
      }
    >
      <ItemView />
    </Suspense>
  );
}
