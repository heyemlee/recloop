import { expect, type Page, test } from "@playwright/test";

// M1 acceptance (docs/design.md §16): click an item, reload the home page, and the
// recommendations follow. Screenshots go to artifacts/e2e/.
//
// The check is causal, not just "the page is different": the clicked item is one that is not in
// the user's history yet, so before the click no candidate can come from it; after the reload
// the history version has gone up by one, the clicked item is the newest history entry, and
// ItemCF candidates seeded by it are on the page.

const USER = Number(process.env.E2E_USER ?? 1213);
const SHOTS = "../artifacts/e2e";

async function tiles(page: Page) {
  return page.getByTestId("rec-tile").evaluateAll((els) =>
    els.map((e) => ({
      id: Number(e.getAttribute("data-item-id")),
      via: (e.getAttribute("data-via") ?? "").split(",").filter(Boolean).map(Number),
    })),
  );
}

async function history(page: Page) {
  return page
    .locator("[data-testid=feed-home] [data-testid=history] li[data-item-id]")
    .evaluateAll((els) => els.map((e) => Number(e.getAttribute("data-item-id"))));
}

async function seen24h(page: Page): Promise<number> {
  const r = await page.request.get(`/api/v1/users/${USER}/context`);
  expect(r.ok()).toBe(true);
  return (await r.json()).seen_24h;
}

async function hv(page: Page) {
  return Number(await page.locator("[data-testid=feed-home] [data-testid=hv]").textContent());
}

test("clicking an item changes the next home feed", async ({ page }) => {
  await page.goto(`/?user=${USER}`);
  await expect(page.getByTestId("feed-home")).toHaveAttribute("data-request-id", /.+/);
  await expect(page.getByTestId("rec-tile").first()).toBeVisible();
  // Read before any impression can have been sent (1 s visible + up to 2 s until the flush).
  const seenBefore = await seen24h(page);
  const hvBefore = await hv(page);
  const historyBefore = await history(page);
  const before = await tiles(page);
  await page.waitForLoadState("networkidle"); // product images
  await page.screenshot({ path: `${SHOTS}/1-home-before.png` });

  // Impressions: tiles >= 50% visible for 1 s are reported, batched, within the next flush. The
  // feed starts below the hero (and the debug panel), so scroll to it first.
  await page.getByTestId("rec-tile").first().scrollIntoViewIfNeeded();
  await expect.poll(() => seen24h(page), { timeout: 8000 }).toBeGreaterThan(seenBefore);

  const target = before.find((t) => !historyBefore.includes(t.id));
  expect(target, "a recommended item that is not in the history yet").toBeDefined();
  expect(before.some((t) => t.via.includes(target!.id))).toBe(false);

  await page.locator(`[data-testid=rec-tile][data-item-id="${target!.id}"] a`).click();
  await expect(page).toHaveURL(new RegExp(`/items/${target!.id}$`));
  await expect(page.getByTestId("item-title")).toBeVisible();
  await expect(page.getByTestId("feed-item").getByTestId("rec-tile").first()).toBeVisible();
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${SHOTS}/2-item-page.png` });

  // Reload the home page as a new document.
  await page.goto("/");
  await expect(page.getByTestId("feed-home")).toHaveAttribute("data-request-id", /.+/);
  await expect(page.getByTestId("rec-tile").first()).toBeVisible();
  expect(await hv(page)).toBe(hvBefore + 1);
  const historyAfter = await history(page);
  expect(historyAfter[0]).toBe(target!.id);

  const after = await tiles(page);
  const fromClick = after.filter((t) => t.via.includes(target!.id));
  expect(fromClick.length, "ItemCF candidates seeded by the clicked item").toBeGreaterThan(0);
  expect(after.map((t) => t.id)).not.toContain(target!.id); // clicked = seen in the last 24 h
  const filteredAsSeen = Number(
    await page.locator("[data-testid=feed-home] [data-testid=filter-seen]").textContent(),
  );
  expect(filteredAsSeen).toBeGreaterThan(1); // impressions + the click
  const changed = after.filter((t) => !before.some((b) => b.id === t.id)).length;
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${SHOTS}/3-home-after.png` });
  console.log(
    JSON.stringify({
      user: USER,
      clicked: target!.id,
      hv: [hvBefore, hvBefore + 1],
      new_items_in_top20: changed,
      filtered_as_seen: filteredAsSeen,
      items_from_clicked_item: fromClick.map((t) => t.id),
    }),
  );
});
