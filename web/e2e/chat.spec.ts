import { expect, test } from "@playwright/test";

// M6 (docs/design.md §10, §22.1): the assistant answers with item cards whose prices are listing
// prices within the asked budget; a card on screen is an impression, and opening one is a click
// the store takes in: the item becomes the user's newest history entry.

// Not the feed test's user: each test checks the newest history entry of its own user.
const USER = Number(process.env.E2E_CHAT_USER ?? 1222);
const SHOTS = "../artifacts/e2e";

test("the assistant answers with cards, and a click reaches the user's history", async ({ page }) => {
  const context = async () => (await (await page.request.get(`/api/v1/users/${USER}/context`)).json());
  await page.goto(`/chat?user=${USER}`);
  // The reply's impressions: cards >= 50% visible for 1 s, sent with the next flush (every 2 s).
  let replyId = "";
  const impressions = page.waitForRequest(
    (r) =>
      replyId !== "" &&
      r.url().endsWith("/api/v1/events") &&
      (r.postData() ?? "").includes(`"impression_id":"${replyId}:`),
    { timeout: 20_000 },
  );

  await page.getByTestId("chat-input").fill("Ink cartridges for my Canon PIXMA MG3620 under $25");
  await page.getByTestId("chat-input").press("Enter");
  const turn = page.getByTestId("chat-turn").last();
  await expect(turn).toHaveAttribute("data-reply-id", /.+/, { timeout: 20_000 });
  replyId = (await turn.getAttribute("data-reply-id")) ?? "";
  const cards = turn.getByTestId("chat-card");
  await expect(cards.first()).toBeVisible();

  // Prices come from the listings and meet the budget; a sponsored card says so.
  for (const price of await turn.getByTestId("chat-price").allTextContents()) {
    expect(price).toMatch(/^\$\d+\.\d\d$/);
    expect(Number(price.slice(1))).toBeLessThanOrEqual(25);
  }
  for (const label of await turn.getByTestId("sponsored-label").allTextContents()) {
    expect(label).toBe("Sponsored");
  }
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${SHOTS}/4-chat.png`, fullPage: true });

  await impressions;

  const first = cards.and(page.locator("[data-slot=organic]")).first();
  const itemId = Number(await first.getAttribute("data-item-id"));
  await first.locator("a").click();
  await expect(page).toHaveURL(new RegExp(`/items/${itemId}$`));
  await expect
    .poll(async () => (await context()).history[0]?.item_id, { timeout: 8000 })
    .toBe(itemId);
});
