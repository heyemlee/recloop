"use client";

// Which product photos can float on the page as cut-outs (the home hero). The catalog's photos
// are whatever the listings had: many are on white, some have a backdrop or a room around them,
// some are a thin pen across an empty frame. A photo is judged once, in the browser, from a
// 32 x 32 copy of it (the image host allows cross-origin reads); the verdicts are kept for the
// life of the page.
import { useEffect, useSyncExternalStore } from "react";

export type Cutout = {
  /**
   * 1: on white, one product that fills its frame (a clean cut-out).
   * 2: on white, but thin, small, pale, or several things with text around them.
   * 3: not on white, or the photo could not be read.
   */
  tier: 1 | 2 | 3;
  /** How close a tier-2 photo is to a clean one: 1 = at every threshold. */
  score: number;
  /** Enlargement (1 to 1.3) that brings a small product up to the size of the others. */
  zoom: number;
};

const UNKNOWN: Cutout = { tier: 3, score: 0, zoom: 1 };

const N = 32;
const WHITE = 238; // a pixel is background when its darkest channel is at least this
const MIN_WHITE_BORDER = 0.85; // share of the photo's outermost pixels that are background
const MIN_INK = 45; // mean darkness (0..255) of the product's pixels: below it the photo is washed out
const MIN_COVER = 0.22; // share of the square slot the product's pixels cover
const MIN_SHAPE = 0.45; // short side / long side of the product's bounding box
const MIN_ONE_PIECE = 0.9; // share of the product's pixels in their largest connected piece
const FULL_COVER = 0.45; // a product covering less is enlarged towards this
const MAX_ZOOM = 1.3;
const GIVE_UP_MS = 3000;

const verdicts = new Map<string, Cutout>();
const started = new Set<string>();
const listeners = new Set<() => void>();
let version = 0;

function settle(url: string, verdict: Cutout) {
  if (verdicts.has(url)) return;
  verdicts.set(url, verdict);
  version++;
  listeners.forEach((l) => l());
}

/** The size of the largest 4-connected piece of the marked cells of an N x N grid. */
function largestPiece(marked: Uint8Array): number {
  const seen = new Uint8Array(N * N);
  let largest = 0;
  for (let start = 0; start < N * N; start++) {
    if (!marked[start] || seen[start]) continue;
    let size = 0;
    const stack = [start];
    seen[start] = 1;
    while (stack.length > 0) {
      const at = stack.pop()!;
      size++;
      const x = at % N;
      const next = [x > 0 ? at - 1 : -1, x < N - 1 ? at + 1 : -1, at - N, at + N];
      for (const to of next) {
        if (to >= 0 && to < N * N && marked[to] && !seen[to]) {
          seen[to] = 1;
          stack.push(to);
        }
      }
    }
    largest = Math.max(largest, size);
  }
  return largest;
}

function measure(img: HTMLImageElement): Cutout {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (w === 0 || h === 0) return UNKNOWN;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = N;
  const g = canvas.getContext("2d", { willReadFrequently: true });
  if (!g) return UNKNOWN;
  g.fillStyle = "#fff";
  g.fillRect(0, 0, N, N);
  g.drawImage(img, 0, 0, N, N); // stretched: one sample grid whatever the photo's shape
  const px = g.getImageData(0, 0, N, N).data;

  const marked = new Uint8Array(N * N);
  let product = 0;
  let ink = 0;
  let whiteBorder = 0;
  let x0 = N;
  let x1 = -1;
  let y0 = N;
  let y1 = -1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const o = (y * N + x) * 4;
      const white = Math.min(px[o], px[o + 1], px[o + 2]) >= WHITE;
      if (!white) {
        marked[y * N + x] = 1;
        product++;
        ink += 255 - (0.299 * px[o] + 0.587 * px[o + 1] + 0.114 * px[o + 2]);
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
      if (white && (x === 0 || y === 0 || x === N - 1 || y === N - 1)) whiteBorder++;
    }
  }
  if (whiteBorder / (4 * N - 4) < MIN_WHITE_BORDER || product === 0) return UNKNOWN;

  // Back to the photo's own proportions, shown `object-contain` in a square slot.
  const side = Math.max(w, h);
  const boxW = ((x1 - x0 + 1) / N) * w;
  const boxH = ((y1 - y0 + 1) / N) * h;
  const cover = (product / (N * N)) * ((w * h) / (side * side));
  const shape = Math.min(boxW, boxH) / Math.max(boxW, boxH);
  const score =
    Math.min(1, ink / product / MIN_INK) *
    Math.min(1, cover / MIN_COVER) *
    Math.min(1, shape / MIN_SHAPE) *
    Math.min(1, largestPiece(marked) / product / MIN_ONE_PIECE);
  // Enlarged about the slot's centre: never past the point where the product's box reaches the slot's edge.
  const reach = Math.max((Math.max(x1 + 1, N - x0) / N) * 2 - 1, 0.01) * (w / side);
  const reachY = Math.max((Math.max(y1 + 1, N - y0) / N) * 2 - 1, 0.01) * (h / side);
  const zoom = Math.max(1, Math.min(MAX_ZOOM, Math.sqrt(FULL_COVER / cover), 1 / Math.max(reach, reachY)));
  return { tier: score >= 1 ? 1 : 2, score, zoom };
}

function judge(url: string) {
  if (started.has(url)) return;
  started.add(url);
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    let verdict = UNKNOWN;
    try {
      verdict = measure(img);
    } catch {
      // a canvas the browser will not let us read: treat the photo as unknown
    }
    settle(url, verdict);
  };
  img.onerror = () => settle(url, UNKNOWN);
  img.src = url;
  setTimeout(() => settle(url, UNKNOWN), GIVE_UP_MS);
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

/** The verdict on a photo that has been judged; a photo that has not is unknown. */
export const cutoutOf = (url: string | undefined): Cutout => verdicts.get(url ?? "") ?? UNKNOWN;

/** False while any of the photos is still being judged (at most 3 s); then read `cutoutOf`. */
export function useCutoutsJudged(urls: string[]): boolean {
  useSyncExternalStore(
    subscribe,
    () => version,
    () => 0,
  );
  useEffect(() => {
    urls.forEach(judge);
  }, [urls]);
  return urls.every((u) => verdicts.has(u));
}
