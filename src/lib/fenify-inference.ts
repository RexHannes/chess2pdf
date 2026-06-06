/**
 * Fenify ONNX inference — browser-side chess board image → FEN
 *
 * Architecture (notnil/fenify):
 *   Input:  [1, 3, 300, 300]  float32  (grayscale→3-channel, ImageNet-normalised)
 *   Output: [1, 64, 13]       float32  (64 squares × 13 piece classes)
 *
 * The Fenify model is optional. If it is not present or cannot load quickly,
 * chess2pdf must continue with local grid/template/occupancy recognition rather
 * than blocking Scan page indefinitely.
 */

import type { BBox } from "@/lib/types";

const MODEL_SIZE = 300;
const MODEL_HEAD_TIMEOUT_MS = 4_000;
const MODEL_CREATE_TIMEOUT_MS = 18_000;
const INFERENCE_TIMEOUT_MS = 8_000;
const LOCAL_MODEL_URL = "/fenify/model.onnx";
const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

const PIECE_CHARS = [
  "",
  "P",
  "N",
  "B",
  "R",
  "Q",
  "K",
  "p",
  "n",
  "b",
  "r",
  "q",
  "k",
] as const;

type OrtSession = import("onnxruntime-web").InferenceSession;

let _session: OrtSession | null = null;
let _loadPromise: Promise<OrtSession | null> | null = null;
let _modelUnavailable = false;
let _statusMessage = "Fenify model not started.";

export function isFenifyReady(): boolean {
  return _session !== null;
}

export function fenifyStatus(): "idle" | "loading" | "ready" | "unavailable" {
  if (_session !== null) return "ready";
  if (_modelUnavailable) return "unavailable";
  if (_loadPromise !== null) return "loading";
  return "idle";
}

export function fenifyStatusMessage(): string {
  return _statusMessage;
}

export async function loadFenifyModel(): Promise<OrtSession | null> {
  if (_session !== null) return _session;
  if (_modelUnavailable) return null;
  if (_loadPromise !== null) return _loadPromise;

  _statusMessage = "Checking optional Fenify board model…";
  _loadPromise = loadFenifyModelOnce();
  return _loadPromise;
}

async function loadFenifyModelOnce(): Promise<OrtSession | null> {
  try {
    const ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = process.env.NEXT_PUBLIC_ORT_WASM_PATH ?? "/ort/";

    const modelUrl = process.env.NEXT_PUBLIC_FENIFY_MODEL_URL ?? LOCAL_MODEL_URL;
    const probe = await fetchWithTimeout(modelUrl, { method: "HEAD" }, MODEL_HEAD_TIMEOUT_MS);
    if (!probe.ok) {
      markFenifyUnavailable(`Optional Fenify model not found at ${modelUrl}; using fallback scanner.`);
      return null;
    }

    _statusMessage = "Loading optional Fenify board model…";
    _session = await withTimeout(
      ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
      MODEL_CREATE_TIMEOUT_MS,
      "Fenify model load timed out",
    );

    _statusMessage = "Fenify board model ready.";
    return _session;
  } catch (err) {
    markFenifyUnavailable(
      err instanceof Error
        ? `Fenify unavailable (${err.message}); using fallback scanner.`
        : "Fenify unavailable; using fallback scanner.",
    );
    return null;
  } finally {
    _loadPromise = null;
  }
}

export async function classifyWithFenify(
  canvas: HTMLCanvasElement,
  bbox: BBox,
): Promise<{ fen: string; confidence: number } | null> {
  const session = await loadFenifyModel();
  if (!session) return null;

  try {
    const ort = await import("onnxruntime-web");
    const inputData = preprocessBoard(canvas, bbox);
    const inputTensor = new ort.Tensor("float32", inputData, [1, 3, MODEL_SIZE, MODEL_SIZE]);
    const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;

    const results = await withTimeout(session.run(feeds), INFERENCE_TIMEOUT_MS, "Fenify inference timed out");
    const outputTensor = results[session.outputNames[0]];
    const rawData = outputTensor.data as Float32Array;
    const numClasses = 13;
    const squarePieces: number[] = new Array(64);

    for (let sq = 0; sq < 64; sq++) {
      let bestClass = 0;
      let bestScore = rawData[sq * numClasses];
      for (let cls = 1; cls < numClasses; cls++) {
        const score = rawData[sq * numClasses + cls];
        if (score > bestScore) {
          bestScore = score;
          bestClass = cls;
        }
      }
      squarePieces[sq] = bestClass;
    }

    const confidence = estimateConfidence(rawData, squarePieces, numClasses);
    const fen = buildFen(squarePieces);

    return { fen, confidence };
  } catch (err) {
    console.warn("[fenify] Inference error — falling back to heuristic:", err);
    return null;
  }
}

function preprocessBoard(canvas: HTMLCanvasElement, bbox: BBox): Float32Array {
  const resized = document.createElement("canvas");
  resized.width = MODEL_SIZE;
  resized.height = MODEL_SIZE;
  const ctx = resized.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas 2D unavailable");
  ctx.drawImage(canvas, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const pixels = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const numPixels = MODEL_SIZE * MODEL_SIZE;
  const tensor = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const base = i * 4;
    const gray = (0.299 * pixels[base] + 0.587 * pixels[base + 1] + 0.114 * pixels[base + 2]) / 255;
    tensor[i] = (gray - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    tensor[numPixels + i] = (gray - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    tensor[2 * numPixels + i] = (gray - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return tensor;
}

function buildFen(squarePieces: number[]): string {
  const rows: string[] = [];

  for (let rank = 7; rank >= 0; rank--) {
    let row = "";
    let empty = 0;
    for (let file = 0; file < 8; file++) {
      const pieceIdx = squarePieces[rank * 8 + file];
      const piece = PIECE_CHARS[pieceIdx] ?? "";
      if (!piece) {
        empty++;
      } else {
        if (empty > 0) {
          row += empty;
          empty = 0;
        }
        row += piece;
      }
    }
    if (empty > 0) row += empty;
    rows.push(row);
  }

  return `${rows.join("/")} w - - 0 1`;
}

function estimateConfidence(rawData: Float32Array, squarePieces: number[], numClasses: number): number {
  let totalMargin = 0;
  const numSquares = 64;

  for (let sq = 0; sq < numSquares; sq++) {
    const base = sq * numClasses;
    const topClass = squarePieces[sq];
    const topScore = rawData[base + topClass];

    let secondBest = -Infinity;
    for (let cls = 0; cls < numClasses; cls++) {
      if (cls !== topClass && rawData[base + cls] > secondBest) {
        secondBest = rawData[base + cls];
      }
    }
    totalMargin += Math.max(0, topScore - secondBest);
  }

  const avgMargin = totalMargin / numSquares;
  return Math.max(0.5, Math.min(0.95, 0.5 + avgMargin * 0.08));
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function markFenifyUnavailable(message: string) {
  console.warn("[fenify]", message);
  _modelUnavailable = true;
  _loadPromise = null;
  _statusMessage = message;
}
