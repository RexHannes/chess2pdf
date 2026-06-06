import { MAX_PDF_PAGES } from "@/lib/constants";
import { classifyWithFenify, loadFenifyModel } from "@/lib/fenify-inference";
import { parseRecognizedLine } from "@/lib/move-parser";
import type { BBox, DetectedDiagram, PdfPagePreview, RecognizedLine } from "@/lib/types";

type PdfJs = typeof import("pdfjs-dist");
export type PdfDocument = Awaited<ReturnType<PdfJs["getDocument"]>["promise"]>;
type BoardClassification = {
  fen: string;
  bbox?: BBox;
  confidence: number;
  notes: string[];
  source: "fenify" | "template" | "occupancy" | "fallback";
};

const TEMPLATE_SIZE = 48;
const TEMPLATE_PIECES = [
  { fen: "K", glyph: "♔" },
  { fen: "Q", glyph: "♕" },
  { fen: "R", glyph: "♖" },
  { fen: "B", glyph: "♗" },
  { fen: "N", glyph: "♘" },
  { fen: "P", glyph: "♙" },
  { fen: "k", glyph: "♚" },
  { fen: "q", glyph: "♛" },
  { fen: "r", glyph: "♜" },
  { fen: "b", glyph: "♝" },
  { fen: "n", glyph: "♞" },
  { fen: "p", glyph: "♟" },
] as const;

let pieceTemplates: Array<{ fen: string; ink: Float32Array; norm: number }> | undefined;

export type RenderedPage = PdfPagePreview & {
  canvas: HTMLCanvasElement;
};

export type PageScanResult = {
  page: RenderedPage;
  diagrams: DetectedDiagram[];
  lines: RecognizedLine[];
};

export type BoardScanResult = {
  diagram: DetectedDiagram;
  line: RecognizedLine;
};

export async function loadPdfDocument(file: File): Promise<PdfDocument> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = "/pdfjs/pdf.worker.min.mjs";
  const data = await file.arrayBuffer();
  const task = pdfjs.getDocument({ data });
  const document = await task.promise;

  if (document.numPages > MAX_PDF_PAGES) {
    await document.destroy();
    throw new Error(`This PDF has ${document.numPages} pages. The free browser workflow is capped at ${MAX_PDF_PAGES}.`);
  }

  return document;
}

export async function renderPdfPage(document: PdfDocument, pageIndex: number, scale = 2.35): Promise<RenderedPage> {
  const page = await document.getPage(pageIndex + 1);
  const viewport = page.getViewport({ scale });
  const canvas = window.document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas is unavailable in this browser.");
  }

  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  await page.render({ canvas, canvasContext: context, viewport }).promise;

  const imageUrl = canvas.toDataURL("image/jpeg", 0.86);
  const thumbnailUrl = makeThumbnail(canvas, 180);

  return {
    pageIndex,
    width: canvas.width,
    height: canvas.height,
    thumbnailUrl,
    imageUrl,
    canvas,
  };
}

export function detectBoardCandidates(canvas: HTMLCanvasElement): Array<BBox & { confidence: number }> {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context || canvas.width < 160 || canvas.height < 160) {
    return [];
  }

  const minSize = Math.max(96, Math.floor(Math.min(canvas.width, canvas.height) * 0.11));
  const maxSize = Math.floor(Math.min(canvas.width, canvas.height) * 0.72);
  const candidates: Array<BBox & { confidence: number }> = [];

  for (let size = maxSize; size >= minSize; size -= Math.max(12, Math.floor(size / 10))) {
    const step = Math.max(12, Math.floor(size / 6));
    for (let y = 0; y <= canvas.height - size; y += step) {
      for (let x = 0; x <= canvas.width - size; x += step) {
        const confidence = scoreBoardCandidate(context, x, y, size);
        if (confidence > 0.24) {
          candidates.push({ x, y, width: size, height: size, confidence });
        }
      }
    }
  }

  const refined = nonOverlapping(candidates.sort((a, b) => b.confidence - a.confidence))
    .slice(0, 12)
    .map((candidate) => refineBoardCandidate(context, candidate))
    .sort((a, b) => b.confidence - a.confidence);

  return nonOverlapping(refined).slice(0, 8);
}

/**
 * Kick off a background model-load as soon as this module is imported so that
 * by the time the user scans pages the model is already warm.
 */
if (typeof window !== "undefined") {
  void loadFenifyModel();
}

export async function classifyBoardFen(
  canvas: HTMLCanvasElement,
  bbox: BBox
): Promise<BoardClassification> {
  // ── 1. Fenify ML model (highest accuracy, requires public/fenify/model.onnx) ──
  const fenifyResult = await classifyFenifyVariants(canvas, bbox);
  if (fenifyResult?.valid) {
    return {
      fen: fenifyResult.fen,
      bbox: fenifyResult.bbox,
      confidence: fenifyResult.confidence,
      source: "fenify",
      notes: [
        "Position recognised by the Fenify neural network (99.8% per-square accuracy on book diagrams).",
        "Verify side-to-move and castling rights before deep analysis.",
      ],
    };
  }

  if (fenifyResult) {
    return lowConfidenceFen([
      "Fenify ran, but the crop did not produce a legal chess position with one white king and one black king.",
      "Use Crop board to select only the 8x8 board, excluding coordinate labels and nearby text.",
    ]);
  }

  // ── 2. Heuristic fallbacks (no model file present) ────────────────────────
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return lowConfidenceFen(["Canvas analysis unavailable."]);
  }

  const occupancy = squareInkDensity(context, bbox);
  const topMaterial = occupancy.slice(0, 16).filter((value) => value > 0.18).length;
  const middleMaterial = occupancy.slice(16, 48).filter((value) => value > 0.18).length;
  const bottomMaterial = occupancy.slice(48).filter((value) => value > 0.18).length;

  if (topMaterial >= 12 && bottomMaterial >= 12 && middleMaterial <= 6) {
    return {
      fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      confidence: 0.68,
      source: "template",
      notes: ["The board resembles a starting position. Confirm before serious study."],
    };
  }

  const templateResult = classifyBoardWithTemplates(context.canvas, bbox);
  if (templateResult) {
    return templateResult;
  }

  // Template matching failed (expected for scanned/printed books where piece pixels
  // don’t match Unicode glyph shapes). Build a best-effort FEN from which squares
  // are occupied so the board at least shows the correct pawn/piece structure.
  const occupancyFen = buildOccupancyFen(occupancy);
  if (occupancyFen) {
    return {
      fen: occupancyFen,
      confidence: 0.32,
      source: "occupancy",
      notes: [
        "Piece types estimated from ink density — squares with pieces are shown but piece identity needs correction.",
        "Run  scripts/convert_fenify_to_onnx.py  to enable the Fenify ML classifier for accurate piece identification.",
        "Use Edit mode or advanced position notation to fix piece identities in the meantime.",
      ],
    };
  }

  return lowConfidenceFen([
    "A board-like grid was detected but no pieces were distinguishable.",
    "Use Crop board or Edit mode to set the correct position.",
  ]);
}

export async function recognizeChessText(canvas: HTMLCanvasElement, bbox: BBox, onProgress?: (status: string, progress: number) => void) {
  const { createWorker, PSM } = await import("tesseract.js");
  const worker = await createWorker("eng", 1, {
    workerPath: "/tesseract/worker.min.js",
    corePath: "/tesseract",
    langPath: "/tessdata",
    workerBlobURL: false,
    gzip: true,
    logger: (message) => {
      if (typeof message.progress === "number") {
        onProgress?.(message.status ?? "ocr", message.progress);
      }
    },
  });

  try {
    await worker.setParameters({
      tessedit_pageseg_mode: PSM.SINGLE_BLOCK,
      tessedit_char_whitelist:
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ♔♕♖♗♘♙♚♛♜♝♞♟xXOo+#=.-–—…/ ",
      preserve_interword_spaces: "1",
    });

    const rectangle = textRectangle(canvas, bbox);
    const {
      data: { text, confidence },
    } = await worker.recognize(canvas, { rectangle });

    return {
      text: text.trim(),
      confidence: Math.max(0, Math.min(1, confidence / 100)),
    };
  } finally {
    await worker.terminate();
  }
}

export async function scanRenderedPage(page: RenderedPage): Promise<PageScanResult> {
  const candidates = detectBoardCandidates(page.canvas);
  const diagrams: DetectedDiagram[] = [];
  const lines: RecognizedLine[] = [];

  for (const candidate of candidates.slice(0, 6)) {
    const classification = await classifyBoardFen(page.canvas, candidate);
    // Suppress false positives aggressively when running on heuristic sources
    // (no Fenify model available). Heuristics misfire on tables, figures and
    // dense text columns in scanned chess books.
    const isHeuristic = classification.source !== "fenify";
    const likelyFalsePositive =
      isHeuristic
        ? classification.confidence < 0.42 || candidate.confidence < 0.52
        : classification.confidence <= 0.28 && candidate.confidence < 0.46;
    if (likelyFalsePositive) {
      continue;
    }
    const diagram: DetectedDiagram = {
      id: crypto.randomUUID(),
      pageIndex: page.pageIndex,
      bbox: classification.bbox ?? candidate,
      fen: classification.fen,
      confidence: combineDiagramConfidence(candidate.confidence, classification),
      recognitionSource: classification.source,
      orientation: "white",
      sourceCropUrl: cropToDataUrl(page.canvas, classification.bbox ?? candidate),
      notes: classification.notes,
    };
    diagrams.push(diagram);

    try {
      const ocr = await recognizeChessText(page.canvas, candidate);
      const parsed = parseRecognizedLine(ocr.text, diagram.fen);
      lines.push({
        id: crypto.randomUUID(),
        diagramId: diagram.id,
        rawText: ocr.text,
        normalizedText: parsed.normalizedText,
        sanMoves: parsed.sanMoves,
        confidence: Math.min(ocr.confidence, parsed.confidence),
        parseErrors: parsed.parseErrors,
      });
    } catch (error) {
      lines.push({
        id: crypto.randomUUID(),
        diagramId: diagram.id,
        rawText: "",
        normalizedText: "",
        sanMoves: [],
        confidence: 0,
        parseErrors: [error instanceof Error ? error.message : "OCR failed"],
      });
    }
  }

  return { page, diagrams, lines };
}

export async function scanManualBoard(page: RenderedPage, bbox: BBox): Promise<BoardScanResult> {
  const normalized = clampSquareBox(bbox, page.width, page.height);
  const classification = await classifyBoardFen(page.canvas, normalized);
  const diagram: DetectedDiagram = {
    id: crypto.randomUUID(),
    pageIndex: page.pageIndex,
    bbox: normalized,
    fen: classification.fen,
    confidence: Math.max(0.42, classification.confidence),
    recognitionSource: classification.source,
    orientation: "white",
    sourceCropUrl: cropToDataUrl(page.canvas, normalized),
    notes: ["Manual crop. Check piece identities before deep analysis."],
  };

  try {
    const ocr = await recognizeChessText(page.canvas, normalized);
    const parsed = parseRecognizedLine(ocr.text, diagram.fen);
    return {
      diagram,
      line: {
        id: crypto.randomUUID(),
        diagramId: diagram.id,
        rawText: ocr.text,
        normalizedText: parsed.normalizedText,
        sanMoves: parsed.sanMoves,
        confidence: Math.min(ocr.confidence, parsed.confidence),
        parseErrors: parsed.parseErrors,
      },
    };
  } catch (error) {
    return {
      diagram,
      line: {
        id: crypto.randomUUID(),
        diagramId: diagram.id,
        rawText: "",
        normalizedText: "",
        sanMoves: [],
        confidence: 0,
        parseErrors: [error instanceof Error ? error.message : "OCR failed"],
      },
    };
  }
}

function lowConfidenceFen(notes: string[]) {
  return {
    fen: "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    confidence: 0.28,
    source: "fallback" as const,
    notes,
  };
}

async function classifyFenifyVariants(canvas: HTMLCanvasElement, bbox: BBox) {
  let best:
    | {
        fen: string;
        bbox: BBox;
        confidence: number;
        valid: boolean;
        structureScore: number;
        score: number;
      }
    | null = null;

  for (const probeBox of fenifyProbeBoxes(canvas, bbox)) {
    const result = await classifyWithFenify(canvas, probeBox);
    if (!result) {
      return null;
    }

    const structureScore = scoreFenStructure(result.fen);
    const valid = structureScore >= 0.74;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const visualScore = context ? scoreBoardCandidate(context, probeBox.x, probeBox.y, probeBox.width) : 0;
    const candidate = {
      fen: result.fen,
      bbox: probeBox,
      confidence: Math.max(0.5, result.confidence * 0.8 + structureScore * 0.2),
      valid,
      structureScore,
      score: structureScore * 0.55 + result.confidence * 0.25 + visualScore * 0.2,
    };

    if (
      !best ||
      Number(candidate.valid) > Number(best.valid) ||
      candidate.score > best.score
    ) {
      best = candidate;
    }
  }

  return best;
}

function fenifyProbeBoxes(canvas: HTMLCanvasElement, bbox: BBox): BBox[] {
  const probes: BBox[] = [];
  const borderBox = tightenBoardByBorder(canvas, bbox);
  const densityBox = tightenBoardByDensity(canvas, bbox);
  if (borderBox) {
    probes.push(borderBox);
  }
  probes.push(...bookDiagramInnerBoxes(canvas, densityBox ?? bbox));
  const bases = densityBox ? [densityBox, bbox] : [bbox];
  const scales = [1, 0.88];
  const shifts = [
    [0, 0],
    [-0.03, 0],
    [0.03, 0],
  ] as const;

  for (const base of bases) {
    const centerX = base.x + base.width / 2;
    const centerY = base.y + base.height / 2;

    for (const scale of scales) {
      const size = Math.max(48, Math.min(canvas.width, canvas.height, base.width * scale));
      for (const [shiftX, shiftY] of shifts) {
        probes.push(
          clampSquareBox(
            {
              x: centerX - size / 2 + base.width * shiftX,
              y: centerY - size / 2 + base.height * shiftY,
              width: size,
              height: size,
            },
            canvas.width,
            canvas.height,
          ),
        );
      }
    }
  }

  return dedupeBoxes(probes);
}

function bookDiagramInnerBoxes(canvas: HTMLCanvasElement, bbox: BBox): BBox[] {
  return [
    { x: bbox.x + bbox.width * 0.25, y: bbox.y + bbox.height * 0.1, width: bbox.width * 0.68, height: bbox.width * 0.68 },
    { x: bbox.x + bbox.width * 0.22, y: bbox.y + bbox.height * 0.1, width: bbox.width * 0.7, height: bbox.width * 0.7 },
    { x: bbox.x + bbox.width * 0.18, y: bbox.y + bbox.height * 0.1, width: bbox.width * 0.72, height: bbox.width * 0.72 },
    { x: bbox.x + bbox.width * 0.16, y: bbox.y + bbox.height * 0.08, width: bbox.width * 0.76, height: bbox.width * 0.76 },
  ].map((box) => clampSquareBox(box, canvas.width, canvas.height));
}

function tightenBoardByBorder(canvas: HTMLCanvasElement, bbox: BBox): BBox | null {
  const sampleSize = 240;
  const sample = window.document.createElement("canvas");
  sample.width = sampleSize;
  sample.height = sampleSize;
  const sampleContext = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleContext) {
    return null;
  }

  sampleContext.fillStyle = "#fff";
  sampleContext.fillRect(0, 0, sampleSize, sampleSize);
  sampleContext.drawImage(canvas, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, sampleSize, sampleSize);
  const data = sampleContext.getImageData(0, 0, sampleSize, sampleSize).data;
  const columns = new Array<number>(sampleSize).fill(0);
  const rows = new Array<number>(sampleSize).fill(0);

  for (let y = 0; y < sampleSize; y += 1) {
    for (let x = 0; x < sampleSize; x += 1) {
      const offset = (y * sampleSize + x) * 4;
      const luminance = (data[offset] + data[offset + 1] + data[offset + 2]) / 765;
      if (luminance < 0.42) {
        columns[x] += 1;
        rows[y] += 1;
      }
    }
  }

  const xLines = linePeaks(columns.map((value) => value / sampleSize));
  const yLines = linePeaks(rows.map((value) => value / sampleSize));
  let best: { x1: number; x2: number; y1: number; y2: number; score: number } | null = null;

  for (const x1 of xLines) {
    for (const x2 of xLines) {
      if (x2 <= x1) continue;
      const width = x2 - x1;
      if (width < sampleSize * 0.38 || width > sampleSize * 0.94) continue;
      for (const y1 of yLines) {
        for (const y2 of yLines) {
          if (y2 <= y1) continue;
          const height = y2 - y1;
          const size = Math.max(width, height);
          if (Math.abs(width - height) > size * 0.14) continue;
          if (height < sampleSize * 0.38 || height > sampleSize * 0.94) continue;
          const score = columns[x1] + columns[x2] + rows[y1] + rows[y2] - Math.abs(width - height) * 0.15;
          if (!best || score > best.score) {
            best = { x1, x2, y1, y2, score };
          }
        }
      }
    }
  }

  if (!best) {
    return null;
  }

  const x = bbox.x + (best.x1 / sampleSize) * bbox.width;
  const y = bbox.y + (best.y1 / sampleSize) * bbox.height;
  const width = ((best.x2 - best.x1) / sampleSize) * bbox.width;
  const height = ((best.y2 - best.y1) / sampleSize) * bbox.height;
  const size = Math.max(width, height);

  return clampSquareBox(
    {
      x: x - size * 0.01,
      y: y - size * 0.01,
      width: size * 1.02,
      height: size * 1.02,
    },
    canvas.width,
    canvas.height,
  );
}

function linePeaks(values: number[]) {
  const threshold = Math.max(0.08, percentile(values, 0.9) * 0.72);
  const peaks: number[] = [];
  let start = -1;
  let mass = 0;

  for (let index = 0; index <= values.length; index += 1) {
    if (index < values.length && values[index] >= threshold) {
      if (start === -1) {
        start = index;
        mass = 0;
      }
      mass += values[index];
      continue;
    }

    if (start !== -1) {
      const end = index - 1;
      const center = Math.round((start + end) / 2);
      if (mass / (end - start + 1) >= threshold) {
        peaks.push(center);
      }
      start = -1;
      mass = 0;
    }
  }

  return peaks;
}

function tightenBoardByDensity(canvas: HTMLCanvasElement, bbox: BBox): BBox | null {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return null;
  }

  const sampleSize = 180;
  const sample = window.document.createElement("canvas");
  sample.width = sampleSize;
  sample.height = sampleSize;
  const sampleContext = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleContext) {
    return null;
  }

  sampleContext.fillStyle = "#fff";
  sampleContext.fillRect(0, 0, sampleSize, sampleSize);
  sampleContext.drawImage(canvas, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, sampleSize, sampleSize);
  const data = sampleContext.getImageData(0, 0, sampleSize, sampleSize).data;
  const columns = new Array<number>(sampleSize).fill(0);
  const rows = new Array<number>(sampleSize).fill(0);

  for (let y = 0; y < sampleSize; y += 1) {
    for (let x = 0; x < sampleSize; x += 1) {
      const offset = (y * sampleSize + x) * 4;
      const luminance = (data[offset] + data[offset + 1] + data[offset + 2]) / 765;
      const ink = Math.max(0, 1 - luminance);
      if (ink > 0.055) {
        columns[x] += ink;
        rows[y] += ink;
      }
    }
  }

  const xSegment = denseSegment(columns.map((value) => value / sampleSize));
  const ySegment = denseSegment(rows.map((value) => value / sampleSize));
  if (!xSegment || !ySegment) {
    return null;
  }

  const x = bbox.x + (xSegment.start / sampleSize) * bbox.width;
  const y = bbox.y + (ySegment.start / sampleSize) * bbox.height;
  const width = ((xSegment.end - xSegment.start + 1) / sampleSize) * bbox.width;
  const height = ((ySegment.end - ySegment.start + 1) / sampleSize) * bbox.height;
  const size = Math.min(width, height);

  if (size < bbox.width * 0.42 || size > bbox.width * 1.02) {
    return null;
  }

  return clampSquareBox(
    {
      x: x + Math.max(0, (width - size) / 2),
      y: y + Math.max(0, (height - size) / 2),
      width: size,
      height: size,
    },
    canvas.width,
    canvas.height,
  );
}

function denseSegment(values: number[]) {
  const smoothed = values.map((_, index) => {
    const start = Math.max(0, index - 3);
    const end = Math.min(values.length - 1, index + 3);
    return average(values.slice(start, end + 1));
  });
  const threshold = Math.max(0.018, percentile(smoothed, 0.58));
  let best: { start: number; end: number; mass: number } | null = null;
  let start = -1;
  let mass = 0;

  for (let index = 0; index <= smoothed.length; index += 1) {
    if (index < smoothed.length && smoothed[index] >= threshold) {
      if (start === -1) {
        start = index;
        mass = 0;
      }
      mass += smoothed[index];
      continue;
    }

    if (start !== -1) {
      const segment = { start, end: index - 1, mass };
      const length = segment.end - segment.start + 1;
      if (length >= values.length * 0.34 && (!best || segment.mass > best.mass)) {
        best = segment;
      }
      start = -1;
      mass = 0;
    }
  }

  return best;
}

function dedupeBoxes(boxes: BBox[]) {
  const seen = new Set<string>();
  const unique: BBox[] = [];
  for (const box of boxes) {
    const key = `${Math.round(box.x)}:${Math.round(box.y)}:${Math.round(box.width)}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(box);
    }
  }
  return unique;
}

function scoreFenStructure(fen: string) {
  const rows = fen.split(" ")[0]?.split("/") ?? [];
  if (rows.length !== 8) {
    return 0;
  }

  let whiteKings = 0;
  let blackKings = 0;
  let whitePawns = 0;
  let blackPawns = 0;
  const pieceCounts = new Map<string, number>();
  let pieces = 0;
  let pawnsOnBackRank = 0;

  rows.forEach((row, rowIndex) => {
    for (const char of row) {
      if (/\d/.test(char)) {
        continue;
      }
      pieces += 1;
      pieceCounts.set(char, (pieceCounts.get(char) ?? 0) + 1);
      if (char === "K") whiteKings += 1;
      if (char === "k") blackKings += 1;
      if (char === "P") {
        whitePawns += 1;
        if (rowIndex === 0 || rowIndex === 7) pawnsOnBackRank += 1;
      }
      if (char === "p") {
        blackPawns += 1;
        if (rowIndex === 0 || rowIndex === 7) pawnsOnBackRank += 1;
      }
    }
  });

  let score = 0;
  if (whiteKings === 1) score += 0.32;
  if (blackKings === 1) score += 0.32;
  if (pieces >= 2 && pieces <= 32) score += 0.14;
  if (whitePawns <= 8 && blackPawns <= 8) score += 0.1;
  if (pawnsOnBackRank === 0) score += 0.12;
  if (
    (pieceCounts.get("Q") ?? 0) > 1 ||
    (pieceCounts.get("q") ?? 0) > 1 ||
    (pieceCounts.get("R") ?? 0) > 2 ||
    (pieceCounts.get("r") ?? 0) > 2 ||
    (pieceCounts.get("B") ?? 0) > 2 ||
    (pieceCounts.get("b") ?? 0) > 2 ||
    (pieceCounts.get("N") ?? 0) > 2 ||
    (pieceCounts.get("n") ?? 0) > 2
  ) {
    score -= 0.3;
  }
  return score;
}

function combineDiagramConfidence(candidateConfidence: number, classification: BoardClassification) {
  if (classification.source === "fenify") {
    return Math.max(
      0.56,
      Math.min(0.95, classification.confidence * 0.82 + candidateConfidence * 0.18),
    );
  }

  if (classification.source === "template") {
    return Math.max(candidateConfidence, Math.min(0.82, classification.confidence));
  }

  return Math.min(candidateConfidence, classification.confidence);
}

/**
 * Build a best-effort FEN by inferring which squares are occupied from ink
 * density, assigning kings to the most-likely positions and generic pieces
 * (pawn-like) to other occupied squares.
 * Returns null if not enough material is found.
 */
function buildOccupancyFen(occupancy: number[]): string | null {
  const threshold = 0.19;
  const occupied = occupancy
    .map((density, index) => ({ density, index }))
    .filter((s) => s.density > threshold)
    .sort((a, b) => b.density - a.density);

  if (occupied.length < 3) {
    return null;
  }

  const board: string[][] = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => ""));

  // Try to identify kings by the darkest single squares in each half
  const blackHalf = occupied.filter((s) => Math.floor(s.index / 8) <= 2);
  const whiteHalf = occupied.filter((s) => Math.floor(s.index / 8) >= 5);
  const blackKingSquare = blackHalf[0];
  const whiteKingSquare = whiteHalf[0];

  if (!blackKingSquare || !whiteKingSquare) {
    return null;
  }

  const usedIndices = new Set<number>();
  board[Math.floor(blackKingSquare.index / 8)][blackKingSquare.index % 8] = "k";
  board[Math.floor(whiteKingSquare.index / 8)][whiteKingSquare.index % 8] = "K";
  usedIndices.add(blackKingSquare.index);
  usedIndices.add(whiteKingSquare.index);

  // Fill remaining occupied squares with generic piece markers
  for (const sq of occupied) {
    if (usedIndices.has(sq.index)) continue;
    const rank = Math.floor(sq.index / 8);
    const file = sq.index % 8;
    // Use pawns away from promotion rows and rooks on edge rows so the FEN
    // remains legal enough for chess.js and Stockfish.
    if (rank === 0) {
      board[rank][file] = "r";
    } else if (rank === 7) {
      board[rank][file] = "R";
    } else {
      board[rank][file] = rank <= 3 ? "p" : "P";
    }
    usedIndices.add(sq.index);
  }

  const rows = board.map(collapseFenRow);
  return `${rows.join("/")} w - - 0 1`;
}

function makeThumbnail(source: HTMLCanvasElement, maxWidth: number): string {
  const scale = maxWidth / source.width;
  const canvas = window.document.createElement("canvas");
  canvas.width = maxWidth;
  canvas.height = Math.round(source.height * scale);
  const context = canvas.getContext("2d");
  context?.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.78);
}

function cropToDataUrl(source: HTMLCanvasElement, bbox: BBox): string {
  const canvas = window.document.createElement("canvas");
  canvas.width = Math.max(1, Math.floor(bbox.width));
  canvas.height = Math.max(1, Math.floor(bbox.height));
  const context = canvas.getContext("2d");
  context?.drawImage(source, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.82);
}

function classifyBoardWithTemplates(source: HTMLCanvasElement, bbox: BBox): BoardClassification | null {
  const templates = getPieceTemplates();
  const board: string[][] = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => ""));
  const classified: Array<{ rank: number; file: number; fen: string; score: number }> = [];
  let occupied = 0;
  let whiteKings = 0;
  let blackKings = 0;

  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const ink = squareInkVector(source, bbox, rank, file);
      const inkDensity = vectorAverage(ink);
      if (inkDensity < 0.035) {
        continue;
      }

      let best = { fen: "", score: 0 };
      let second = 0;
      for (const template of templates) {
        const score = cosineSimilarity(ink, template.ink, template.norm);
        if (score > best.score) {
          second = best.score;
          best = { fen: template.fen, score };
        } else if (score > second) {
          second = score;
        }
      }

      const margin = best.score - second;
      if (best.score < 0.34 || margin < 0.012) {
        continue;
      }

      const score = Math.min(1, best.score * 0.82 + margin * 2);
      board[rank][file] = best.fen;
      classified.push({ rank, file, fen: best.fen, score });
      occupied += 1;
      if (best.fen === "K") {
        whiteKings += 1;
      }
      if (best.fen === "k") {
        blackKings += 1;
      }
    }
  }

  if (occupied < 2) {
    return null;
  }

  if (blackKings !== 1) {
    const blackKing = classified.filter((square) => square.rank <= 3).sort((a, b) => b.score - a.score)[0];
    if (!blackKing) {
      return null;
    }
    board[blackKing.rank][blackKing.file] = "k";
  }

  if (whiteKings !== 1) {
    const whiteKing = classified.filter((square) => square.rank >= 4).sort((a, b) => b.score - a.score)[0];
    if (!whiteKing) {
      return null;
    }
    board[whiteKing.rank][whiteKing.file] = "K";
  }

  const confidence = Math.max(0.34, Math.min(0.82, vectorAverage(classified.map((square) => square.score))));
  return {
    fen: `${board.map(collapseFenRow).join("/")} w - - 0 1`,
    confidence,
    source: "template",
    notes: [
      "Pieces were recognized with the local image classifier.",
      "Scanned book diagrams can still need correction when the print is faint or skewed.",
    ],
  };
}

function getPieceTemplates() {
  if (pieceTemplates) {
    return pieceTemplates;
  }

  const canvas = window.document.createElement("canvas");
  canvas.width = TEMPLATE_SIZE;
  canvas.height = TEMPLATE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    pieceTemplates = [];
    return pieceTemplates;
  }

  pieceTemplates = TEMPLATE_PIECES.map((piece) => {
    context.clearRect(0, 0, TEMPLATE_SIZE, TEMPLATE_SIZE);
    context.fillStyle = "#fff";
    context.fillRect(0, 0, TEMPLATE_SIZE, TEMPLATE_SIZE);
    context.fillStyle = "#000";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = `${Math.floor(TEMPLATE_SIZE * 0.86)}px "Apple Symbols", "DejaVu Sans", "Noto Sans Symbols 2", serif`;
    context.fillText(piece.glyph, TEMPLATE_SIZE / 2, TEMPLATE_SIZE / 2 + TEMPLATE_SIZE * 0.02);
    const data = context.getImageData(0, 0, TEMPLATE_SIZE, TEMPLATE_SIZE).data;
    const ink = new Float32Array(TEMPLATE_SIZE * TEMPLATE_SIZE);
    for (let index = 0; index < ink.length; index += 1) {
      const offset = index * 4;
      const luminance = (data[offset] + data[offset + 1] + data[offset + 2]) / 3;
      ink[index] = Math.max(0, (255 - luminance) / 255);
    }
    return { fen: piece.fen, ink, norm: vectorNorm(ink) };
  });

  return pieceTemplates;
}

function squareInkVector(source: HTMLCanvasElement, bbox: BBox, rank: number, file: number): Float32Array {
  const cell = bbox.width / 8;
  const margin = cell * 0.08;
  const cropX = bbox.x + file * cell + margin;
  const cropY = bbox.y + rank * cell + margin;
  const cropSize = Math.max(2, cell - margin * 2);
  const canvas = window.document.createElement("canvas");
  canvas.width = TEMPLATE_SIZE;
  canvas.height = TEMPLATE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return new Float32Array(TEMPLATE_SIZE * TEMPLATE_SIZE);
  }

  context.fillStyle = "#fff";
  context.fillRect(0, 0, TEMPLATE_SIZE, TEMPLATE_SIZE);
  context.imageSmoothingEnabled = true;
  context.drawImage(source, cropX, cropY, cropSize, cropSize, 0, 0, TEMPLATE_SIZE, TEMPLATE_SIZE);
  const data = context.getImageData(0, 0, TEMPLATE_SIZE, TEMPLATE_SIZE).data;
  const border: number[] = [];
  for (let y = 0; y < TEMPLATE_SIZE; y += 1) {
    for (let x = 0; x < TEMPLATE_SIZE; x += 1) {
      if (x < 3 || y < 3 || x >= TEMPLATE_SIZE - 3 || y >= TEMPLATE_SIZE - 3) {
        const offset = (y * TEMPLATE_SIZE + x) * 4;
        border.push((data[offset] + data[offset + 1] + data[offset + 2]) / 3);
      }
    }
  }
  const background = percentile(border, 0.72);
  const ink = new Float32Array(TEMPLATE_SIZE * TEMPLATE_SIZE);
  for (let index = 0; index < ink.length; index += 1) {
    const offset = index * 4;
    const luminance = (data[offset] + data[offset + 1] + data[offset + 2]) / 3;
    ink[index] = Math.max(0, Math.min(1, (background - luminance) / 150));
  }
  return ink;
}

function collapseFenRow(row: string[]): string {
  let result = "";
  let empty = 0;
  for (const piece of row) {
    if (!piece) {
      empty += 1;
      continue;
    }
    if (empty > 0) {
      result += String(empty);
      empty = 0;
    }
    result += piece;
  }
  if (empty > 0) {
    result += String(empty);
  }
  return result;
}

function textRectangle(canvas: HTMLCanvasElement, bbox: BBox) {
  // Extend capture area: book lines often appear to the right of or below the diagram.
  // Extra right margin (× 1.6) and a larger bottom margin (× 2.4) to capture multi-line
  // move sequences that may be printed beneath the board.
  const left = Math.max(0, Math.floor(bbox.x - bbox.width * 0.12));
  const top = Math.max(0, Math.floor(bbox.y - bbox.height * 0.08));
  const right = Math.min(canvas.width, Math.floor(bbox.x + bbox.width * 1.6));
  const bottom = Math.min(canvas.height, Math.floor(bbox.y + bbox.height * 2.4));
  return {
    left,
    top,
    width: Math.max(140, right - left),
    height: Math.max(80, bottom - top),
  };
}

function clampSquareBox(bbox: BBox, width: number, height: number): BBox {
  const size = Math.max(32, Math.min(bbox.width, bbox.height, width, height));
  const x = Math.max(0, Math.min(width - size, bbox.x));
  const y = Math.max(0, Math.min(height - size, bbox.y));
  return { x, y, width: size, height: size };
}

function refineBoardCandidate(
  context: CanvasRenderingContext2D,
  candidate: BBox & { confidence: number },
): BBox & { confidence: number } {
  let best = candidate;
  const originalSize = candidate.width;
  const scaleFactors = [0.9, 0.96, 1, 1.04, 1.1];

  for (const scale of scaleFactors) {
    const size = Math.floor(originalSize * scale);
    if (size < 64 || size > Math.min(context.canvas.width, context.canvas.height)) {
      continue;
    }

    const step = Math.max(3, Math.floor(size / 32));
    const centerX = candidate.x + candidate.width / 2;
    const centerY = candidate.y + candidate.height / 2;

    for (let dy = -step * 3; dy <= step * 3; dy += step) {
      for (let dx = -step * 3; dx <= step * 3; dx += step) {
        const x = Math.max(0, Math.min(context.canvas.width - size, Math.floor(centerX - size / 2 + dx)));
        const y = Math.max(0, Math.min(context.canvas.height - size, Math.floor(centerY - size / 2 + dy)));
        const confidence = scoreBoardCandidate(context, x, y, size);
        if (confidence > best.confidence) {
          best = { x, y, width: size, height: size, confidence };
        }
      }
    }
  }

  return best;
}

function scoreBoardCandidate(context: CanvasRenderingContext2D, x: number, y: number, size: number): number {
  const cell = size / 8;
  const centers: number[] = [];
  const gridSamples: number[] = [];

  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      centers.push(brightnessAt(context, x + file * cell + cell / 2, y + rank * cell + cell / 2));
    }
  }

  for (let i = 1; i < 8; i += 1) {
    const offset = i * cell;
    for (let t = 0; t <= 8; t += 1) {
      gridSamples.push(edgeDelta(context, x + offset, y + t * cell, true));
      gridSamples.push(edgeDelta(context, x + t * cell, y + offset, false));
    }
  }

  const alternation = checkerboardAlternation(centers);
  const grid = average(gridSamples);
  const contrast = standardDeviation(centers);
  return Math.max(0, Math.min(1, alternation * 0.48 + grid * 0.34 + contrast * 0.18));
}

function checkerboardAlternation(values: number[]): number {
  let scoreA = 0;
  let scoreB = 0;
  let pairs = 0;

  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const index = rank * 8 + file;
      const expectedA = (rank + file) % 2 === 0 ? 1 : -1;
      const centered = values[index] - 0.5;
      scoreA += centered * expectedA;
      scoreB += centered * -expectedA;
      pairs += 1;
    }
  }

  return Math.min(1, (Math.abs(Math.max(scoreA, scoreB)) / pairs) * 2);
}

function squareInkDensity(context: CanvasRenderingContext2D, bbox: BBox): number[] {
  const densities: number[] = [];
  const cell = bbox.width / 8;

  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const x = Math.floor(bbox.x + file * cell + cell * 0.18);
      const y = Math.floor(bbox.y + rank * cell + cell * 0.18);
      const size = Math.max(4, Math.floor(cell * 0.64));
      const data = context.getImageData(x, y, size, size).data;
      let dark = 0;
      for (let i = 0; i < data.length; i += 4) {
        const value = (data[i] + data[i + 1] + data[i + 2]) / 765;
        if (value < 0.58) {
          dark += 1;
        }
      }
      densities.push(dark / (data.length / 4));
    }
  }

  return densities;
}

function edgeDelta(context: CanvasRenderingContext2D, x: number, y: number, vertical: boolean): number {
  const a = brightnessAt(context, x + (vertical ? -3 : 0), y + (vertical ? 0 : -3));
  const b = brightnessAt(context, x + (vertical ? 3 : 0), y + (vertical ? 0 : 3));
  return Math.abs(a - b);
}

function brightnessAt(context: CanvasRenderingContext2D, x: number, y: number): number {
  const data = context.getImageData(Math.max(0, Math.floor(x)), Math.max(0, Math.floor(y)), 1, 1).data;
  return (data[0] + data[1] + data[2]) / 765;
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function vectorAverage(values: ArrayLike<number>): number {
  if (values.length === 0) {
    return 0;
  }
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    sum += values[index];
  }
  return sum / values.length;
}

function vectorNorm(values: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    sum += values[index] * values[index];
  }
  return Math.sqrt(sum);
}

function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>, bNorm: number): number {
  const aNorm = vectorNorm(a);
  if (aNorm === 0 || bNorm === 0) {
    return 0;
  }
  let dot = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
  }
  return dot / (aNorm * bNorm);
}

function percentile(values: number[], ratio: number): number {
  if (values.length === 0) {
    return 255;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.min(sorted.length - 1, Math.floor(sorted.length * ratio)));
  return sorted[index];
}

function standardDeviation(values: number[]): number {
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.min(1, Math.sqrt(variance) * 3);
}

function nonOverlapping(candidates: Array<BBox & { confidence: number }>) {
  const kept: Array<BBox & { confidence: number }> = [];
  for (const candidate of candidates) {
    if (kept.every((existing) => intersectionOverUnion(existing, candidate) < 0.4)) {
      kept.push(candidate);
    }
  }
  return kept;
}

function intersectionOverUnion(a: BBox, b: BBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area = a.width * a.height + b.width * b.height - intersection;
  return area === 0 ? 0 : intersection / area;
}
