import {
  classifyBoardFen,
  detectBoardCandidates as detectBaseBoardCandidates,
  loadPdfDocument,
  recognizeChessText,
  renderPdfPage,
  scanManualBoard,
  type BoardScanResult,
  type PageScanResult,
  type PdfDocument,
  type RenderedPage,
} from "./pdf-processing";
import { parseRecognizedLine } from "@/lib/move-parser";
import type { BBox, DetectedDiagram, RecognizedLine } from "@/lib/types";

export {
  classifyBoardFen,
  detectBaseBoardCandidates as detectBoardCandidates,
  loadPdfDocument,
  recognizeChessText,
  renderPdfPage,
  scanManualBoard,
};
export type { BoardScanResult, PageScanResult, PdfDocument, RenderedPage };

type BoardCandidate = BBox & { confidence: number; detector?: "base" | "line-grid" };
type BoardClassification = Awaited<ReturnType<typeof classifyBoardFen>>;

/**
 * More permissive page scanning wrapper used by the app. The original scanner
 * is kept as the conservative primitive; this wrapper adds monochrome line-grid
 * detection and source-specific false-positive filtering so scanned chess-book
 * diagrams are still surfaced when Fenify is unavailable.
 */
export async function scanRenderedPage(page: RenderedPage): Promise<PageScanResult> {
  const candidates = mergeCandidateBoxes([
    ...detectBaseBoardCandidates(page.canvas).map((candidate) => ({ ...candidate, detector: "base" as const })),
    ...detectLineGridCandidates(page.canvas),
  ]).slice(0, 10);

  const diagrams: DetectedDiagram[] = [];
  const lines: RecognizedLine[] = [];

  for (const candidate of candidates) {
    const classification = await classifyBoardFen(page.canvas, candidate);
    if (!shouldKeepScanResult(classification, candidate)) {
      continue;
    }

    const bbox = classification.bbox ?? candidate;
    const diagram: DetectedDiagram = {
      id: crypto.randomUUID(),
      pageIndex: page.pageIndex,
      bbox,
      fen: classification.fen,
      confidence: combineDiagramConfidence(candidate.confidence, classification),
      recognitionSource: classification.source,
      orientation: "white",
      sourceCropUrl: cropToDataUrl(page.canvas, bbox),
      notes: scanNotes(classification, candidate),
    };
    diagrams.push(diagram);

    try {
      const ocr = await recognizeChessText(page.canvas, bbox);
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

function shouldKeepScanResult(classification: BoardClassification, candidate: BoardCandidate) {
  switch (classification.source) {
    case "fenify":
      return classification.confidence > 0.28 || candidate.confidence >= 0.46;
    case "template":
      return classification.confidence >= 0.38 && candidate.confidence >= 0.32;
    case "occupancy":
      return classification.confidence >= 0.3 && candidate.confidence >= 0.33;
    case "fallback":
      return false;
    default:
      return false;
  }
}

function combineDiagramConfidence(candidateConfidence: number, classification: BoardClassification) {
  if (classification.source === "fenify") {
    return Math.max(0.56, Math.min(0.95, classification.confidence * 0.82 + candidateConfidence * 0.18));
  }

  if (classification.source === "template") {
    return Math.max(0.42, Math.min(0.82, classification.confidence * 0.72 + candidateConfidence * 0.28));
  }

  if (classification.source === "occupancy") {
    // Keep occupancy boards loadable/editable, but below the non-Fenify
    // auto-apply threshold so a guessed position is never silently installed.
    return Math.max(0.3, Math.min(0.49, classification.confidence * 0.7 + candidateConfidence * 0.3));
  }

  return Math.min(candidateConfidence, classification.confidence);
}

function scanNotes(classification: BoardClassification, candidate: BoardCandidate) {
  if (candidate.detector !== "line-grid") {
    return classification.notes;
  }
  return [
    "Detected by line-grid scanning for monochrome book diagrams.",
    ...classification.notes,
  ];
}

function detectLineGridCandidates(canvas: HTMLCanvasElement): BoardCandidate[] {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context || canvas.width < 160 || canvas.height < 160) {
    return [];
  }

  const minDimension = Math.min(canvas.width, canvas.height);
  const minSize = Math.max(72, Math.floor(minDimension * 0.075));
  const maxSize = Math.floor(minDimension * 0.74);
  const candidates: BoardCandidate[] = [];

  for (let size = maxSize; size >= minSize; size -= Math.max(10, Math.floor(size / 9))) {
    const step = Math.max(10, Math.floor(size / 5));
    for (let y = 0; y <= canvas.height - size; y += step) {
      for (let x = 0; x <= canvas.width - size; x += step) {
        const confidence = scoreLineGridCandidate(context, x, y, size);
        if (confidence >= 0.3) {
          candidates.push({ x, y, width: size, height: size, confidence, detector: "line-grid" });
        }
      }
    }
  }

  const refined = nonOverlapping(candidates.sort((a, b) => b.confidence - a.confidence))
    .slice(0, 18)
    .map((candidate) => refineLineGridCandidate(context, candidate))
    .sort((a, b) => b.confidence - a.confidence);

  return nonOverlapping(refined).slice(0, 8);
}

function refineLineGridCandidate(context: CanvasRenderingContext2D, candidate: BoardCandidate): BoardCandidate {
  let best = candidate;
  const originalSize = candidate.width;
  const scales = [0.9, 0.96, 1, 1.04, 1.1];

  for (const scale of scales) {
    const size = Math.floor(originalSize * scale);
    if (size < 48 || size > Math.min(context.canvas.width, context.canvas.height)) {
      continue;
    }

    const centerX = candidate.x + candidate.width / 2;
    const centerY = candidate.y + candidate.height / 2;
    const step = Math.max(2, Math.floor(size / 40));
    for (let dy = -step * 4; dy <= step * 4; dy += step) {
      for (let dx = -step * 4; dx <= step * 4; dx += step) {
        const x = Math.max(0, Math.min(context.canvas.width - size, Math.floor(centerX - size / 2 + dx)));
        const y = Math.max(0, Math.min(context.canvas.height - size, Math.floor(centerY - size / 2 + dy)));
        const confidence = scoreLineGridCandidate(context, x, y, size);
        if (confidence > best.confidence) {
          best = { x, y, width: size, height: size, confidence, detector: "line-grid" };
        }
      }
    }
  }

  return best;
}

function scoreLineGridCandidate(context: CanvasRenderingContext2D, x: number, y: number, size: number) {
  const cell = size / 8;
  const lineDensities: number[] = [];
  const midDensities: number[] = [];

  for (let index = 0; index <= 8; index += 1) {
    const offset = index * cell;
    lineDensities.push(darkLineDensity(context, x + offset, y, size, true));
    lineDensities.push(darkLineDensity(context, x, y + offset, size, false));
  }

  for (let index = 0; index < 8; index += 1) {
    const offset = index * cell + cell / 2;
    midDensities.push(darkLineDensity(context, x + offset, y, size, true));
    midDensities.push(darkLineDensity(context, x, y + offset, size, false));
  }

  const lineScore = average(lineDensities);
  const midScore = average(midDensities);
  const contrastScore = Math.max(0, lineScore - midScore * 0.72);
  const regularityScore = Math.max(0, 1 - standardDeviation(lineDensities) * 3.8);
  const borderScore = average([lineDensities[0], lineDensities[1], lineDensities[16], lineDensities[17]]);

  return Math.max(
    0,
    Math.min(1, contrastScore * 1.75 + lineScore * 0.65 + borderScore * 0.22 + regularityScore * 0.16),
  );
}

function darkLineDensity(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  length: number,
  vertical: boolean,
) {
  const samples = 72;
  const spread = Math.max(1, Math.floor(length / 360));
  let dark = 0;
  let total = 0;

  for (let index = 0; index <= samples; index += 1) {
    const t = (index / samples) * length;
    for (let offset = -spread; offset <= spread; offset += 1) {
      const px = vertical ? x + offset : x + t;
      const py = vertical ? y + t : y + offset;
      if (px < 0 || py < 0 || px >= context.canvas.width || py >= context.canvas.height) {
        continue;
      }
      if (brightnessAt(context, px, py) < 0.74) {
        dark += 1;
      }
      total += 1;
    }
  }

  return total === 0 ? 0 : dark / total;
}

function brightnessAt(context: CanvasRenderingContext2D, x: number, y: number) {
  const data = context.getImageData(Math.max(0, Math.floor(x)), Math.max(0, Math.floor(y)), 1, 1).data;
  return (data[0] + data[1] + data[2]) / 765;
}

function cropToDataUrl(source: HTMLCanvasElement, bbox: BBox): string {
  const canvas = window.document.createElement("canvas");
  canvas.width = Math.max(1, Math.floor(bbox.width));
  canvas.height = Math.max(1, Math.floor(bbox.height));
  const context = canvas.getContext("2d");
  context?.drawImage(source, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.82);
}

function mergeCandidateBoxes(candidates: BoardCandidate[]) {
  return nonOverlapping(candidates.sort((a, b) => b.confidence - a.confidence));
}

function nonOverlapping(candidates: BoardCandidate[]) {
  const kept: BoardCandidate[] = [];
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

function average(values: number[]) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[]) {
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}
