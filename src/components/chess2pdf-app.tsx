"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { DEFAULT_DEPTH, MAX_CACHED_CANVASES, OCR_CONCURRENCY, STARTING_FEN } from "@/lib/constants";
import { clearFen, isSquare, isValidFen, movePieceInFen, placePieceInFen, safeFen } from "@/lib/fen-editor";
import { validatePdfFile } from "@/lib/file-validation";
import { lineToPgn, parseRecognizedLine } from "@/lib/move-parser";
import {
  loadPdfDocument,
  renderPdfPage,
  scanManualBoard,
  scanRenderedPage,
  type PdfDocument,
  type RenderedPage,
} from "@/lib/pdf-processing";
import { fenifyStatus, isFenifyReady } from "@/lib/fenify-inference";
import { clearLocalData, listSessions, saveSession } from "@/lib/storage";
import { StockfishClient } from "@/lib/stockfish";
import type { ChessAiMode, ChessAiResponse } from "@/lib/ai-chess";
import type { BBox, DetectedDiagram, EngineEval, PdfPagePreview, PdfSession, PieceCode, RecognizedLine, ScanStatus } from "@/lib/types";

const AUTO_APPLY_CONFIDENCE = 0.55;
const UNRECOGNIZED_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";

const PIECES: PieceCode[] = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"];
const PIECE_LABEL: Record<PieceCode, string> = {
  wK: "White king",
  wQ: "White queen",
  wR: "White rook",
  wB: "White bishop",
  wN: "White knight",
  wP: "White pawn",
  bK: "Black king",
  bQ: "Black queen",
  bR: "Black rook",
  bB: "Black bishop",
  bN: "Black knight",
  bP: "Black pawn",
};

type BoardOrientation = "white" | "black";
type CropPoint = { x: number; y: number };

export function Chess2PdfApp() {
  const [status, setStatus] = useState<ScanStatus>("ready");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("Open a PDF to begin.");
  const [hasPdfLoaded, setHasPdfLoaded] = useState(false);
  const [fileMeta, setFileMeta] = useState<{ name: string; size: number } | null>(null);
  const [pages, setPages] = useState<PdfPagePreview[]>([]);
  const [totalPages, setTotalPages] = useState(0);
  const [scannedPages, setScannedPages] = useState<number[]>([]);
  const [currentPage, setCurrentPage] = useState(0);
  const [scanRangeStart, setScanRangeStart] = useState(1);
  const [scanRangeEnd, setScanRangeEnd] = useState(1);
  const [diagrams, setDiagrams] = useState<DetectedDiagram[]>([]);
  const [lines, setLines] = useState<RecognizedLine[]>([]);
  const [selectedDiagramId, setSelectedDiagramId] = useState<string | null>(null);
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<PdfSession[]>([]);
  const [fen, setFen] = useState(UNRECOGNIZED_FEN);
  const [boardHasPosition, setBoardHasPosition] = useState(false);
  const [fenDraft, setFenDraft] = useState(UNRECOGNIZED_FEN);
  const [playedMoves, setPlayedMoves] = useState<string[]>([]);
  const [deviationPly, setDeviationPly] = useState<number | undefined>();
  const [engineEval, setEngineEval] = useState<EngineEval | undefined>();
  const [bookEval, setBookEval] = useState<EngineEval | undefined>();
  const [evalDeltaCp, setEvalDeltaCp] = useState<number | undefined>();
  const [engineStatus, setEngineStatus] = useState("Engine idle");
  const [orientation, setOrientation] = useState<BoardOrientation>("white");
  const [editMode, setEditMode] = useState(false);
  const [selectedPiece, setSelectedPiece] = useState<PieceCode | null>("wQ");
  const [ocrTextDraft, setOcrTextDraft] = useState("");
  const [aiStatus, setAiStatus] = useState("AI coach idle");
  const [aiResult, setAiResult] = useState<ChessAiResponse | null>(null);
  const [aiConfigured, setAiConfigured] = useState<boolean | null>(null);
  const [cropMode, setCropMode] = useState(false);
  const [cropStart, setCropStart] = useState<CropPoint | null>(null);
  const [cropBox, setCropBox] = useState<BBox | null>(null);
  const [modelStatus, setModelStatus] = useState<"idle" | "loading" | "ready" | "unavailable">("idle");

  const canvasesRef = useRef(new Map<number, HTMLCanvasElement>());
  const pagePreviewsRef = useRef(new Map<number, PdfPagePreview>());
  const scannedPagesRef = useRef(new Set<number>());
  const pdfRef = useRef<PdfDocument | null>(null);
  const stockfishRef = useRef<StockfishClient | null>(null);
  const scanRunRef = useRef(0);

  const selectedDiagram = diagrams.find((diagram) => diagram.id === selectedDiagramId) ?? null;
  const selectedLines = selectedDiagram ? lines.filter((line) => line.diagramId === selectedDiagram.id) : [];
  const expectedLine = selectedLines.find((line) => line.id === selectedLineId) ?? selectedLines.find((line) => line.sanMoves.length > 0) ?? selectedLines[0];
  const pagePreviewMap = useMemo(() => new Map(pages.map((page) => [page.pageIndex, page])), [pages]);
  const scannedPageSet = useMemo(() => new Set(scannedPages), [scannedPages]);
  const diagramCountByPage = useMemo(() => {
    const counts = new Map<number, number>();
    for (const diagram of diagrams) {
      counts.set(diagram.pageIndex, (counts.get(diagram.pageIndex) ?? 0) + 1);
    }
    return counts;
  }, [diagrams]);
  const currentChess = useMemo(() => {
    try {
      return new Chess(fen);
    } catch {
      return null;
    }
  }, [fen]);

  const resetBoard = useCallback((nextFen = UNRECOGNIZED_FEN, options: { loaded?: boolean } = {}) => {
    const resolved = safeFen(nextFen);
    setFen(resolved);
    setBoardHasPosition(options.loaded ?? resolved !== UNRECOGNIZED_FEN);
    setFenDraft(resolved);
    setPlayedMoves([]);
    setDeviationPly(undefined);
    setEngineEval(undefined);
    setBookEval(undefined);
    setEvalDeltaCp(undefined);
    setEngineStatus("Engine idle");
  }, []);

  const refreshSessions = useCallback(async () => {
    setSessions(await listSessions());
  }, []);

  useEffect(() => {
    stockfishRef.current = new StockfishClient();
    queueMicrotask(() => {
      void refreshSessions();
    });
    // Probe whether AI coach is configured (OPENROUTER_API_KEY set on server)
    void fetch("/api/ai/chess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "line-summary", startingFen: "", currentFen: "", recognizedMoves: [], playedMoves: [], rawText: "" }),
    })
      .then((r) => r.json())
      .then((data: { configured?: boolean }) => setAiConfigured(data.configured !== false))
      .catch(() => setAiConfigured(false));

    return () => {
      stockfishRef.current?.dispose();
      pdfRef.current?.destroy();
    };
  }, [refreshSessions]);

  // Poll for Fenify model load status so the UI can reflect when recognition is ready.
  useEffect(() => {
    setModelStatus(fenifyStatus());
    if (isFenifyReady()) return;
    const timer = setInterval(() => {
      const s = fenifyStatus();
      setModelStatus(s);
      if (s === "ready" || s === "unavailable") {
        clearInterval(timer);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, []);

  async function handleFile(file: File) {
    setError("");
    setStatus("loading");
    setProgress("Checking PDF safety...");

    const validation = await validatePdfFile(file);
    if (!validation.ok) {
      setStatus("error");
      setError(validation.reason);
      return;
    }

    try {
      pdfRef.current?.destroy();
      canvasesRef.current.clear();
      pagePreviewsRef.current.clear();
      scannedPagesRef.current.clear();
      setScannedPages([]);
      const document = await loadPdfDocument(file);
      pdfRef.current = document;
      setHasPdfLoaded(true);
      setFileMeta({ name: file.name, size: file.size });
      setPages([]);
      setTotalPages(document.numPages);
      setScanRangeStart(1);
      setScanRangeEnd(Math.min(document.numPages, 20));
      setDiagrams([]);
      setLines([]);
      setSelectedDiagramId(null);
      setSelectedLineId(null);
      setFen(UNRECOGNIZED_FEN);
      setBoardHasPosition(false);
      setFenDraft(UNRECOGNIZED_FEN);
      setPlayedMoves([]);
      setDeviationPly(undefined);
      setCropMode(false);
      setCropStart(null);
      setCropBox(null);
      setBookEval(undefined);
      setEvalDeltaCp(undefined);
      setEngineStatus("Engine idle");
      setCurrentPage(0);
      setProgress(`Opened ${document.numPages} page${document.numPages === 1 ? "" : "s"}. Rendering page 1...`);
      await renderAndCachePage(document, 0);
      setStatus("ready");
      setProgress("Page 1 ready. Scan the current page or choose a range.");
      await scanPages([0]);
    } catch (scanError) {
      setStatus("error");
      setError(scanError instanceof Error ? scanError.message : "Could not open this PDF.");
    }
  }

  async function scanPages(indices: number[]) {
    const document = pdfRef.current;
    if (!document) {
      setProgress("No PDF loaded yet.");
      return;
    }
    const loadedDocument = document;

    const runId = scanRunRef.current + 1;
    scanRunRef.current = runId;
    setStatus("scanning");
    setError("");
    setProgress(`Scanning ${indices.length} page${indices.length === 1 ? "" : "s"} for boards and lines...`);

    const queue = [...indices];
    const newDiagrams: DetectedDiagram[] = [];
    const newLines: RecognizedLine[] = [];

    async function worker() {
      while (queue.length > 0 && scanRunRef.current === runId) {
        const pageIndex = queue.shift();
        if (pageIndex === undefined) {
          return;
        }
        const rendered = await getRenderedPage(loadedDocument, pageIndex);
        setProgress(`Scanning page ${pageIndex + 1} with local OCR and grid detection...`);
        const result = await scanRenderedPage(rendered);
        newDiagrams.push(...result.diagrams);
        newLines.push(...result.lines);
        setDiagrams((current) => mergeById(current, result.diagrams, (item) => item.id));
        setLines((current) => mergeById(current, result.lines, (item) => item.id));
      }
    }

    await Promise.all(Array.from({ length: Math.min(OCR_CONCURRENCY, indices.length) }, () => worker()));

    if (scanRunRef.current !== runId) {
      setProgress("Scan cancelled.");
      setStatus("ready");
      return;
    }

    // mark each scanned page so navigateToPage won't re-scan them
    for (const pageIndex of indices) {
      scannedPagesRef.current.add(pageIndex);
    }
    setScannedPages(Array.from(scannedPagesRef.current.values()).sort((a, b) => a - b));

    const mergedDiagrams = mergeById(diagrams, newDiagrams, (item) => item.id);
    const mergedLines = mergeById(lines, newLines, (item) => item.id);

    // Use indices[0] as the priority page, since currentPage may still hold the
    // previous value (React state updates are async, not yet flushed).
    const priorityPage = indices[0] ?? currentPage;
    const currentPageDiagram = bestDiagramForPage(priorityPage, mergedDiagrams);
    const firstDetected = currentPageDiagram ?? newDiagrams[0];
    if (firstDetected) {
      await selectDiagram(firstDetected, mergedLines, { auto: true });
    }

    const scannedCount = indices.length;
    const foundDiagramCount = newDiagrams.length;
    const foundRecognizedBoardCount = newDiagrams.filter((diagram) => canLoadBoard(diagram)).length;
    const foundLineCount = newLines.filter((line) => line.sanMoves.length > 0).length;
    setProgress(
      foundRecognizedBoardCount > 0
        ? `Scan complete: ${foundRecognizedBoardCount} reliable board${foundRecognizedBoardCount === 1 ? "" : "s"}, ${foundDiagramCount} board-like crop${foundDiagramCount === 1 ? "" : "s"}, and ${foundLineCount} parsed line${foundLineCount === 1 ? "" : "s"} across ${scannedCount} page${scannedCount === 1 ? "" : "s"}.`
        : foundDiagramCount > 0
        ? `Scan complete: ${foundDiagramCount} board-like crop${foundDiagramCount === 1 ? "" : "s"} found, but none were reliable enough to load automatically. Try Crop board on the exact 8x8 diagram.`
        : "No board-like diagram found in that scan range.",
    );
    setStatus("ready");

    const session = buildSession(mergedDiagrams, mergedLines);
    if (session && (newDiagrams.length > 0 || foundLineCount > 0)) {
      await saveSession(session);
      await refreshSessions();
    }
  }

  async function getRenderedPage(document: PdfDocument, pageIndex: number) {
    const canvas = canvasesRef.current.get(pageIndex);
    const preview = pagePreviewsRef.current.get(pageIndex);
    if (canvas && preview) {
      return { ...preview, canvas };
    }
    return renderAndCachePage(document, pageIndex);
  }

  async function navigateToPage(pageIndex: number) {
    const document = pdfRef.current;
    const boundedPageIndex = Math.max(0, Math.min(totalPages - 1, pageIndex));
    setCurrentPage(boundedPageIndex);
    if (!document) {
      return;
    }
    if (!pagePreviewsRef.current.has(boundedPageIndex) || !canvasesRef.current.has(boundedPageIndex)) {
      setProgress(`Rendering page ${boundedPageIndex + 1}...`);
      await renderAndCachePage(document, boundedPageIndex);
      setProgress(`Page ${boundedPageIndex + 1} ready.`);
    }
    // If already scanned, auto-select best diagram on this page
    if (scannedPagesRef.current.has(boundedPageIndex)) {
      const pageFirstDiagram = bestDiagramForPage(boundedPageIndex, diagrams);
      if (pageFirstDiagram) {
        await selectDiagram(pageFirstDiagram, lines, { auto: true, navigate: false });
      } else {
        clearBoardSelection(`Page ${boundedPageIndex + 1} has been scanned, but no board was recognised on this page.`);
      }
    }
    // Don't auto-scan — user clicks "Scan page" / "Scan range" explicitly.
    // Auto-scanning every unvisited page was causing ~8s OCR freezes per click.
  }

  async function renderAndCachePage(document: PdfDocument, pageIndex: number) {
    const rendered = await renderPdfPage(document, pageIndex);
    canvasesRef.current.set(pageIndex, rendered.canvas);
    pagePreviewsRef.current.set(pageIndex, stripCanvas(rendered));
    evictCachedPages(pageIndex);
    setPages(Array.from(pagePreviewsRef.current.values()).sort((a, b) => a.pageIndex - b.pageIndex));
    return rendered;
  }

  function evictCachedPages(anchorPageIndex: number) {
    if (canvasesRef.current.size <= MAX_CACHED_CANVASES) {
      return;
    }

    const keep = new Set(
      Array.from(canvasesRef.current.keys())
        .sort((a, b) => Math.abs(a - anchorPageIndex) - Math.abs(b - anchorPageIndex))
        .slice(0, MAX_CACHED_CANVASES),
    );

    for (const pageIndex of Array.from(canvasesRef.current.keys())) {
      if (!keep.has(pageIndex)) {
        canvasesRef.current.delete(pageIndex);
        pagePreviewsRef.current.delete(pageIndex);
      }
    }
  }

  function buildSession(sessionDiagrams: DetectedDiagram[], sessionLines: RecognizedLine[]): PdfSession | null {
    if (!fileMeta) {
      return null;
    }
    return {
      id: crypto.randomUUID(),
      fileName: fileMeta.name,
      fileSize: fileMeta.size,
      createdAt: new Date().toISOString(),
      pages,
      diagrams: sessionDiagrams,
      exercises: sessionLines,
    };
  }

  function cancelScan() {
    scanRunRef.current += 1;
    setProgress("Cancelling after the current OCR job...");
  }

  function bestDiagramForPage(pageIndex: number, sourceDiagrams: DetectedDiagram[]) {
    return sourceDiagrams
      .filter((diagram) => diagram.pageIndex === pageIndex)
      .sort((a, b) => b.confidence - a.confidence)[0];
  }

  function canLoadBoard(diagram: DetectedDiagram | null | undefined) {
    if (!diagram || !isValidFen(diagram.fen)) {
      return false;
    }
    if (diagram.recognitionSource === "fenify") {
      return true;
    }
    if (diagram.recognitionSource === "template") {
      return diagram.confidence >= 0.5;
    }
    if (diagram.recognitionSource === "occupancy") {
      return diagram.confidence >= 0.3;
    }
    return false;
  }

  function canAutoApplyBoard(diagram: DetectedDiagram | null | undefined) {
    if (!diagram || !canLoadBoard(diagram)) {
      return false;
    }
    return diagram.recognitionSource === "fenify" || diagram.confidence >= AUTO_APPLY_CONFIDENCE;
  }

  function boardLoadLabel(diagram: DetectedDiagram) {
    if (!canLoadBoard(diagram)) {
      return "Crop found, pieces not reliable";
    }
    if (diagram.recognitionSource === "fenify") {
      return "Ready to load";
    }
    if (diagram.recognitionSource === "template") {
      return "Template match, please verify";
    }
    return "Occupancy estimate, edit before analysis";
  }

  function clearBoardSelection(message: string) {
    setSelectedDiagramId(null);
    setSelectedLineId(null);
    setOcrTextDraft("");
    resetBoard(UNRECOGNIZED_FEN, { loaded: false });
    setProgress(message);
  }

  async function selectDiagram(
    diagram: DetectedDiagram,
    sourceLines = lines,
    options: { auto?: boolean; navigate?: boolean } = {},
  ) {
    const shouldNavigatePage = options.navigate ?? true;
    const canApplyBoard = canLoadBoard(diagram);
    const shouldApplyBoard = canApplyBoard && (!options.auto || canAutoApplyBoard(diagram));
    setSelectedDiagramId(diagram.id);
    if (shouldNavigatePage && diagram.pageIndex !== currentPage) {
      setCurrentPage(diagram.pageIndex);
    }
    const line = sourceLines.find((item) => item.diagramId === diagram.id);
    setSelectedLineId(line?.id ?? null);
    setOcrTextDraft(line?.rawText || "");
    if (!isValidFen(diagram.fen)) {
      setEngineEval(undefined);
      setBookEval(undefined);
      setEvalDeltaCp(undefined);
      setEngineStatus("Engine idle");
      setProgress("Recognition found a board crop, but the position was not legal. Crop only the 8x8 board or choose another detected board.");
      return;
    }
    if (!canApplyBoard) {
      resetBoard(UNRECOGNIZED_FEN, { loaded: false });
      setEngineEval(undefined);
      setBookEval(undefined);
      setEvalDeltaCp(undefined);
      setEngineStatus("Engine idle");
      setProgress(
        "A board-like crop was found, but piece recognition was not reliable enough to load it. Try Crop board on the exact 8x8 diagram or scan another page.",
      );
      return;
    }
    if (!shouldApplyBoard) {
      const moveCount = line?.sanMoves.length ?? 0;
      resetBoard(UNRECOGNIZED_FEN, { loaded: false });
      setEngineEval(undefined);
      setBookEval(undefined);
      setEvalDeltaCp(undefined);
      setEngineStatus("Engine idle");
      setProgress(
        moveCount > 0
          ? `Low-confidence board selected. ${moveCount} move${moveCount === 1 ? "" : "s"} detected, but the position was not auto-applied.`
          : "Low-confidence board selected. Use Crop board or Edit mode before analysis.",
      );
      return;
    }
    resetBoard(diagram.fen, { loaded: true });
    if (diagram.confidence < AUTO_APPLY_CONFIDENCE) {
      const moveCount = line?.sanMoves.length ?? 0;
      setProgress(
        moveCount > 0
          ? `Board loaded (low confidence). ${moveCount} move${moveCount === 1 ? "" : "s"} detected — step through with the move buttons or chips.`
          : "Board loaded. Confidence is low — use Edit mode if the pieces look wrong.",
      );
    }
    await analyzePosition(diagram.fen);
  }

  function applyFenDraft() {
    if (!isValidFen(fenDraft)) {
      setError("That position is invalid. Keep both kings on the board before analysis.");
      return;
    }
    setError("");
    resetBoard(fenDraft, { loaded: true });
    setProgress("Position applied.");
    void analyzePosition(fenDraft);
  }

  function reparseOcrDraft() {
    const targetDiagramId = selectedDiagram?.id ?? "manual";
    const parsed = parseRecognizedLine(ocrTextDraft, safeFen(fen));
    const updated: RecognizedLine = {
      id: expectedLine?.id ?? crypto.randomUUID(),
      diagramId: targetDiagramId,
      rawText: ocrTextDraft,
      normalizedText: parsed.normalizedText,
      sanMoves: parsed.sanMoves,
      confidence: parsed.confidence,
      parseErrors: parsed.parseErrors,
    };

    setLines((current) => {
      const without = current.filter((line) => line.id !== updated.id);
      return [...without, updated];
    });
    setSelectedLineId(updated.id);
    setProgress(`Parsed ${updated.sanMoves.length} legal move${updated.sanMoves.length === 1 ? "" : "s"}.`);
  }

  function handlePieceDrop({ sourceSquare, targetSquare }: { sourceSquare: string; targetSquare: string | null }) {
    if (!targetSquare) {
      return false;
    }

    if (!isSquare(sourceSquare) || !isSquare(targetSquare)) {
      return false;
    }

    if (editMode) {
      const nextFen = movePieceInFen(fen, sourceSquare, targetSquare);
      setFen(nextFen);
      setBoardHasPosition(true);
      setFenDraft(nextFen);
      setPlayedMoves([]);
      setDeviationPly(undefined);
      return true;
    }

    const game = currentChess;
    if (!game) {
      setError("Current position is not legal enough for move validation.");
      return false;
    }

    try {
      const positionBeforeMove = game.fen();
      const move = game.move({ from: sourceSquare, to: targetSquare, promotion: "q" });
      if (!move) {
        return false;
      }
      const nextMoves = [...playedMoves, move.san];
      const expected = expectedLine?.sanMoves[playedMoves.length];
      const deviated = expected !== undefined && move.san !== expected;
      let expectedReplyFen: string | undefined;
      if (deviated && expected) {
        try {
          const expectedGame = new Chess(positionBeforeMove);
          const expectedMove = expectedGame.move(expected, { strict: false });
          expectedReplyFen = expectedMove ? expectedGame.fen() : undefined;
        } catch {
          expectedReplyFen = undefined;
        }
      }
      setFen(game.fen());
      setBoardHasPosition(true);
      setFenDraft(game.fen());
      setPlayedMoves(nextMoves);
      if (deviated && deviationPly === undefined) {
        setDeviationPly(nextMoves.length);
      }
      void analyzePosition(game.fen(), expectedReplyFen);
      return true;
    } catch {
      return false;
    }
  }

  function handleSquareClick({ square }: { square: string }) {
    if (!editMode || !isSquare(square)) {
      return;
    }
    const nextFen = placePieceInFen(fen, square, selectedPiece);
    setFen(nextFen);
    setBoardHasPosition(true);
    setFenDraft(nextFen);
    setPlayedMoves([]);
    setDeviationPly(undefined);
    setEngineEval(undefined);
    setBookEval(undefined);
    setEvalDeltaCp(undefined);
    setEngineStatus("Engine idle");
  }

  function playExpectedMove() {
    if (!expectedLine || !currentChess) {
      return;
    }
    const san = expectedLine.sanMoves[playedMoves.length];
    if (!san) {
      setProgress("End of the recognized line.");
      return;
    }
    const move = currentChess.move(san, { strict: false });
    if (!move) {
      setError(`Could not play ${san} from the current position.`);
      return;
    }
    setFen(currentChess.fen());
    setBoardHasPosition(true);
    setFenDraft(currentChess.fen());
    const nextMoves = [...playedMoves, move.san];
    setPlayedMoves(nextMoves);
    void analyzePosition(currentChess.fen());
  }

  function undoMove() {
    const source = safeFen(selectedDiagram?.fen ?? STARTING_FEN);
    const game = new Chess(source);
    const nextMoves = playedMoves.slice(0, -1);
    for (const san of nextMoves) {
      game.move(san, { strict: false });
    }
    setFen(game.fen());
    setBoardHasPosition(true);
    setFenDraft(game.fen());
    setPlayedMoves(nextMoves);
    if (deviationPly !== undefined && nextMoves.length < deviationPly) {
      setDeviationPly(undefined);
    }
    void analyzePosition(game.fen());
  }

  function jumpToPly(targetPly: number) {
    if (!expectedLine) {
      return;
    }
    const source = safeFen(selectedDiagram?.fen ?? STARTING_FEN);
    const game = new Chess(source);
    const bounded = Math.max(0, Math.min(expectedLine.sanMoves.length, targetPly));
    const nextMoves: string[] = [];
    for (let index = 0; index < bounded; index += 1) {
      const san = expectedLine.sanMoves[index];
      const moved = game.move(san, { strict: false });
      if (!moved) {
        break;
      }
      nextMoves.push(moved.san);
    }
    setFen(game.fen());
    setBoardHasPosition(true);
    setFenDraft(game.fen());
    setPlayedMoves(nextMoves);
    setDeviationPly(undefined);
    void analyzePosition(game.fen());
  }

  async function analyzePosition(targetFen = fen, compareFen?: string) {
    if (!isValidFen(targetFen)) {
      setError("Stockfish needs a valid position with both kings.");
      return;
    }
    setEngineStatus(`Analyzing to depth ${DEFAULT_DEPTH} locally...`);
    setEvalDeltaCp(undefined);
    setBookEval(undefined);
    try {
      const result = await stockfishRef.current?.evaluate(targetFen, DEFAULT_DEPTH);
      if (result) {
        setEngineEval(result);
      }
      if (compareFen && isValidFen(compareFen)) {
        setEngineStatus("Comparing against the book move...");
        const expected = await stockfishRef.current?.evaluate(compareFen, DEFAULT_DEPTH);
        if (expected) {
          setBookEval(expected);
          if (result?.scoreCp !== undefined && expected.scoreCp !== undefined) {
            setEvalDeltaCp(result.scoreCp - expected.scoreCp);
          }
        }
      }
      setEngineStatus("Analysis ready");
    } catch (analysisError) {
      setEngineStatus(analysisError instanceof Error ? analysisError.message : "Engine failed");
    }
  }

  async function askAi(mode: ChessAiMode) {
    setAiStatus(mode === "line-summary" ? "Summarizing book line..." : "Explaining deviation...");
    setAiResult(null);

    try {
      const response = await fetch("/api/ai/chess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          startingFen: safeFen(selectedDiagram?.fen ?? STARTING_FEN),
          currentFen: fen,
          recognizedMoves: expectedLine?.sanMoves ?? [],
          playedMoves,
          rawText: expectedLine?.rawText ?? ocrTextDraft,
          deviationPly,
          engineEval,
        }),
      });
      const payload = (await response.json()) as ChessAiResponse;
      setAiResult(payload);
      setAiStatus(payload.ok ? "AI coach ready" : payload.title);
    } catch (aiError) {
      setAiResult({
        ok: false,
        configured: true,
        title: "AI request failed",
        explanation: aiError instanceof Error ? aiError.message : "Could not reach the AI route.",
      });
      setAiStatus("AI request failed");
    }
  }

  async function copyFen() {
    await navigator.clipboard.writeText(fen);
    setProgress("Position notation copied.");
  }

  async function copyPgn() {
    const pgn = expectedLine ? lineToPgn(safeFen(selectedDiagram?.fen ?? STARTING_FEN), expectedLine.sanMoves) : "";
    await navigator.clipboard.writeText(pgn || currentChess?.pgn() || "");
    setProgress("PGN copied.");
  }

  async function wipeLocalData() {
    await clearLocalData();
    await refreshSessions();
    setProgress("History cleared.");
  }

  async function restoreSession(session: PdfSession) {
    setFileMeta({ name: session.fileName, size: session.fileSize });
    setPages(session.pages);
    setDiagrams(session.diagrams);
    setLines(session.exercises);
    const inferredPages = Math.max(
      session.pages.reduce((max, page) => Math.max(max, page.pageIndex), 0) + 1,
      session.diagrams.reduce((max, diagram) => Math.max(max, diagram.pageIndex), 0) + 1,
    );
    setTotalPages(inferredPages);
    setHasPdfLoaded(session.pages.length > 0 || session.diagrams.length > 0);
    if (session.diagrams.length > 0) {
      setCurrentPage(session.diagrams[0].pageIndex);
      await selectDiagram(session.diagrams[0], session.exercises, { navigate: false });
    } else {
      setProgress(`Loaded ${session.fileName}, but no boards were stored in that session.`);
    }
  }

  function selectLine(line: RecognizedLine) {
    setSelectedLineId(line.id);
    setOcrTextDraft(line.rawText);
    const diagram = diagrams.find((item) => item.id === line.diagramId);
    if (diagram) {
      setSelectedDiagramId(diagram.id);
      if (canLoadBoard(diagram)) {
        const sourceFen = safeFen(diagram.fen);
        resetBoard(sourceFen, { loaded: true });
        void analyzePosition(sourceFen);
      } else {
        resetBoard(UNRECOGNIZED_FEN, { loaded: false });
        setProgress("That line belongs to a crop whose pieces were not recognised reliably enough to load.");
      }
    }
  }

  function scanSelectedRange() {
    if (!totalPages) {
      return;
    }
    const start = Math.max(1, Math.min(totalPages, scanRangeStart));
    const end = Math.max(start, Math.min(totalPages, scanRangeEnd));
    const indices = Array.from({ length: end - start + 1 }, (_, offset) => start - 1 + offset);
    void scanPages(indices);
  }

  function scanNextRange() {
    if (!totalPages) {
      return;
    }
    const nextStart = Math.min(totalPages, Math.max(scanRangeEnd + 1, 1));
    const nextEnd = Math.min(totalPages, nextStart + 19);
    setScanRangeStart(nextStart);
    setScanRangeEnd(nextEnd);
    const indices = Array.from({ length: nextEnd - nextStart + 1 }, (_, offset) => nextStart - 1 + offset);
    void scanPages(indices);
  }

  async function scanManualCrop() {
    const document = pdfRef.current;
    if (!document || !cropBox) {
      return;
    }

    setStatus("scanning");
    setProgress(`Scanning manual crop on page ${currentPage + 1}...`);
    setError("");

    try {
      const rendered = await getRenderedPage(document, currentPage);
      const result = await scanManualBoard(rendered, cropBox);
      const mergedDiagrams = mergeById(diagrams, [result.diagram], (item) => item.id);
      const mergedLines = mergeById(lines, [result.line], (item) => item.id);
      setDiagrams(mergedDiagrams);
      setLines(mergedLines);
      scannedPagesRef.current.add(currentPage);
      setScannedPages(Array.from(scannedPagesRef.current.values()).sort((a, b) => a - b));
      setCropMode(false);
      setCropStart(null);
      setCropBox(null);
      await selectDiagram(result.diagram, mergedLines);
      setProgress(
        result.line.sanMoves.length
          ? `Manual crop added with ${result.line.sanMoves.length} parsed move${result.line.sanMoves.length === 1 ? "" : "s"}.`
          : "Manual crop added. Correct the OCR text or position if needed.",
      );

      const session = buildSession(mergedDiagrams, mergedLines);
      if (session) {
        await saveSession(session);
        await refreshSessions();
      }
    } catch (cropError) {
      setError(cropError instanceof Error ? cropError.message : "Manual crop failed.");
    } finally {
      setStatus("ready");
    }
  }

  function beginCrop(event: PointerEvent<HTMLDivElement>) {
    if (!cropMode || !currentPreview) {
      return;
    }
    const point = cropPointFromEvent(event, currentPreview);
    setCropStart(point);
    setCropBox({ x: point.x, y: point.y, width: 1, height: 1 });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function updateCrop(event: PointerEvent<HTMLDivElement>) {
    if (!cropMode || !cropStart || !currentPreview) {
      return;
    }
    setCropBox(squareFromPoints(cropStart, cropPointFromEvent(event, currentPreview), currentPreview));
  }

  function finishCrop(event: PointerEvent<HTMLDivElement>) {
    if (!cropMode || !cropStart || !currentPreview) {
      return;
    }
    setCropBox(squareFromPoints(cropStart, cropPointFromEvent(event, currentPreview), currentPreview));
    setCropStart(null);
  }

  function moveDiagramSelection(offset: number) {
    if (!visiblePageDiagrams.length) {
      return;
    }
    const currentIndex = selectedPageDiagramIndex >= 0 ? selectedPageDiagramIndex : 0;
    const nextIndex = Math.max(0, Math.min(visiblePageDiagrams.length - 1, currentIndex + offset));
    const nextDiagram = visiblePageDiagrams[nextIndex];
    if (nextDiagram) {
      void selectDiagram(nextDiagram, lines, { navigate: false });
    }
  }

  const currentPreview = pagePreviewMap.get(currentPage);
  const visiblePageDiagrams = diagrams.filter((diagram) => diagram.pageIndex === currentPage).sort((a, b) => b.confidence - a.confidence);
  const visiblePageLineItems = visiblePageDiagrams.flatMap((diagram, diagramIndex) =>
    lines
      .filter((line) => line.diagramId === diagram.id)
      .map((line) => ({ diagram, diagramIndex, line })),
  );
  const selectedPageDiagramIndex = selectedDiagram ? visiblePageDiagrams.findIndex((diagram) => diagram.id === selectedDiagram.id) : -1;
  const selectedPageDiagramNumber = selectedPageDiagramIndex >= 0 ? selectedPageDiagramIndex + 1 : 0;
  const evalText = formatEval(engineEval);
  const lichessUrl = `https://lichess.org/analysis/standard/${fen.replace(/\s/g, "_")}`;

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-[1700px] flex-col gap-4 px-4 py-4">
        <header className="grid gap-3 border-b border-line pb-4 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <h1 className="text-3xl font-semibold">Chess2pdf</h1>
          </div>
          <div className="flex flex-wrap gap-2 text-sm">
            <label className="cursor-pointer rounded-md bg-accent px-4 py-2 font-semibold text-white hover:bg-accent-strong">
              Open PDF
              <input
                className="sr-only"
                type="file"
                accept="application/pdf,.pdf"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void handleFile(file);
                  }
                }}
              />
            </label>
            <button className="rounded-md border border-line bg-panel px-4 py-2 font-semibold" onClick={wipeLocalData}>
              Clear history
            </button>
          </div>
        </header>

        {error ? <div className="rounded-md border border-bad bg-white px-4 py-3 text-sm text-bad">{error}</div> : null}

        {status === "loading" || status === "scanning" ? (
          <div className="flex items-center gap-3 rounded-md border border-accent bg-[#e8f5f1] px-4 py-3 text-sm">
            <svg className="h-4 w-4 animate-spin text-accent" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="font-medium text-accent">{progress}</span>
          </div>
        ) : null}

        <section className="grid flex-1 gap-4 xl:grid-cols-[360px_minmax(430px,1fr)_420px]">
          <aside className="min-h-[520px] rounded-md border border-line bg-panel">
            <div className="border-b border-line p-4">
              <h2 className="text-lg font-semibold">PDF pages</h2>
              {fileMeta ? (
                <p className="text-sm text-muted">{`${fileMeta.name} (${formatBytes(fileMeta.size)}), ${totalPages} page${totalPages === 1 ? "" : "s"}`}</p>
              ) : null}
            </div>
            <div
              className="m-4 rounded-md border border-dashed border-line p-4 text-sm text-muted"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const file = event.dataTransfer.files[0];
                if (file) {
                  void handleFile(file);
                }
              }}
            >
              Drop a PDF here. The app checks the PDF signature and size before rendering it locally.
            </div>
            <div className="flex flex-wrap gap-2 px-4 pb-3">
              <button
                className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!hasPdfLoaded || status === "scanning"}
                onClick={() => void scanPages([currentPage])}
              >
                Scan page
              </button>
              <button
                className="rounded-md border border-line px-3 py-2 text-sm font-semibold"
                disabled={status !== "scanning"}
                onClick={cancelScan}
              >
                Cancel
              </button>
            </div>
            <div className="grid grid-cols-[1fr_1fr_auto_auto] gap-2 px-4 pb-3">
              <label className="text-xs font-semibold text-muted">
                From
                <input
                  className="mt-1 w-full rounded-md border border-line bg-white px-2 py-2 text-sm text-foreground"
                  min={1}
                  max={Math.max(1, totalPages)}
                  type="number"
                  value={scanRangeStart}
                  onChange={(event) => setScanRangeStart(Number(event.target.value))}
                />
              </label>
              <label className="text-xs font-semibold text-muted">
                To
                <input
                  className="mt-1 w-full rounded-md border border-line bg-white px-2 py-2 text-sm text-foreground"
                  min={1}
                  max={Math.max(1, totalPages)}
                  type="number"
                  value={scanRangeEnd}
                  onChange={(event) => setScanRangeEnd(Number(event.target.value))}
                />
              </label>
              <button
                className="self-end rounded-md border border-line px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!hasPdfLoaded || status === "scanning"}
                onClick={scanSelectedRange}
              >
                Scan range
              </button>
              <button
                className="self-end rounded-md border border-line px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!hasPdfLoaded || status === "scanning" || scanRangeEnd >= totalPages}
                onClick={scanNextRange}
              >
                Next 20
              </button>
              <p className="col-span-4 text-xs text-muted">Estimate: about 8 seconds per page on typical scans.</p>
              {/* Model status strip */}
              <div className={`col-span-4 flex items-center gap-2 rounded-md px-3 py-2 text-xs font-semibold ${
                modelStatus === "ready"
                  ? "bg-[#e8f5f1] text-accent"
                  : modelStatus === "unavailable"
                    ? "bg-[#fff3e0] text-warn"
                    : "bg-background text-muted"
              }`}>
                {modelStatus === "loading" ? (
                  <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : null}
                {modelStatus === "ready"
                  ? "AI board recognition ready — scan for accurate results."
                  : modelStatus === "loading"
                    ? "AI model loading (first time, ~30 s) — scan will use it once ready."
                    : modelStatus === "unavailable"
                      ? "AI model unavailable — results will be heuristic only."
                      : "AI model not yet started."}
              </div>
            </div>
            <div className="max-h-[58vh] overflow-auto px-4 pb-4">
              {totalPages === 0 ? (
                <div className="rounded-md bg-background p-4 text-sm text-muted">
                  Upload a PDF to see pages here.
                </div>
              ) : (
                <div className="grid grid-cols-4 gap-2">
                  {Array.from({ length: totalPages }, (_, index) => {
                    const preview = pagePreviewMap.get(index);
                    const diagramCount = diagramCountByPage.get(index) ?? 0;
                    const isScanned = scannedPageSet.has(index);
                    return (
                      <button
                        key={index}
                        className={`min-h-14 rounded-md border p-1 text-center text-xs font-semibold ${
                          index === currentPage ? "border-accent bg-[#e8f5f1]" : preview ? "border-line bg-white" : "border-line bg-[#f3f6f4]"
                        }`}
                        onClick={() => void navigateToPage(index)}
                      >
                        {preview ? (
                          <img className="mx-auto mb-1 max-h-12 rounded object-contain" src={preview.thumbnailUrl} alt={`PDF page ${index + 1}`} />
                        ) : null}
                        <span>{index + 1}</span>
                        {diagramCount > 0 ? <span className="ml-1 text-accent">{diagramCount}</span> : null}
                        {diagramCount === 0 && isScanned ? <span className="ml-1 text-muted">done</span> : null}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </aside>

          <section className="grid gap-4">
            <div className="rounded-md border border-line bg-panel p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold">Study board</h2>
                  <p className="text-sm text-muted">{progress}</p>
                </div>
                <div className="flex flex-wrap gap-2 text-sm">
                  <button className="rounded-md border border-line px-3 py-2 font-semibold" onClick={() => setOrientation(orientation === "white" ? "black" : "white")}>
                    Flip board
                  </button>
                  <button className="rounded-md border border-line px-3 py-2 font-semibold" onClick={() => setEditMode(!editMode)}>
                    {editMode ? "Play mode" : "Edit mode"}
                  </button>
                  <button
                    className="rounded-md border border-line px-3 py-2 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={!boardHasPosition && !canLoadBoard(selectedDiagram)}
                    onClick={() => {
                      const sourceFen = selectedDiagram && canLoadBoard(selectedDiagram) ? safeFen(selectedDiagram.fen) : fen;
                      resetBoard(sourceFen, { loaded: sourceFen !== UNRECOGNIZED_FEN });
                      if (sourceFen !== UNRECOGNIZED_FEN) {
                        void analyzePosition(sourceFen);
                      }
                    }}
                  >
                    Reset
                  </button>
                  <button
                    className="rounded-md border border-line px-3 py-2 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={selectedPageDiagramIndex <= 0}
                    onClick={() => moveDiagramSelection(-1)}
                  >
                    Prev board
                  </button>
                  <button
                    className="rounded-md border border-line px-3 py-2 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={!visiblePageDiagrams.length || selectedPageDiagramIndex >= visiblePageDiagrams.length - 1}
                    onClick={() => moveDiagramSelection(1)}
                  >
                    Next board
                  </button>
                </div>
              </div>
              {visiblePageDiagrams.length > 0 ? (
                <p className="mb-3 text-sm text-muted">
                  Page {currentPage + 1} board {selectedPageDiagramNumber || 1} of {visiblePageDiagrams.length}
                </p>
              ) : null}
              {selectedDiagram && selectedDiagram.pageIndex !== currentPage ? (
                <p className="mb-3 rounded-md bg-[#fff8e6] p-3 text-sm text-warn">
                  Loaded crop is from page {selectedDiagram.pageIndex + 1}. The PDF viewer is currently on page {currentPage + 1}.
                </p>
              ) : null}
              {selectedDiagram && boardHasPosition ? (
                <p className="mb-3 rounded-md bg-background p-3 text-sm text-muted">
                  Board source: page {selectedDiagram.pageIndex + 1}, confidence {Math.round(selectedDiagram.confidence * 100)} percent,{" "}
                  {selectedDiagram.recognitionSource === "fenify" ? "Fenify recognition" : "high-confidence template recognition"}.
                </p>
              ) : null}

              <div className="mx-auto max-w-[620px]">
                {boardHasPosition || editMode ? (
                  <Chessboard
                    options={{
                      id: "chess2pdf-board",
                      position: fen,
                      boardOrientation: orientation,
                      showNotation: true,
                      animationDurationInMs: 120,
                      darkSquareStyle: { backgroundColor: "#6f8f7f" },
                      lightSquareStyle: { backgroundColor: "#edf3ee" },
                      boardStyle: { borderRadius: 6, boxShadow: "0 10px 30px rgba(24, 33, 29, 0.16)" },
                      onPieceDrop: handlePieceDrop,
                      onSquareClick: handleSquareClick,
                    }}
                  />
                ) : (
                  <div className="grid aspect-square place-items-center rounded-md border border-dashed border-line bg-background p-8 text-center">
                    <div>
                      <p className="text-lg font-semibold">No reliable board loaded yet.</p>
                      <p className="mt-2 text-sm text-muted">
                        Scan a page with a clear chess diagram. If a crop is found but not loaded, use Crop board around the exact 8x8 board.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {editMode ? (
                <div className="mt-4 rounded-md border border-line bg-background p-3">
                  <p className="mb-2 text-sm font-semibold">Edit position</p>
                  <div className="flex flex-wrap gap-2">
                    {PIECES.map((piece) => (
                      <button
                        key={piece}
                        className={`rounded-md border px-3 py-2 text-sm ${selectedPiece === piece ? "border-accent bg-[#e8f5f1]" : "border-line bg-white"}`}
                        onClick={() => setSelectedPiece(piece)}
                        title={PIECE_LABEL[piece]}
                      >
                        {piece}
                      </button>
                    ))}
                    <button className="rounded-md border border-line bg-white px-3 py-2 text-sm" onClick={() => setSelectedPiece(null)}>
                      Erase
                    </button>
                    <button
                      className="rounded-md border border-line bg-white px-3 py-2 text-sm"
                      onClick={() => {
                        const next = clearFen();
                        setFen(next);
                        setBoardHasPosition(true);
                        setFenDraft(next);
                        setPlayedMoves([]);
                        setDeviationPly(undefined);
                        setEngineEval(undefined);
                        setBookEval(undefined);
                        setEvalDeltaCp(undefined);
                        setEngineStatus("Engine idle");
                      }}
                    >
                      Clear board
                    </button>
                  </div>
                </div>
              ) : null}

              {/* Compact engine eval bar */}
              <div className="mt-3 flex flex-wrap items-center gap-3 rounded-md bg-background px-3 py-2 text-sm">
                <span className={`font-semibold ${engineEval && (engineEval.scoreCp ?? 0) < -30 ? "text-bad" : "text-accent"}`}>{evalText}</span>
                {engineEval?.bestMove ? (
                  <span className="text-muted">
                    Best: <span className="font-medium text-foreground">{engineEval.bestMove}</span>
                  </span>
                ) : null}
                {evalDeltaCp !== undefined ? (
                  <span className={`font-semibold ${evalDeltaCp <= -20 ? "text-bad" : "text-accent"}`}>
                    Deviation: {formatCpDelta(evalDeltaCp)}
                  </span>
                ) : null}
                <span className="ml-auto text-xs text-muted">{engineStatus}</span>
              </div>
              {/* Move step controls */}
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  className="rounded-md border border-line px-4 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={!playedMoves.length}
                  onClick={undoMove}
                >
                  ← Prev
                </button>
                <button
                  className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={!boardHasPosition || !expectedLine?.sanMoves.length || playedMoves.length >= (expectedLine?.sanMoves.length ?? 0)}
                  onClick={playExpectedMove}
                >
                  Next →
                </button>
                <button
                  className="rounded-md border border-line px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={!playedMoves.length}
                  onClick={() => jumpToPly(0)}
                >
                  Reset line
                </button>
                <button
                  className="rounded-md border border-line px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={!boardHasPosition}
                  onClick={() => void analyzePosition()}
                >
                  Analyze
                </button>
              </div>
              {/* Advanced position notation — collapsed to reduce noise */}
              <details className="mt-3">
                <summary className="cursor-pointer rounded px-1 py-1 text-xs font-semibold text-muted hover:text-foreground">
                  Position tools (advanced)
                </summary>
                <div className="mt-2">
                  <label className="mb-1 block text-sm font-semibold" htmlFor="fen-input">
                    Position notation
                  </label>
                  <textarea
                    id="fen-input"
                    className="min-h-20 w-full rounded-md border border-line bg-white p-3 text-sm"
                    value={fenDraft}
                    onChange={(event) => setFenDraft(event.target.value)}
                  />
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white" onClick={applyFenDraft}>
                      Apply position
                    </button>
                    <button className="rounded-md border border-line px-3 py-2 text-sm font-semibold" onClick={() => void copyFen()}>
                      Copy notation
                    </button>
                    <a className="rounded-md border border-line px-3 py-2 text-sm font-semibold" href={lichessUrl} target="_blank" rel="noreferrer">
                      Open Lichess
                    </a>
                  </div>
                </div>
              </details>
            </div>

            <div className="rounded-md border border-line bg-panel p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-lg font-semibold">PDF page view</h2>
                  {cropMode ? <p className="text-sm text-muted">Drag a square around the board.</p> : null}
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    className="rounded-md border border-line px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={!currentPreview || status === "scanning"}
                    onClick={() => {
                      setCropMode(!cropMode);
                      setCropStart(null);
                      setCropBox(null);
                    }}
                  >
                    {cropMode ? "Cancel crop" : "Crop board"}
                  </button>
                  {cropMode ? (
                    <button
                      className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                      disabled={!cropBox || cropBox.width < 32 || status === "scanning"}
                      onClick={() => void scanManualCrop()}
                    >
                      Use crop
                    </button>
                  ) : null}
                </div>
              </div>
              {currentPreview ? (
                <div className="relative mx-auto max-h-[720px] overflow-auto rounded-md border border-line bg-white">
                  <div
                    className={`relative inline-block ${cropMode ? "cursor-crosshair" : ""}`}
                    onPointerDown={beginCrop}
                    onPointerMove={updateCrop}
                    onPointerUp={finishCrop}
                  >
                    <img className="max-w-none" src={currentPreview.imageUrl} alt={`Rendered PDF page ${currentPage + 1}`} />
                    {visiblePageDiagrams.map((diagram) => (
                      <button
                        key={diagram.id}
                        className={`absolute border-2 ${diagram.id === selectedDiagram?.id ? "border-accent bg-[#147c6c33]" : "border-[#147c6c88] bg-[#147c6c1f]"}`}
                        style={{
                          left: diagram.bbox.x,
                          top: diagram.bbox.y,
                          width: diagram.bbox.width,
                          height: diagram.bbox.height,
                        }}
                        title={`Detected diagram, ${Math.round(diagram.confidence * 100)} percent confidence`}
                        disabled={cropMode}
                        onClick={() => void selectDiagram(diagram)}
                      />
                    ))}
                    {cropBox ? (
                      <div
                        className="pointer-events-none absolute border-2 border-bad bg-[#c5403030]"
                        style={{
                          left: cropBox.x,
                          top: cropBox.y,
                          width: cropBox.width,
                          height: cropBox.height,
                        }}
                      />
                    ) : null}
                  </div>
                </div>
              ) : (
                <div className="rounded-md border border-dashed border-line p-8 text-center text-sm text-muted">
                  The rendered page will appear here after opening a PDF.
                </div>
              )}
            </div>
          </section>

          <aside className="grid gap-4">
            <div className="rounded-md border border-line bg-panel p-4">
              <h2 className="text-lg font-semibold">Detected boards</h2>
              <p className="mb-3 text-sm text-muted">
                Automatic recognition is best effort. Low-confidence diagrams should be corrected before analysis.
              </p>
              <div className="space-y-3">
                {visiblePageDiagrams.map((diagram, index) => (
                  <button
                    key={diagram.id}
                    className={`block w-full rounded-md border p-3 text-left ${diagram.id === selectedDiagram?.id ? "border-accent bg-[#e8f5f1]" : "border-line bg-white"}`}
                    onClick={() => void selectDiagram(diagram)}
                  >
                    <div className="flex gap-3">
                      {diagram.sourceCropUrl ? <img className="h-20 w-20 rounded object-cover" src={diagram.sourceCropUrl} alt={`Detected chess diagram ${index + 1}`} /> : null}
                      <div>
                        <p className="font-semibold">Board {index + 1}</p>
                        <p className="text-sm text-muted">Page {diagram.pageIndex + 1}, confidence {Math.round(diagram.confidence * 100)} percent</p>
                        <p className={`text-xs font-semibold ${canLoadBoard(diagram) ? "text-accent" : "text-warn"}`}>
                          {boardLoadLabel(diagram)}
                        </p>
                        {diagram.notes.map((note) => (
                          <p key={note} className="mt-1 text-xs text-warn">
                            {note}
                          </p>
                        ))}
                      </div>
                    </div>
                  </button>
                ))}
                {visiblePageDiagrams.length === 0 ? (
                  <p className="rounded-md bg-background p-3 text-sm text-muted">No detected boards on this page yet. Scan this page to populate boards.</p>
                ) : null}
              </div>
            </div>

            <div className="rounded-md border border-line bg-panel p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h2 className="text-lg font-semibold">Book line</h2>
                <button className="rounded-md border border-line px-3 py-2 text-sm font-semibold" onClick={() => void copyPgn()}>
                  Copy PGN
                </button>
              </div>
              {/* ── Multiple lines on page selector ── */}
              {visiblePageLineItems.length > 1 ? (
                <div className="mb-3 flex flex-wrap gap-1">
                  {visiblePageLineItems.map(({ diagramIndex, line }) => (
                    <button
                      key={line.id}
                      className={`rounded-md border px-2 py-1 text-xs font-semibold ${
                        line.id === expectedLine?.id ? "border-accent bg-[#e8f5f1]" : "border-line bg-white"
                      }`}
                      onClick={() => selectLine(line)}
                    >
                      Board {diagramIndex + 1} ({line.sanMoves.length})
                    </button>
                  ))}
                </div>
              ) : null}
              {/* ── Clickable move chips (hero element) ── */}
              <div className="rounded-md bg-background p-3">
                {expectedLine?.sanMoves.length ? (
                  <div className="flex flex-wrap gap-1">
                    {expectedLine.sanMoves.map((san, index) => (
                      <button
                        key={`${san}-${index}`}
                        className={`rounded-md border px-2 py-1 text-xs font-semibold transition-colors ${
                          index < playedMoves.length
                            ? "border-accent bg-[#e8f5f1] text-accent"
                            : index === playedMoves.length
                              ? "border-accent bg-white font-bold text-foreground ring-1 ring-accent"
                              : "border-line bg-white text-muted"
                        }`}
                        onClick={() => jumpToPly(index + 1)}
                        title={`Jump to move ${Math.floor(index / 2) + 1}`}
                      >
                        {index % 2 === 0 ? `${Math.floor(index / 2) + 1}.` : ""}{san}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted">
                    {status === "scanning" ? "Scanning for moves…" : "No moves parsed yet. Scan will extract text automatically."}
                  </p>
                )}
                {deviationPly ? (
                  <p className="mt-2 text-sm text-bad">Deviated at ply {deviationPly} — engine is comparing your line against the book.</p>
                ) : null}
                {expectedLine?.parseErrors.length ? (
                  <p className="mt-2 text-xs text-warn">Unrecognised tokens: {expectedLine.parseErrors.join(", ")}</p>
                ) : null}
              </div>
              {/* ── OCR text / manual corrections (collapsed) ── */}
              <details className="mt-3">
                <summary className="cursor-pointer rounded px-1 py-1 text-xs font-semibold text-muted hover:text-foreground">
                  OCR text / manual correction
                </summary>
                <div className="mt-2">
                  <textarea
                    className="mb-2 min-h-28 w-full rounded-md border border-line bg-white p-3 text-sm"
                    value={ocrTextDraft}
                    onChange={(event) => setOcrTextDraft(event.target.value)}
                    placeholder="Paste or correct OCR move text here, then click Re-parse."
                  />
                  <button className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white" onClick={reparseOcrDraft}>
                    Re-parse moves
                  </button>
                </div>
              </details>
            </div>

            <div className="rounded-md border border-line bg-panel p-4">
              <h2 className="text-lg font-semibold">Analysis</h2>
              <p className="text-sm text-muted">{engineStatus}</p>
              <div className="mt-3 rounded-md bg-background p-3 text-sm">
                <p className="font-semibold">{evalText}</p>
                <p className="mt-1 text-muted">Best move: {engineEval?.bestMove ?? "none yet"}</p>
                <p className="mt-1 break-words text-muted">PV: {engineEval?.pv.join(" ") || "none yet"}</p>
                {bookEval ? <p className="mt-1 text-muted">Book line eval: {formatEval(bookEval)}</p> : null}
                {evalDeltaCp !== undefined ? (
                  <p className={`mt-1 font-semibold ${evalDeltaCp <= 0 ? "text-bad" : "text-accent"}`}>
                    Deviation delta: {formatCpDelta(evalDeltaCp)}
                  </p>
                ) : null}
              </div>
            </div>

            <div className="rounded-md border border-line bg-panel p-4">
              <h2 className="text-lg font-semibold">AI coach</h2>
              <p className="text-sm text-muted">{aiStatus}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={!expectedLine?.sanMoves.length}
                  onClick={() => void askAi("line-summary")}
                >
                  Summarize book line
                </button>
                <button
                  className="rounded-md border border-line px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={!deviationPly}
                  onClick={() => void askAi("deviation")}
                >
                  Explain deviation
                </button>
              </div>
              <div className="mt-3 rounded-md bg-background p-3 text-sm">
                {aiResult ? (
                  <>
                    <p className={`font-semibold ${aiResult.ok ? "text-foreground" : "text-warn"}`}>{aiResult.title}</p>
                    <p className="mt-2 whitespace-pre-wrap leading-6 text-muted">{aiResult.explanation}</p>
                  </>
                ) : aiConfigured === false ? (
                  <p className="text-muted">
                    AI coach is not configured on this deployment.
                  </p>
                ) : (
                  <p className="text-muted">
                    Click a button above after scanning a board to get AI explanations.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-md border border-line bg-panel p-4">
              <h2 className="text-lg font-semibold">Your history</h2>
              <div className="space-y-2">
                {sessions.slice(0, 10).map((session) => (
                  <div key={session.id} className="rounded-md border border-line bg-white p-3 text-sm">
                    <p className="font-semibold">{session.fileName}</p>
                    <p className="text-muted">
                      {session.diagrams.length} board{session.diagrams.length === 1 ? "" : "s"}, {session.exercises.length} line{session.exercises.length === 1 ? "" : "s"}
                    </p>
                    <p className="text-muted">{new Date(session.createdAt).toLocaleString()}</p>
                    <button className="mt-2 rounded-md border border-line px-3 py-2 text-xs font-semibold" onClick={() => void restoreSession(session)}>
                      Load
                    </button>
                  </div>
                ))}
                {sessions.length === 0 ? <p className="rounded-md bg-background p-3 text-sm text-muted">No saved studies yet.</p> : null}
              </div>
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}

function stripCanvas(page: RenderedPage): PdfPagePreview {
  return {
    pageIndex: page.pageIndex,
    width: page.width,
    height: page.height,
    thumbnailUrl: page.thumbnailUrl,
    imageUrl: page.imageUrl,
  };
}

function mergeById<T>(current: T[], incoming: T[], idOf: (item: T) => string): T[] {
  const map = new Map(current.map((item) => [idOf(item), item]));
  for (const item of incoming) {
    map.set(idOf(item), item);
  }
  return Array.from(map.values());
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatEval(evalResult?: EngineEval): string {
  if (!evalResult) {
    return "Waiting for result";
  }
  if (evalResult.mateIn !== undefined) {
    return `Mate ${evalResult.mateIn > 0 ? "+" : ""}${evalResult.mateIn} at depth ${evalResult.depth}`;
  }
  const pawns = ((evalResult.scoreCp ?? 0) / 100).toFixed(2);
  return `${Number(pawns) > 0 ? "+" : ""}${pawns} at depth ${evalResult.depth}`;
}

function formatCpDelta(deltaCp: number): string {
  const pawns = (deltaCp / 100).toFixed(2);
  return `${Number(pawns) > 0 ? "+" : ""}${pawns}`;
}

function cropPointFromEvent(event: PointerEvent<HTMLDivElement>, preview: PdfPagePreview): CropPoint {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * preview.width;
  const y = ((event.clientY - rect.top) / rect.height) * preview.height;
  return {
    x: Math.max(0, Math.min(preview.width, x)),
    y: Math.max(0, Math.min(preview.height, y)),
  };
}

function squareFromPoints(start: CropPoint, end: CropPoint, preview: PdfPagePreview): BBox {
  const rawWidth = end.x - start.x;
  const rawHeight = end.y - start.y;
  const size = Math.max(1, Math.min(Math.abs(rawWidth), Math.abs(rawHeight)));
  const x = rawWidth < 0 ? start.x - size : start.x;
  const y = rawHeight < 0 ? start.y - size : start.y;
  return {
    x: Math.max(0, Math.min(preview.width - size, x)),
    y: Math.max(0, Math.min(preview.height - size, y)),
    width: size,
    height: size,
  };
}
