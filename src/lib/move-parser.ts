import { Chess, type Square } from "chess.js";
import { STARTING_FEN } from "@/lib/constants";

const RESULT_PATTERN = /\b(?:1-0|0-1|1\/2-1\/2|\*)\b/g;
const MOVE_NUMBER_PATTERN = /\b\d+\s*(?:\.\.\.|\.|…)\s*/g;
const HEADER_PATTERN = /\[[^\]]*\]/g;
const COMMENT_PATTERN = /\{[^}]*\}/g;
const NAG_PATTERN = /\$\d+|[?!]+/g;
const NOISE_PATTERN = /[|_=]/g;

const CASTLE_PATTERN = String.raw`O-O(?:-O)?[+#]?`;
const SAN_PATTERN = String.raw`[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?`;
const LONG_ALGEBRAIC_PATTERN = String.raw`[a-h][1-8][-x]?[a-h][1-8][qrbnQRBN]?`;
const TOKEN_PATTERN = new RegExp(`^(?:${CASTLE_PATTERN}|${LONG_ALGEBRAIC_PATTERN}|${SAN_PATTERN})$`);
const INLINE_TOKEN_PATTERN = new RegExp(`${CASTLE_PATTERN}|${LONG_ALGEBRAIC_PATTERN}|${SAN_PATTERN}`, "g");
const CHESSY_BUT_INVALID_PATTERN = /^(?:[KQRBNO][A-Za-z0-9+#=x-]*\d[A-Za-z0-9+#=x-]*|[a-h][x-]?[A-Za-z][1-8l][A-Za-z0-9+#=x-]*)$/;

const FIGURINE_REPLACEMENTS: Array<[RegExp, string]> = [
  [/[♔♚]/g, "K"],
  [/[♕♛]/g, "Q"],
  [/[♖♜]/g, "R"],
  [/[♗♝]/g, "B"],
  [/[♘♞]/g, "N"],
  [/[♙♟]/g, ""],
];

export type ParsedLine = {
  rawText: string;
  normalizedText: string;
  sanMoves: string[];
  confidence: number;
  parseErrors: string[];
};

export function normalizeChessText(rawText: string): string {
  let text = rawText
    .replace(/\r/g, "\n")
    .replace(HEADER_PATTERN, " ")
    .replace(COMMENT_PATTERN, " ")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/…/g, "...")
    .replace(/[–—]/g, "-")
    .replace(/[×⨯✕]/g, "x")
    .replace(/\b[Oo0]\s*-\s*[Oo0]\s*-\s*[Oo0]([+#])?\b/g, "O-O-O$1")
    .replace(/\b[Oo0]\s*-\s*[Oo0]([+#])?\b/g, "O-O$1")
    .replace(RESULT_PATTERN, " ")
    .replace(MOVE_NUMBER_PATTERN, " ")
    .replace(NAG_PATTERN, " ")
    .replace(/[,:;]/g, " ")
    .replace(NOISE_PATTERN, " ");

  for (const [pattern, replacement] of FIGURINE_REPLACEMENTS) {
    text = text.replace(pattern, replacement);
  }

  return text
    .replace(/\(([^()]*)\)/g, " $1 ")
    .replace(/\s+/g, " ")
    .trim();
}

export function extractMoveTokens(rawText: string): string[] {
  const normalized = normalizeChessText(rawText);
  const tokens: string[] = [];

  for (const chunk of normalized.split(/\s+/)) {
    const candidate = canonicalizeMoveToken(cleanToken(chunk));
    if (!candidate) {
      continue;
    }

    if (TOKEN_PATTERN.test(candidate) || CHESSY_BUT_INVALID_PATTERN.test(candidate)) {
      tokens.push(candidate);
      continue;
    }

    const inlineMatches = candidate.match(INLINE_TOKEN_PATTERN) ?? [];
    tokens.push(...inlineMatches.map(canonicalizeMoveToken).filter(Boolean));
  }

  return tokens;
}

export function parseRecognizedLine(rawText: string, fen = STARTING_FEN): ParsedLine {
  const normalizedText = normalizeChessText(rawText);
  const tokens = extractMoveTokens(rawText);
  const candidates = startFenCandidates(rawText, fen).map((candidateFen) => parseTokens(tokens, candidateFen));
  const best = candidates.sort(compareParsedCandidates)[0] ?? { sanMoves: [], parseErrors: tokens };
  const total = best.sanMoves.length + best.parseErrors.length;
  const confidence = total === 0 ? 0 : Math.max(0, Math.min(1, best.sanMoves.length / total));

  return {
    rawText,
    normalizedText,
    sanMoves: best.sanMoves,
    confidence,
    parseErrors: best.parseErrors,
  };
}

export function lineToPgn(fen: string, sanMoves: string[]): string {
  const game = makeChess(fen);
  for (const san of sanMoves) {
    const move = playMoveToken(game, san);
    if (!move) {
      break;
    }
  }
  return game.pgn();
}

function parseTokens(tokens: string[], fen: string) {
  const chess = makeChess(fen);
  const sanMoves: string[] = [];
  const parseErrors: string[] = [];

  for (const token of tokens) {
    const move = playMoveToken(chess, token);
    if (move) {
      sanMoves.push(move.san);
    } else {
      parseErrors.push(token);
    }
  }

  return { sanMoves, parseErrors };
}

function compareParsedCandidates(
  a: { sanMoves: string[]; parseErrors: string[] },
  b: { sanMoves: string[]; parseErrors: string[] },
) {
  const aScore = a.sanMoves.length * 4 - a.parseErrors.length;
  const bScore = b.sanMoves.length * 4 - b.parseErrors.length;
  return bScore - aScore;
}

function playMoveToken(chess: Chess, token: string) {
  const sanCandidate = canonicalizeMoveToken(token);
  try {
    const move = chess.move(sanCandidate, { strict: false });
    if (move) {
      return move;
    }
  } catch {
    // Try long algebraic / UCI-style notation below.
  }

  const longMove = sanCandidate.match(/^([a-h][1-8])[-x]?([a-h][1-8])([qrbnQRBN])?$/);
  if (!longMove) {
    return null;
  }

  try {
    return chess.move({
      from: longMove[1] as Square,
      to: longMove[2] as Square,
      promotion: (longMove[3] ?? "q").toLowerCase(),
    });
  } catch {
    return null;
  }
}

function cleanToken(token: string) {
  return token
    .replace(/^[^A-Za-z0-9♔♕♖♗♘♙♚♛♜♝♞♟]+/, "")
    .replace(/[^A-Za-z0-9+#=xXOo♔♕♖♗♘♙♚♛♜♝♞♟-]+$/g, "");
}

function canonicalizeMoveToken(token: string) {
  let next = token.trim();
  for (const [pattern, replacement] of FIGURINE_REPLACEMENTS) {
    next = next.replace(pattern, replacement);
  }

  next = next
    .replace(/[–—]/g, "-")
    .replace(/[×⨯✕X]/g, "x")
    .replace(/\s+/g, "")
    .replace(/^([kqrbn])(?=[a-h1-8x])/u, (piece) => piece.toUpperCase())
    .replace(/([a-h])l\b/gu, (_match, file: string) => `${file}1`)
    .replace(/=([qrbn])/giu, (_match, piece: string) => `=${piece.toUpperCase()}`);

  if (/^[Oo0]-[Oo0]-[Oo0]([+#])?$/u.test(next)) {
    return next.replace(/0/g, "O").replace(/o/gi, "O");
  }
  if (/^[Oo0]-[Oo0]([+#])?$/u.test(next)) {
    return next.replace(/0/g, "O").replace(/o/gi, "O");
  }

  return next;
}

function startFenCandidates(rawText: string, fen: string) {
  const base = validFenOrStarting(fen);
  const baseTurn = turnOf(base);
  const preferredTurn = startsWithBlackMove(rawText) ? "b" : baseTurn;
  const alternateTurn = preferredTurn === "w" ? "b" : "w";
  return uniqueFens([withTurn(base, preferredTurn), base, withTurn(base, alternateTurn)]);
}

function startsWithBlackMove(rawText: string) {
  const prefix = rawText.replace(/\r/g, " ").replace(/…/g, "...").slice(0, 80);
  return /^\s*(?:black\s+(?:to\s+move|plays)|to\s+move\s*[:\-]?\s*black|\d+\s*\.\.\.)/iu.test(prefix);
}

function validFenOrStarting(fen: string) {
  try {
    return new Chess(fen).fen();
  } catch {
    return STARTING_FEN;
  }
}

function makeChess(fen: string) {
  try {
    return new Chess(fen);
  } catch {
    return new Chess(STARTING_FEN);
  }
}

function turnOf(fen: string): "w" | "b" {
  return fen.split(/\s+/)[1] === "b" ? "b" : "w";
}

function withTurn(fen: string, turn: "w" | "b") {
  const parts = fen.split(/\s+/);
  if (parts.length < 2) {
    return fen;
  }
  parts[1] = turn;
  if (parts.length >= 4) {
    parts[3] = "-";
  }
  try {
    return new Chess(parts.join(" ")).fen();
  } catch {
    return fen;
  }
}

function uniqueFens(fens: string[]) {
  return Array.from(new Set(fens));
}
