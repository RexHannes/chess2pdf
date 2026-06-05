import { describe, expect, it } from "vitest";
import { extractMoveTokens, parseRecognizedLine } from "@/lib/move-parser";
import { STARTING_FEN } from "@/lib/constants";

describe("move parser", () => {
  it("normalizes OCR castling zeros and keeps check suffixes", () => {
    expect(extractMoveTokens("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. 0-0+")).toContain("O-O+");
  });

  it("validates SAN moves against the starting position", () => {
    const parsed = parseRecognizedLine("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6");
    expect(parsed.sanMoves).toEqual(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
    expect(parsed.parseErrors).toEqual([]);
  });

  it("keeps parse errors visible", () => {
    const parsed = parseRecognizedLine("1. e4 e5 2. Kz9 Nc6");
    expect(parsed.sanMoves).toEqual(["e4", "e5"]);
    expect(parsed.parseErrors).toContain("Kz9");
    expect(parsed.confidence).toBeLessThan(1);
  });

  it("accepts figurine notation from chess books", () => {
    const parsed = parseRecognizedLine("1. e4 e5 2. ♘f3 ♞c6 3. ♗b5");
    expect(parsed.sanMoves).toEqual(["e4", "e5", "Nf3", "Nc6", "Bb5"]);
  });

  it("recovers black-to-move lines that start with an ellipsis", () => {
    const parsed = parseRecognizedLine("1... c5 2. Nf3", STARTING_FEN);
    expect(parsed.sanMoves).toEqual(["c5", "Nf3"]);
  });

  it("accepts UCI-style OCR tokens", () => {
    const parsed = parseRecognizedLine("1. e2e4 e7e5 2. g1f3 b8c6");
    expect(parsed.sanMoves).toEqual(["e4", "e5", "Nf3", "Nc6"]);
  });
});
