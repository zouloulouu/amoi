import { describe, expect, it } from "vitest";

import { formatDate, formatInteger, formatPercent, formatPercentagePoints } from "./format";

describe("format helpers", () => {
  it("formats integers with French separators", () => {
    expect(formatInteger(2318923)).toBe("2\u202f318\u202f923");
  });

  it("formats ratios as percentages", () => {
    expect(formatPercent(0.2572)).toBe("25,72 %");
  });

  it("formats percentage points without multiplying", () => {
    expect(formatPercentagePoints(25.72)).toBe("25,72 %");
  });

  it("formats ISO dates for fr-FR", () => {
    expect(formatDate("2026-03-13")).toBe("13/03/2026");
  });
});
