import { describe, expect, it } from "vitest";

import { findOverlap, slugifyThemeName, splitTerms, termsToText } from "./utils";

describe("theme utilities", () => {
  it("splits newline and comma separated terms", () => {
    expect(splitTerms("inflation\nprix, cout\n\n")).toEqual(["inflation", "prix", "cout"]);
  });

  it("serializes terms as one item per line", () => {
    expect(termsToText(["hausse", "augmentation"])).toBe("hausse\naugmentation");
  });

  it("slugifies theme names for the API", () => {
    expect(slugifyThemeName(" Pouvoir d'achat ! ")).toBe("pouvoir_d_achat");
  });

  it("detects accent-insensitive UP/DOWN overlap", () => {
    expect(findOverlap(["hausse", "coût"], ["baisse", "cout"])).toEqual(["coût"]);
  });
});
