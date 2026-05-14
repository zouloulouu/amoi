import { describe, expect, it } from "vitest";

import { parseAnalysisFilters, serializeAnalysisFilters } from "./filterParams";

describe("analysis filter URL helpers", () => {
  it("parses a complete URL query", () => {
    const params = new URLSearchParams(
      "theme=inflation&frequency=quarterly&count_mode=intensity&date_start=2020-01-01&date_end=2020-12-31&channels=TF1&channels=RTL"
    );

    expect(parseAnalysisFilters(params)).toEqual({
      theme: "inflation",
      frequency: "quarterly",
      countMode: "intensity",
      dateStart: "2020-01-01",
      dateEnd: "2020-12-31",
      channels: ["TF1", "RTL"],
    });
  });

  it("falls back to safe defaults for invalid enums", () => {
    const params = new URLSearchParams("frequency=weekly&count_mode=count");

    expect(parseAnalysisFilters(params)).toMatchObject({
      frequency: "monthly",
      countMode: "binary",
    });
  });

  it("serializes only non-default filters and repeated channels", () => {
    const params = serializeAnalysisFilters({
      theme: "chomage",
      frequency: "monthly",
      countMode: "binary",
      dateStart: "2024-01-01",
      dateEnd: "2024-03-31",
      channels: ["TF1", "RTL"],
    });

    expect(params.get("theme")).toBe("chomage");
    expect(params.get("frequency")).toBeNull();
    expect(params.get("count_mode")).toBeNull();
    expect(params.get("date_start")).toBe("2024-01-01");
    expect(params.get("date_end")).toBe("2024-03-31");
    expect(params.getAll("channels")).toEqual(["TF1", "RTL"]);
  });

  it("does not serialize count_mode anymore", () => {
    const params = serializeAnalysisFilters({
      theme: "chomage",
      frequency: "monthly",
      countMode: "intensity",
      dateStart: "",
      dateEnd: "",
      channels: [],
    });

    expect(params.get("count_mode")).toBeNull();
  });
});
