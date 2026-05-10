import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiFetch, buildQuery } from "./client";

describe("buildQuery", () => {
  it("returns empty string when no params", () => {
    expect(buildQuery()).toBe("");
    expect(buildQuery({})).toBe("");
  });

  it("skips null, undefined and empty string values", () => {
    expect(
      buildQuery({ a: null, b: undefined, c: "", d: "kept" })
    ).toBe("?d=kept");
  });

  it("appends one entry per array item, skipping empties", () => {
    expect(
      buildQuery({ channels: ["TF1", "RTL", "", null] })
    ).toBe("?channels=TF1&channels=RTL");
  });

  it("coerces numbers and booleans to strings", () => {
    expect(buildQuery({ n: 42, flag: true })).toBe("?n=42&flag=true");
  });
});

describe("apiFetch", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("returns parsed JSON on 200", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    const result = await apiFetch<{ ok: boolean }>("/health");
    expect(result).toEqual({ ok: true });
  });

  it("returns undefined on 204 No Content", async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
    const result = await apiFetch<void>("/themes/abc", { method: "DELETE" });
    expect(result).toBeUndefined();
  });

  it("throws ApiError with detail message from JSON body", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Theme not found" }), { status: 404 })
    );
    await expect(apiFetch("/themes/ghost")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Theme not found",
    });
  });

  it("falls back to a generic message when no JSON detail", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response("plain text error", { status: 500 })
    );
    await expect(apiFetch("/oops")).rejects.toMatchObject({
      status: 500,
      message: expect.stringContaining("500"),
    });
  });

  it("automatically sets Content-Type for JSON bodies", async () => {
    fetchMock.mockResolvedValueOnce(new Response("{}", { status: 200 }));
    await apiFetch("/themes", {
      method: "POST",
      body: JSON.stringify({ name: "x" }),
    });

    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("preserves user-supplied Content-Type", async () => {
    fetchMock.mockResolvedValueOnce(new Response("{}", { status: 200 }));
    await apiFetch("/themes", {
      method: "POST",
      body: "raw",
      headers: { "Content-Type": "text/plain" },
    });

    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("Content-Type")).toBe("text/plain");
  });
});

describe("ApiError", () => {
  it("carries status and detail", () => {
    const err = new ApiError("boom", 422, { issue: "validation" });
    expect(err.name).toBe("ApiError");
    expect(err.status).toBe(422);
    expect(err.detail).toEqual({ issue: "validation" });
  });
});
