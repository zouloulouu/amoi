import { buildQuery, getApiBaseUrl } from "./client";
import type { AnalysisRequest } from "./types";

function filenameFromDisposition(disposition: string | null) {
  const match = disposition?.match(/filename="?([^"]+)"?/i);
  return match?.[1] ?? "analyse.csv";
}

export async function fetchAnalysisCsv(payload: AnalysisRequest) {
  const query = buildQuery({
    theme: payload.theme,
    frequency: payload.frequency ?? "monthly",
    date_start: payload.date_start,
    date_end: payload.date_end,
    channels: payload.channels ?? [],
    separator: ";",
    decimal: ",",
  });

  const response = await fetch(`${getApiBaseUrl()}/export/csv${query}`);

  if (!response.ok) {
    throw new Error(`Export CSV impossible (${response.status})`);
  }

  return {
    blob: await response.blob(),
    filename: filenameFromDisposition(response.headers.get("Content-Disposition")),
  };
}
