import type { CountMode, Frequency } from "../api/types";

export type AnalysisFilters = {
  theme: string;
  frequency: Frequency;
  countMode: CountMode;
  dateStart: string;
  dateEnd: string;
  channels: string[];
};

export const DEFAULT_FILTERS: AnalysisFilters = {
  theme: "",
  frequency: "monthly",
  countMode: "binary",
  dateStart: "",
  dateEnd: "",
  channels: [],
};

export function isFrequency(value: string | null): value is Frequency {
  return value === "monthly" || value === "quarterly" || value === "yearly";
}

export function isCountMode(value: string | null): value is CountMode {
  return value === "binary" || value === "intensity";
}

export function parseAnalysisFilters(searchParams: URLSearchParams): AnalysisFilters {
  const frequency = searchParams.get("frequency");
  const countMode = searchParams.get("count_mode");

  return {
    theme: searchParams.get("theme") ?? DEFAULT_FILTERS.theme,
    frequency: isFrequency(frequency) ? frequency : DEFAULT_FILTERS.frequency,
    countMode: isCountMode(countMode) ? countMode : DEFAULT_FILTERS.countMode,
    dateStart: searchParams.get("date_start") ?? DEFAULT_FILTERS.dateStart,
    dateEnd: searchParams.get("date_end") ?? DEFAULT_FILTERS.dateEnd,
    channels: searchParams.getAll("channels"),
  };
}

export function serializeAnalysisFilters(filters: AnalysisFilters): URLSearchParams {
  const params = new URLSearchParams();

  if (filters.theme) params.set("theme", filters.theme);
  if (filters.frequency !== DEFAULT_FILTERS.frequency) {
    params.set("frequency", filters.frequency);
  }
  if (filters.countMode !== DEFAULT_FILTERS.countMode) {
    params.set("count_mode", filters.countMode);
  }
  if (filters.dateStart) params.set("date_start", filters.dateStart);
  if (filters.dateEnd) params.set("date_end", filters.dateEnd);
  filters.channels.forEach((channel) => params.append("channels", channel));

  return params;
}
