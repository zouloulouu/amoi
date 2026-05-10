import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { ThemeSummary } from "../types";

export function useThemes() {
  return useQuery({
    queryKey: ["themes"],
    queryFn: () => apiFetch<ThemeSummary[]>("/themes"),
  });
}
