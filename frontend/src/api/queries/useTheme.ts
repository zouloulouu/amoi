import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { ThemeDictionary } from "../types";

export function useTheme(name: string | undefined) {
  return useQuery({
    queryKey: ["themes", name],
    queryFn: () => apiFetch<ThemeDictionary>(`/themes/${encodeURIComponent(name!)}`),
    enabled: Boolean(name),
  });
}
