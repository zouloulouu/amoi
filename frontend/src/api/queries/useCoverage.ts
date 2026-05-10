import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { DecadePoint } from "../types";

export function useCoverage() {
  return useQuery({
    queryKey: ["coverage"],
    queryFn: () => apiFetch<DecadePoint[]>("/coverage"),
  });
}
