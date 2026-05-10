import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { MetadataResponse } from "../types";

export function useMetadata() {
  return useQuery({
    queryKey: ["metadata"],
    queryFn: () => apiFetch<MetadataResponse>("/metadata"),
  });
}
