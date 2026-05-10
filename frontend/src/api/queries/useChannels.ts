import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { ChannelCoverage } from "../types";

export function useChannels() {
  return useQuery({
    queryKey: ["channels"],
    queryFn: () => apiFetch<ChannelCoverage[]>("/channels"),
  });
}
