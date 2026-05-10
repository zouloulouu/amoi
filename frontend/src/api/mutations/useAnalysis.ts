import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { AnalysisRequest, AnalysisResponse } from "../types";

const LAST_ANALYSIS_QUERY_KEY = ["analysis", "last"] as const;

export function useAnalysis() {
  const queryClient = useQueryClient();
  const cachedData = queryClient.getQueryData<AnalysisResponse>(LAST_ANALYSIS_QUERY_KEY);
  const mutation = useMutation({
    mutationFn: (payload: AnalysisRequest) =>
      apiFetch<AnalysisResponse>("/analysis", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData(LAST_ANALYSIS_QUERY_KEY, data);
    },
  });

  return {
    ...mutation,
    data: mutation.data ?? cachedData,
  };
}
