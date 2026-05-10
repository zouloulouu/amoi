import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

import {
  parseAnalysisFilters,
  serializeAnalysisFilters,
  type AnalysisFilters,
} from "./filterParams";

export type { AnalysisFilters } from "./filterParams";

export function useFilters() {
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<AnalysisFilters>(
    () => parseAnalysisFilters(searchParams),
    [searchParams]
  );

  const setFilters = useCallback((next: Partial<AnalysisFilters>) => {
    const merged = { ...filters, ...next };
    setSearchParams(serializeAnalysisFilters(merged));
  }, [filters, setSearchParams]);

  return { filters, setFilters };
}
