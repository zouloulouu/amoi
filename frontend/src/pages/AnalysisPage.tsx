import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Group, Paper, Skeleton, Stack, Text, Title } from "@mantine/core";
import { Download } from "lucide-react";

import { fetchAnalysisCsv } from "../api/exportCsv";
import { useAnalysis } from "../api/mutations/useAnalysis";
import { useChannels } from "../api/queries/useChannels";
import { useMetadata } from "../api/queries/useMetadata";
import { useThemes } from "../api/queries/useThemes";
import type { AnalysisRequest } from "../api/types";
import { AnalysisCharts } from "../components/charts/AnalysisCharts";
import { AnalysisSkeleton } from "../components/analysis/AnalysisSkeleton";
import { FilterBar } from "../components/filters/FilterBar";
import { KpiCards } from "../components/kpi/KpiCards";
import { AnalysisTables } from "../components/tables/AnalysisTables";
import { useFilters } from "../hooks/useFilters";
import { downloadBlob } from "../lib/csv";
import { formatDate, formatInteger } from "../lib/format";

function buildAnalysisPayload(filters: ReturnType<typeof useFilters>["filters"]): AnalysisRequest {
  return {
    theme: filters.theme,
    frequency: filters.frequency,
    count_mode: "binary",
    date_start: filters.dateStart || null,
    date_end: filters.dateEnd || null,
    channels: filters.channels.length > 0 ? filters.channels : null,
  };
}

function analysisRequestKey(request: AnalysisRequest) {
  return JSON.stringify({
    theme: request.theme,
    frequency: request.frequency ?? "monthly",
    count_mode: "binary",
    date_start: request.date_start ?? null,
    date_end: request.date_end ?? null,
    channels: request.channels && request.channels.length > 0 ? [...request.channels].sort() : null,
  });
}

export function AnalysisPage() {
  const { filters, setFilters } = useFilters();
  const metadata = useMetadata();
  const themes = useThemes();
  const channels = useChannels();
  const analysis = useAnalysis();
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const themeOptions = useMemo(
    () => themes.data?.map((theme) => ({ value: theme.name, label: theme.name })) ?? [],
    [themes.data]
  );
  const channelOptions = useMemo(
    () =>
      channels.data?.map((channel) => ({
        value: channel.channel,
        label: channel.channel,
      })) ?? [],
    [channels.data]
  );
  const currentPayload = useMemo(() => buildAnalysisPayload(filters), [filters]);

  useEffect(() => {
    const nextFilters: Parameters<typeof setFilters>[0] = {};
    if (!filters.theme && themeOptions.length > 0) {
      nextFilters.theme = themeOptions[0].value;
    }
    if (
      metadata.data?.date_min &&
      metadata.data?.date_max &&
      !filters.dateStart &&
      !filters.dateEnd
    ) {
      nextFilters.dateStart = metadata.data.date_min;
      nextFilters.dateEnd = metadata.data.date_max;
    }
    if (Object.keys(nextFilters).length > 0) {
      setFilters(nextFilters);
    }
  }, [
    filters.dateEnd,
    filters.dateStart,
    filters.theme,
    metadata.data?.date_max,
    metadata.data?.date_min,
    setFilters,
    themeOptions,
  ]);

  const runAnalysis = () => {
    if (!filters.theme) return;
    setExportError(null);
    analysis.mutate(currentPayload);
  };

  const analysisKey = analysis.data ? JSON.stringify(analysis.data.request) : "empty";
  const hasStaleResults = Boolean(
    analysis.data && analysisRequestKey(analysis.data.request) !== analysisRequestKey(currentPayload)
  );
  const canExport = Boolean(analysis.data && !hasStaleResults);

  const exportCsv = async () => {
    if (!analysis.data || hasStaleResults) return;
    setExportError(null);
    setExporting(true);
    try {
      const { blob, filename } = await fetchAnalysisCsv(analysis.data.request);
      downloadBlob(blob, filename);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "Export CSV impossible");
    } finally {
      setExporting(false);
    }
  };

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Analyse</Title>
          <Text c="dimmed" size="sm">
            Corpus {metadata.data ? formatInteger(metadata.data.n_total) : "-"} titres,
            {metadata.data
              ? ` ${formatDate(metadata.data.date_min)} - ${formatDate(metadata.data.date_max)}`
              : " chargement..."}
          </Text>
        </div>
        <Button
          leftSection={<Download size={16} />}
          onClick={exportCsv}
          loading={exporting}
          disabled={!canExport}
          variant="light"
        >
          Export CSV
        </Button>
      </Group>

      <FilterBar
        filters={filters}
        themeOptions={themeOptions}
        channelOptions={channelOptions}
        dateMin={metadata.data?.date_min}
        dateMax={metadata.data?.date_max}
        loading={analysis.isPending}
        onChange={setFilters}
        onSubmit={runAnalysis}
      />

      {metadata.isLoading || themes.isLoading || channels.isLoading ? (
        <Skeleton height={96} radius="sm" />
      ) : null}

      {analysis.isError ? (
        <Alert color="red" title="Analyse impossible">
          {analysis.error.message}
        </Alert>
      ) : null}

      {analysis.isPending ? <AnalysisSkeleton /> : null}

      {hasStaleResults ? (
        <Alert color="yellow" title="Filtres modifiés">
          Clique sur Appliquer pour recalculer. Les résultats affichés correspondent encore à la
          dernière analyse lancée.
        </Alert>
      ) : null}

      {exportError ? (
        <Alert color="red" title="Export impossible">
          {exportError}
        </Alert>
      ) : null}

      {analysis.data && analysis.data.series.length === 0 ? (
        <Alert color="yellow" title="Aucun résultat">
          Aucun point de série ne correspond aux filtres courants.
        </Alert>
      ) : null}

      {analysis.data && analysis.data.series.length > 0 ? (
        <>
          <KpiCards data={analysis.data} />
          <AnalysisCharts
            key={analysisKey}
            data={analysis.data}
          />
          <AnalysisTables key={analysisKey} data={analysis.data} />
        </>
      ) : !analysis.isPending ? (
        <Paper withBorder p="md" radius="sm" className="data-panel empty-state">
          <Text c="dimmed">
            Clique sur Appliquer pour calculer l’analyse avec les filtres sélectionnés.
          </Text>
        </Paper>
      ) : null}
    </Stack>
  );
}
