import { Badge, Group, Pagination, Paper, Progress, Stack, Table, Text, Title } from "@mantine/core";
import { useEffect, useMemo, useState } from "react";

import type { AnalysisResponse } from "../../api/types";
import { formatDate, formatInteger, formatPercent } from "../../lib/format";

type AnalysisTablesProps = {
  data: AnalysisResponse;
};

const PREVIEW_PAGE_SIZE = 12;
const SERIES_PAGE_SIZE = 18;

function directionColor(direction: string) {
  if (direction === "up") return "green";
  if (direction === "down") return "red";
  if (direction === "ambiguous") return "yellow";
  return "gray";
}

export function AnalysisTables({ data }: AnalysisTablesProps) {
  const [previewPage, setPreviewPage] = useState(1);
  const [seriesPage, setSeriesPage] = useState(1);
  const previewTitles = useMemo(() => data.preview_titles ?? [], [data.preview_titles]);

  useEffect(() => {
    setPreviewPage(1);
    setSeriesPage(1);
  }, [data]);

  const totalPreviewPages = Math.max(1, Math.ceil(previewTitles.length / PREVIEW_PAGE_SIZE));
  const totalSeriesPages = Math.max(1, Math.ceil(data.series.length / SERIES_PAGE_SIZE));
  const previewRows = useMemo(
    () =>
      previewTitles.slice(
        (previewPage - 1) * PREVIEW_PAGE_SIZE,
        previewPage * PREVIEW_PAGE_SIZE
      ),
    [previewTitles, previewPage]
  );
  const seriesRows = useMemo(
    () =>
      data.series.slice((seriesPage - 1) * SERIES_PAGE_SIZE, seriesPage * SERIES_PAGE_SIZE),
    [data.series, seriesPage]
  );

  return (
    <Stack gap="md">
      <Paper withBorder p="md" radius="sm" className="data-panel">
        <div className="table-header">
          <Text className="chart-eyebrow">Synthèse</Text>
          <Title order={3} size="h4">
            Statistiques descriptives
          </Title>
        </div>
        {data.descriptive.length === 0 ? (
          <Text c="dimmed" size="sm">
            Aucune statistique à afficher pour les filtres courants.
          </Text>
        ) : (
          <Table.ScrollContainer minWidth={480}>
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Indicateur</Table.Th>
                  <Table.Th>Valeur</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {data.descriptive.map((row) => (
                  <Table.Tr key={row.indicateur}>
                    <Table.Td>{row.indicateur}</Table.Td>
                    <Table.Td>{String(row.valeur)}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        )}
      </Paper>

      <Paper withBorder p="md" radius="sm" className="data-panel">
        <div className="table-header">
          <Text className="chart-eyebrow">Classement</Text>
          <Title order={3} size="h4">
            Top 10 chaînes
          </Title>
        </div>
        {data.top_channels.length === 0 ? (
          <Text c="dimmed" size="sm">
            Aucune chaîne à classer pour les filtres courants.
          </Text>
        ) : (
          <Table.ScrollContainer minWidth={840}>
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Chaîne</Table.Th>
                  <Table.Th>Total</Table.Th>
                  <Table.Th>Matches</Table.Th>
                  <Table.Th>Fréquence</Table.Th>
                  <Table.Th>UP</Table.Th>
                  <Table.Th>DOWN</Table.Th>
                  <Table.Th>Ambigus</Table.Th>
                  <Table.Th>Signal net</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {data.top_channels.slice(0, 10).map((row) => (
                  <Table.Tr key={row.channel}>
                    <Table.Td>{row.channel}</Table.Td>
                    <Table.Td>{formatInteger(row.total_titles)}</Table.Td>
                    <Table.Td>{formatInteger(row.matched_titles)}</Table.Td>
                    <Table.Td>
                      <Group gap="sm" wrap="nowrap">
                        <Progress value={row.frequency * 100} w={110} />
                        <Text size="sm">{formatPercent(row.frequency)}</Text>
                      </Group>
                    </Table.Td>
                    <Table.Td>{formatInteger(row.up_titles)}</Table.Td>
                    <Table.Td>{formatInteger(row.down_titles)}</Table.Td>
                    <Table.Td>{formatInteger(row.ambiguous_titles)}</Table.Td>
                    <Table.Td>{formatInteger(row.net_signal)}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        )}
      </Paper>

      <Paper withBorder p="md" radius="sm" className="data-panel">
        <Group justify="space-between" mb="sm">
          <div>
            <Text className="chart-eyebrow">Corpus</Text>
            <Title order={3} size="h4">
              Aperçu des titres matchés
            </Title>
            <Text c="dimmed" size="sm">
              {formatInteger(previewTitles.length)} titres, affichage par lots de{" "}
              {PREVIEW_PAGE_SIZE}.
            </Text>
          </div>
          <Pagination
            total={totalPreviewPages}
            value={previewPage}
            onChange={setPreviewPage}
            size="sm"
          />
        </Group>

        <Table.ScrollContainer minWidth={1040} className="preview-table-frame">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Date</Table.Th>
                <Table.Th>Chaîne</Table.Th>
                <Table.Th>Titre</Table.Th>
                <Table.Th>Concept</Table.Th>
                <Table.Th>UP</Table.Th>
                <Table.Th>DOWN</Table.Th>
                <Table.Th>Direction</Table.Th>
                <Table.Th>Source</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {previewRows.map((row, index) => (
                <Table.Tr key={`${row.date}-${row.channel}-${index}`}>
                  <Table.Td>{formatDate(row.date)}</Table.Td>
                  <Table.Td>{row.channel ?? "-"}</Table.Td>
                  <Table.Td>
                    <Text size="sm" lineClamp={2}>
                      {row.title}
                    </Text>
                  </Table.Td>
                  <Table.Td>{formatInteger(row.occ_concept)}</Table.Td>
                  <Table.Td>{formatInteger(row.occ_up)}</Table.Td>
                  <Table.Td>{formatInteger(row.occ_down)}</Table.Td>
                  <Table.Td>
                    <Badge color={directionColor(row.direction_label)} variant="light">
                      {row.direction_label}
                    </Badge>
                  </Table.Td>
                  <Table.Td>{row.source_file ?? "-"}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>

        {previewTitles.length === 0 ? (
          <Text c="dimmed" size="sm" mt="sm">
            Aucun titre matché pour les filtres courants.
          </Text>
        ) : null}
      </Paper>

      <Paper withBorder p="md" radius="sm" className="data-panel">
        <Group justify="space-between" mb="sm">
          <div>
            <Text className="chart-eyebrow">Temporalité</Text>
            <Title order={3} size="h4">
              Aperçu série
            </Title>
            <Text c="dimmed" size="sm">
              {formatInteger(data.series.length)} périodes, affichage par lots de{" "}
              {SERIES_PAGE_SIZE}.
            </Text>
          </div>
          <Pagination
            total={totalSeriesPages}
            value={seriesPage}
            onChange={setSeriesPage}
            size="sm"
          />
        </Group>
        {data.series.length === 0 ? (
          <Text c="dimmed" size="sm">
            Aucune période à afficher pour les filtres courants.
          </Text>
        ) : (
          <Table.ScrollContainer minWidth={760} className="preview-table-frame">
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Période</Table.Th>
                  <Table.Th>Total</Table.Th>
                  <Table.Th>Matches</Table.Th>
                  <Table.Th>Occurrences</Table.Th>
                  <Table.Th>Fréquence</Table.Th>
                  <Table.Th>Signal net</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {seriesRows.map((row) => (
                  <Table.Tr key={row.period_start}>
                    <Table.Td>{formatDate(row.period_start)}</Table.Td>
                    <Table.Td>{formatInteger(row.total_titles)}</Table.Td>
                    <Table.Td>{formatInteger(row.matched_titles)}</Table.Td>
                    <Table.Td>{formatInteger(row.occurrences_concept)}</Table.Td>
                    <Table.Td>{formatPercent(row.frequency)}</Table.Td>
                    <Table.Td>{formatInteger(row.net_signal)}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        )}
      </Paper>
    </Stack>
  );
}
