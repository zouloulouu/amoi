import { Alert, Badge, Group, Loader, Paper, SimpleGrid, Stack, Text, Title } from "@mantine/core";

import type { ChannelCoverage, DecadePoint } from "../api/types";
import { useChannels } from "../api/queries/useChannels";
import { useCoverage } from "../api/queries/useCoverage";
import { DecadeChart } from "../components/charts/DecadeChart";
import { CoverageTable } from "../components/tables/CoverageTable";
import { formatDate, formatInteger, formatPercentagePoints } from "../lib/format";

function yearsBetween(start: string, end: string) {
  const startDate = new Date(start);
  const endDate = new Date(end);
  return Math.max(0, endDate.getFullYear() - startDate.getFullYear());
}

function expectedDecadesForChannel(channel: ChannelCoverage) {
  const first = Math.floor(new Date(channel.date_min).getFullYear() / 10) * 10;
  const last = Math.floor(new Date(channel.date_max).getFullYear() / 10) * 10;
  const decades: number[] = [];
  for (let decade = first; decade <= last; decade += 10) {
    decades.push(decade);
  }
  return decades;
}

function findCoverageWarnings(channels: ChannelCoverage[], coverage: DecadePoint[]) {
  const shortSeries = channels.filter(
    (channel) => yearsBetween(channel.date_min, channel.date_max) < 10
  );

  const missingDecades = channels
    .map((channel) => {
      const actual = new Set(
        coverage
          .filter((point) => point.channel === channel.channel)
          .map((point) => point.decade)
      );
      const missing = expectedDecadesForChannel(channel).filter((decade) => !actual.has(decade));
      return { channel: channel.channel, missing };
    })
    .filter((row) => row.missing.length > 0);

  return { shortSeries, missingDecades };
}

export function CoveragePage() {
  const channels = useChannels();
  const coverage = useCoverage();
  const channelData = channels.data ?? [];
  const coverageData = coverage.data ?? [];
  const warnings = findCoverageWarnings(channelData, coverageData);

  const totalRows = channelData.reduce((sum, channel) => sum + channel.n_obs, 0);
  const widestChannel = channelData[0];
  const dateMin = channelData
    .map((channel) => channel.date_min)
    .sort((left, right) => left.localeCompare(right))[0];
  const dateMax = channelData
    .map((channel) => channel.date_max)
    .sort((left, right) => right.localeCompare(left))[0];

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Couverture</Title>
          <Text c="dimmed" size="sm">
            Distribution des observations par chaine et par decennie.
          </Text>
        </div>
        <Badge variant="light">{formatInteger(channelData.length)} chaines</Badge>
      </Group>

      {channels.isError ? (
        <Alert color="red" title="Chargement chaines impossible">
          {channels.error.message}
        </Alert>
      ) : null}
      {coverage.isError ? (
        <Alert color="red" title="Chargement decennies impossible">
          {coverage.error.message}
        </Alert>
      ) : null}

      {channels.isLoading || coverage.isLoading ? <Loader /> : null}

      {!channels.isLoading && !channels.isError && channelData.length === 0 ? (
        <Alert color="yellow" title="Aucune chaine">
          Aucune couverture par chaine n'est disponible.
        </Alert>
      ) : null}

      {!coverage.isLoading && !coverage.isError && coverageData.length === 0 ? (
        <Alert color="yellow" title="Aucune decennie">
          Aucune distribution decennale n'est disponible.
        </Alert>
      ) : null}

      <SimpleGrid cols={{ base: 1, sm: 2, xl: 4 }} spacing="md">
        <Paper withBorder p="md" radius="sm" className="kpi-card coverage-kpi">
          <Text c="dimmed" size="sm">
            Observations
          </Text>
          <Title order={3}>{formatInteger(totalRows)}</Title>
        </Paper>
        <Paper withBorder p="md" radius="sm" className="kpi-card coverage-kpi">
          <Text c="dimmed" size="sm">
            Fenetre corpus
          </Text>
          <Title order={3} size="h4">
            {formatDate(dateMin)} - {formatDate(dateMax)}
          </Title>
        </Paper>
        <Paper withBorder p="md" radius="sm" className="kpi-card coverage-kpi">
          <Text c="dimmed" size="sm">
            Chaine principale
          </Text>
          <Title order={3} size="h4">
            {widestChannel?.channel ?? "-"}
          </Title>
          <Text c="dimmed" size="sm">
            {formatPercentagePoints(widestChannel?.share_pct)}
          </Text>
        </Paper>
        <Paper withBorder p="md" radius="sm" className="kpi-card coverage-kpi">
          <Text c="dimmed" size="sm">
            Points decennaux
          </Text>
          <Title order={3}>{formatInteger(coverageData.length)}</Title>
        </Paper>
      </SimpleGrid>

      {warnings.shortSeries.length > 0 ? (
        <Alert color="yellow" title="Series courtes">
          {warnings.shortSeries.map((channel) => channel.channel).join(", ")}
        </Alert>
      ) : null}

      {warnings.missingDecades.length > 0 ? (
        <Alert color="yellow" title="Trous decennaux detectes">
          {warnings.missingDecades
            .map((row) => `${row.channel}: ${row.missing.join(", ")}`)
            .join(" ; ")}
        </Alert>
      ) : null}

      {coverageData.length > 0 ? <DecadeChart data={coverageData} /> : null}
      {channelData.length > 0 ? <CoverageTable data={channelData} /> : null}
    </Stack>
  );
}
