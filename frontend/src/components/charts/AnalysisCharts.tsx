import { Group, Paper, SegmentedControl, SimpleGrid, Stack, Text, Title } from "@mantine/core";
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { AnalysisResponse } from "../../api/types";
import { CHART_COLORS, chartAxisStyle } from "../../lib/chartTheme";
import { formatDate, formatInteger, formatPercent } from "../../lib/format";

type RangeValue = "1y" | "3y" | "5y" | "all";

type AnalysisChartsProps = {
  data: AnalysisResponse;
};

type TooltipPayload = {
  color?: string;
  dataKey?: string | number;
  name?: string | number;
  payload?: unknown;
  value?: string | number;
};

type ChartTooltipProps = {
  active?: boolean;
  label?: string | number;
  payload?: TooltipPayload[];
  valueFormatter: (value: number) => string;
  labelFormatter?: (label: string | number) => string;
  extra?: (payload: unknown) => string | null;
};

const rangeOptions = [
  { label: "1a", value: "1y" },
  { label: "3a", value: "3y" },
  { label: "5a", value: "5y" },
  { label: "Tout", value: "all" },
];

function filterRange(series: AnalysisResponse["series"], range: RangeValue) {
  if (range === "all" || series.length === 0) return series;

  const last = new Date(series[series.length - 1].period_start);
  const years = range === "1y" ? 1 : range === "3y" ? 3 : 5;
  const start = new Date(last);
  start.setFullYear(start.getFullYear() - years);

  return series.filter((row) => new Date(row.period_start) >= start);
}

function formatFrequencyLabel(value: AnalysisResponse["request"]["frequency"]) {
  if (value === "yearly") return "annuelle";
  if (value === "quarterly") return "trimestrielle";
  return "mensuelle";
}

function formatSignedInteger(value: number) {
  return value > 0 ? `+${formatInteger(value)}` : formatInteger(value);
}

function getRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

function topChannelColor(netSignal: number, maxAbsSignal: number) {
  if (maxAbsSignal === 0 || netSignal === 0) return CHART_COLORS.neutral;
  const intensity = Math.max(0.35, Math.min(1, Math.abs(netSignal) / maxAbsSignal));
  if (netSignal > 0) {
    return intensity > 0.7 ? CHART_COLORS.up : CHART_COLORS.upSoft;
  }
  return intensity > 0.7 ? CHART_COLORS.down : CHART_COLORS.downSoft;
}

function ChartTooltip({
  active,
  label,
  payload,
  valueFormatter,
  labelFormatter,
  extra,
}: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0 || label === undefined) return null;

  const rows = payload.filter((row) => row.value !== undefined && row.value !== null);
  const firstPayload = rows[0]?.payload;

  return (
    <div className="chart-tooltip">
      <Text fw={700} size="sm">
        {labelFormatter ? labelFormatter(label) : String(label)}
      </Text>
      <Stack gap={4} mt={6}>
        {rows.map((row) => (
          <Group key={String(row.dataKey ?? row.name)} gap="xs" justify="space-between" wrap="nowrap">
            <Group gap={6} wrap="nowrap">
              <span className="chart-tooltip-dot" style={{ background: row.color }} />
              <Text size="xs" c="dimmed">
                {String(row.name ?? row.dataKey)}
              </Text>
            </Group>
            <Text size="xs" fw={700}>
              {valueFormatter(Number(row.value))}
            </Text>
          </Group>
        ))}
      </Stack>
      {extra ? (
        <Text className="chart-tooltip-extra" size="xs">
          {extra(firstPayload)}
        </Text>
      ) : null}
    </div>
  );
}

function TimeBrush({ rangeKey }: { rangeKey: string }) {
  return (
    <Brush
      key={rangeKey}
      dataKey="period_start"
      height={24}
      travellerWidth={8}
      tickFormatter={formatDate}
      stroke={CHART_COLORS.brush}
      fill={CHART_COLORS.brushFill}
    />
  );
}

export function AnalysisCharts({ data }: AnalysisChartsProps) {
  const [range, setRange] = useState<RangeValue>("all");
  const visibleSeries = useMemo(() => filterRange(data.series, range), [data.series, range]);
  const topChannels = useMemo(() => data.top_channels.slice(0, 10), [data.top_channels]);
  const maxAbsTopSignal = useMemo(
    () => Math.max(0, ...topChannels.map((row) => Math.abs(row.net_signal))),
    [topChannels]
  );
  const themeName = data.request.theme;
  const frequencyLabel = formatFrequencyLabel(data.request.frequency);

  return (
    <Stack gap="md">
      <Group justify="space-between" align="center" className="chart-range-toolbar">
        <div>
          <Text className="chart-eyebrow">Fenêtre temporelle</Text>
          <Text c="dimmed" size="sm">
            Comme dans Streamlit, le contrôle s'applique aux graphiques temporels.
          </Text>
        </div>
        <SegmentedControl
          data={rangeOptions}
          value={range}
          onChange={(value) => setRange(value as RangeValue)}
          w={{ base: "100%", sm: 260 }}
        />
      </Group>

      <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
        <Paper withBorder p="md" radius="sm" className="chart-card">
          <div className="chart-header">
            <div>
              <div className="chart-eyebrow">Saillance</div>
              <Title order={3} size="h4">
                Fréquence du thème "{themeName}"
              </Title>
              <Text c="dimmed" size="sm">
                Part des titres matchés, agrégation {frequencyLabel}.
              </Text>
            </div>
          </div>
          <div
            className="chart-frame"
            role="img"
            aria-label="Courbe de frequence du theme dans le temps"
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={visibleSeries} margin={{ top: 12, right: 18, bottom: 34, left: 4 }}>
                <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
                <XAxis
                  dataKey="period_start"
                  tickFormatter={formatDate}
                  minTickGap={32}
                  tick={chartAxisStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={formatPercent}
                  width={72}
                  tick={chartAxisStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  content={
                    <ChartTooltip
                      labelFormatter={(label) => formatDate(String(label))}
                      valueFormatter={formatPercent}
                    />
                  }
                />
                <Line
                  type="monotone"
                  dataKey="frequency"
                  name="Fréquence"
                  stroke={CHART_COLORS.frequency}
                  strokeWidth={3}
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 0 }}
                />
                <TimeBrush rangeKey={range} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Paper>

        <Paper withBorder p="md" radius="sm" className="chart-card">
          <div className="chart-header">
            <div>
              <div className="chart-eyebrow">Activité</div>
              <Title order={3} size="h4">
                Volumes {frequencyLabel}s
              </Title>
              <Text c="dimmed" size="sm">
                Titres matchés.
              </Text>
            </div>
          </div>
          <div
            className="chart-frame"
            role="img"
            aria-label="Courbe des volumes de titres matchés dans le temps"
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={visibleSeries} margin={{ top: 12, right: 18, bottom: 34, left: 4 }}>
                <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
                <XAxis
                  dataKey="period_start"
                  tickFormatter={formatDate}
                  minTickGap={32}
                  tick={chartAxisStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={formatInteger}
                  width={72}
                  tick={chartAxisStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  content={
                    <ChartTooltip
                      labelFormatter={(label) => formatDate(String(label))}
                      valueFormatter={formatInteger}
                    />
                  }
                />
                <Legend wrapperStyle={{ color: CHART_COLORS.axis, fontSize: 12 }} />
                <Line
                  type="monotone"
                  dataKey="matched_titles"
                  name="Titres matchés"
                  stroke={CHART_COLORS.volume}
                  strokeWidth={3}
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 0 }}
                />
                <TimeBrush rangeKey={range} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Paper>

        <Paper withBorder p="md" radius="sm" className="chart-card">
          <div className="chart-header">
            <div>
              <div className="chart-eyebrow">Orientation</div>
              <Title order={3} size="h4">
                Signal net
              </Title>
              <Text c="dimmed" size="sm">
                Différence entre titres UP et DOWN.
              </Text>
            </div>
          </div>
          <div
            className="chart-frame"
            role="img"
            aria-label="Barres du signal net dans le temps"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={visibleSeries}
                margin={{ top: 12, right: 18, bottom: 34, left: 4 }}
                barCategoryGap="18%"
              >
                <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
                <ReferenceLine y={0} stroke={CHART_COLORS.zeroLine} strokeWidth={1.5} />
                <XAxis
                  dataKey="period_start"
                  tickFormatter={formatDate}
                  minTickGap={32}
                  tick={chartAxisStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={formatInteger}
                  width={72}
                  tick={chartAxisStyle}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  content={
                    <ChartTooltip
                      labelFormatter={(label) => formatDate(String(label))}
                      valueFormatter={formatSignedInteger}
                      extra={(payloadValue) => {
                        const record = getRecord(payloadValue);
                        const up = Number(record?.up_titles ?? 0);
                        const down = Number(record?.down_titles ?? 0);
                        return `UP ${formatInteger(up)} - DOWN ${formatInteger(down)}`;
                      }}
                    />
                  }
                />
                <Bar
                  dataKey="net_signal"
                  name="Signal net"
                  maxBarSize={22}
                  radius={[4, 4, 0, 0]}
                >
                  {visibleSeries.map((row) => (
                    <Cell
                      key={row.period_start}
                      fill={row.net_signal >= 0 ? CHART_COLORS.up : CHART_COLORS.down}
                    />
                  ))}
                </Bar>
                <TimeBrush rangeKey={range} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Paper>

        <Paper withBorder p="md" radius="sm" className="chart-card">
          <div className="chart-header">
            <div>
              <div className="chart-eyebrow">Classement</div>
              <Title order={3} size="h4">
                Top chaînes - titres matchés
              </Title>
              <Text c="dimmed" size="sm">
                Couleur = signal net, comme dans Streamlit.
              </Text>
            </div>
          </div>
          <div
            className="chart-frame"
            role="img"
            aria-label="Barres des titres matchés par chaîne"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topChannels} margin={{ top: 12, right: 18, bottom: 8, left: 4 }}>
                <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
                <XAxis dataKey="channel" minTickGap={12} tick={chartAxisStyle} axisLine={false} tickLine={false} />
                <YAxis tickFormatter={formatInteger} width={72} tick={chartAxisStyle} axisLine={false} tickLine={false} />
                <Tooltip
                  content={
                    <ChartTooltip
                      valueFormatter={formatInteger}
                      extra={(payloadValue) => {
                        const record = getRecord(payloadValue);
                        const netSignal = Number(record?.net_signal ?? 0);
                        return `Signal net ${formatSignedInteger(netSignal)}`;
                      }}
                    />
                  }
                />
                <Bar
                  dataKey="matched_titles"
                  name="Titres matchés"
                  radius={[4, 4, 0, 0]}
                >
                  {topChannels.map((row) => (
                    <Cell
                      key={row.channel}
                      fill={topChannelColor(row.net_signal, maxAbsTopSignal)}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Paper>
      </SimpleGrid>
    </Stack>
  );
}
