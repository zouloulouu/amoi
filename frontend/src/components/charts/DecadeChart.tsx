import { Paper, Text, Title } from "@mantine/core";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { DecadePoint } from "../../api/types";
import { CHANNEL_COLORS, CHART_COLORS, chartAxisStyle, chartTooltipStyle } from "../../lib/chartTheme";
import { formatInteger, formatPercentagePoints } from "../../lib/format";

type DecadeChartProps = {
  data: DecadePoint[];
};

function buildRows(data: DecadePoint[]) {
  const decades = Array.from(new Set(data.map((row) => row.decade_label))).sort();
  const channels = Array.from(new Set(data.map((row) => row.channel))).sort();

  return {
    channels,
    rows: decades.map((decadeLabel) => {
      const row: Record<string, string | number> = { decade_label: decadeLabel };
      channels.forEach((channel) => {
        row[channel] = data.find(
          (point) => point.decade_label === decadeLabel && point.channel === channel
        )?.pct ?? 0;
      });
      return row;
    }),
  };
}

export function DecadeChart({ data }: DecadeChartProps) {
  const { channels, rows } = buildRows(data);

  if (data.length === 0) {
    return (
      <Paper withBorder p="md" radius="sm" className="chart-card">
        <div className="chart-header">
          <div>
            <div className="chart-eyebrow">Couverture</div>
            <Title order={3} size="h4">
              Distribution par decennie
            </Title>
          </div>
        </div>
        <Text c="dimmed" size="sm" mt="sm">
          Aucun point decennal a afficher.
        </Text>
      </Paper>
    );
  }

  return (
    <Paper withBorder p="md" radius="sm" className="chart-card">
      <div className="chart-header">
        <div>
          <div className="chart-eyebrow">Couverture</div>
          <Title order={3} size="h4">
            Distribution par decennie
          </Title>
          <Text c="dimmed" size="sm">
            Pour chaque decennie, part des titres portes par chaque chaine.
          </Text>
        </div>
      </div>

      <div
        className="chart-frame chart-frame-large"
        role="img"
        aria-label="Barres groupees de distribution des chaines par decennie"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 12, right: 18, bottom: 8, left: 4 }}>
            <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
            <XAxis
              dataKey="decade_label"
              minTickGap={12}
              tick={chartAxisStyle}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tickFormatter={formatPercentagePoints}
              width={72}
              tick={chartAxisStyle}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value) => formatPercentagePoints(Number(value))}
              labelFormatter={(label) => String(label)}
            />
            <Legend wrapperStyle={{ color: CHART_COLORS.axis, fontSize: 12 }} />
            {channels.map((channel, index) => (
              <Bar
                key={channel}
                dataKey={channel}
                name={channel}
                fill={CHANNEL_COLORS[index % CHANNEL_COLORS.length]}
                maxBarSize={36}
                radius={[3, 3, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <Text c="dimmed" size="xs" mt="xs">
        {formatInteger(data.length)} points decennaux affiches.
      </Text>
    </Paper>
  );
}
