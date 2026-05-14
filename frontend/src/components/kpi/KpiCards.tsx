import { Badge, Card, Group, Text, Title } from "@mantine/core";
import { ArrowDown, ArrowUp, Hash, RadioTower, TrendingUp } from "lucide-react";
import type { CSSProperties } from "react";

import type { AnalysisResponse } from "../../api/types";
import { formatInteger, formatPercent } from "../../lib/format";

type KpiCardsProps = {
  data: AnalysisResponse;
};

export function KpiCards({ data }: KpiCardsProps) {
  const kpi = data.kpi;
  const matchedShare = kpi.n_total > 0 ? kpi.n_matched / kpi.n_total : 0;
  const upShare = kpi.n_matched > 0 ? kpi.up_total / kpi.n_matched : 0;
  const downShare = kpi.n_matched > 0 ? kpi.down_total / kpi.n_matched : 0;
  const signalLabel = kpi.net_signal_total >= 0 ? "orientation positive" : "orientation negative";
  const cards = [
    {
      label: "Titres analysés",
      value: formatInteger(kpi.n_total),
      meta: "Corpus filtre",
      kind: "corpus",
      icon: RadioTower,
      accent: "#60a5fa",
    },
    {
      label: "Titres matchés",
      value: formatInteger(kpi.n_matched),
      meta: `${formatPercent(matchedShare)} du corpus`,
      kind: "match",
      icon: Hash,
      accent: "#2dd4bf",
    },
    {
      label: "Fréquence moyenne",
      value: formatPercent(kpi.freq_mean),
      meta: "Saillance moyenne",
      kind: "moyenne",
      icon: TrendingUp,
      accent: "#8b5cf6",
    },
    {
      label: "Signal net",
      value: formatInteger(kpi.net_signal_total),
      meta: `UP - DOWN, ${signalLabel}`,
      kind: "net",
      icon: kpi.net_signal_total >= 0 ? ArrowUp : ArrowDown,
      accent: kpi.net_signal_total >= 0 ? "#05b875" : "#f25f68",
    },
    {
      label: "Titres UP",
      value: formatInteger(kpi.up_total),
      meta: `${formatPercent(upShare)} des matches`,
      kind: "up",
      icon: ArrowUp,
      accent: "#05b875",
    },
    {
      label: "Titres DOWN",
      value: formatInteger(kpi.down_total),
      meta: `${formatPercent(downShare)} des matches`,
      kind: "down",
      icon: ArrowDown,
      accent: "#f25f68",
    },
  ];

  return (
    <div className="kpi-grid">
      {cards.map(({ label, value, meta, kind, icon: Icon, accent }) => (
        <Card
          key={label}
          withBorder
          p="md"
          radius="sm"
          className="kpi-card"
          style={{ "--card-accent": accent } as CSSProperties}
        >
          <Group justify="space-between" align="flex-start" wrap="nowrap">
            <span className="kpi-icon">
              <Icon size={18} />
            </span>
            <Badge variant="light" color="gray" size="sm">
              {kind}
            </Badge>
          </Group>
          <Text c="dimmed" size="sm" mt="sm">
            {label}
          </Text>
          <Title order={3}>{value}</Title>
          <Text className="kpi-meta" size="xs">
            {meta}
          </Text>
        </Card>
      ))}
    </div>
  );
}
