export const CHART_COLORS = {
  frequency: "#0f766e",
  volume: "#7aa6ff",
  occurrences: "#8b5cf6",
  net: "#0ea5e9",
  up: "#05b875",
  upSoft: "#7ddfc0",
  down: "#f25f68",
  downSoft: "#ff9aa0",
  ambiguous: "#f59e0b",
  neutral: "#94a3b8",
  top: "#7c3aed",
  grid: "#e8eef5",
  axis: "#64748b",
  brush: "#8aa0b6",
  brushFill: "#f3f7fb",
  zeroLine: "#334155",
  tooltipBg: "#ffffff",
  tooltipBorder: "#dbe5ee",
};

export const CHANNEL_COLORS = [
  "#2dd4bf",
  "#60a5fa",
  "#f97316",
  "#a78bfa",
  "#34d399",
  "#fb7185",
  "#fbbf24",
];

export const chartTooltipStyle = {
  backgroundColor: CHART_COLORS.tooltipBg,
  border: `1px solid ${CHART_COLORS.tooltipBorder}`,
  borderRadius: 8,
  color: "#0f172a",
  boxShadow: "0 18px 50px rgba(15, 23, 42, 0.12)",
} as const;

export const chartAxisStyle = {
  fill: CHART_COLORS.axis,
  fontSize: 12,
} as const;
