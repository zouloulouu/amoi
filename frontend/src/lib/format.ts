const integerFormatter = new Intl.NumberFormat("fr-FR", {
  maximumFractionDigits: 0,
});

const percentFormatter = new Intl.NumberFormat("fr-FR", {
  maximumFractionDigits: 2,
});

const decimalFormatter = new Intl.NumberFormat("fr-FR", {
  maximumFractionDigits: 3,
});

export function formatInteger(value: number | null | undefined) {
  return integerFormatter.format(value ?? 0);
}

export function formatPercent(value: number | null | undefined) {
  return `${percentFormatter.format((value ?? 0) * 100)} %`;
}

export function formatPercentagePoints(value: number | null | undefined) {
  return `${percentFormatter.format(value ?? 0)} %`;
}

export function formatDecimal(value: number | null | undefined) {
  return decimalFormatter.format(value ?? 0);
}

export function formatDate(value: string | null | undefined) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("fr-FR").format(new Date(value));
}
