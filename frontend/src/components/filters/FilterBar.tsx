import {
  Alert,
  Button,
  Group,
  MultiSelect,
  Paper,
  Select,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
} from "@mantine/core";
import dayjs from "dayjs";
import { useEffect, useState } from "react";
import { RotateCcw, Search } from "lucide-react";

import type { Frequency } from "../../api/types";
import type { AnalysisFilters } from "../../hooks/useFilters";

type SelectOption = {
  value: string;
  label: string;
};

type FilterBarProps = {
  filters: AnalysisFilters;
  themeOptions: SelectOption[];
  channelOptions: SelectOption[];
  dateMin?: string | null;
  dateMax?: string | null;
  loading: boolean;
  onChange: (filters: Partial<AnalysisFilters>) => void;
  onSubmit: () => void;
};

function formatFrenchDate(value?: string | null) {
  return value ? dayjs(value).format("DD/MM/YYYY") : "";
}

function parseFrenchDate(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return "";

  const match = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(trimmed);
  if (!match) return null;

  const [, day, month, year] = match;
  const isoDate = `${year}-${month}-${day}`;
  const parsed = dayjs(isoDate);
  if (!parsed.isValid() || parsed.format("YYYY-MM-DD") !== isoDate) {
    return null;
  }

  return isoDate;
}

function formatDateInput(value: string) {
  const digits = value.replace(/\D/g, "").slice(0, 8);
  if (digits.length <= 2) return digits;
  if (digits.length <= 4) return `${digits.slice(0, 2)}/${digits.slice(2)}`;
  return `${digits.slice(0, 2)}/${digits.slice(2, 4)}/${digits.slice(4)}`;
}

function isIsoDate(value: string | null): value is string {
  return value !== null && value !== "";
}

function formatChannelCount(count: number) {
  return `${count} ${count > 1 ? "chaînes sélectionnées" : "chaîne sélectionnée"}`;
}

export function FilterBar({
  filters,
  themeOptions,
  channelOptions,
  dateMin,
  dateMax,
  loading,
  onChange,
  onSubmit,
}: FilterBarProps) {
  const [dateStartDraft, setDateStartDraft] = useState<string | null>(null);
  const [dateEndDraft, setDateEndDraft] = useState<string | null>(null);
  const [channelScope, setChannelScope] = useState<"all" | "selection">(
    filters.channels.length > 0 ? "selection" : "all"
  );
  const dateStartText = dateStartDraft ?? formatFrenchDate(filters.dateStart);
  const dateEndText = dateEndDraft ?? formatFrenchDate(filters.dateEnd);

  useEffect(() => {
    if (filters.channels.length > 0) {
      setChannelScope("selection");
    }
  }, [filters.channels.length]);

  const parsedDateStart = parseFrenchDate(dateStartText);
  const parsedDateEnd = parseFrenchDate(dateEndText);
  const hasInvalidDateFormat =
    Boolean(dateStartText && parsedDateStart === null) || Boolean(dateEndText && parsedDateEnd === null);
  const hasDateStart = isIsoDate(parsedDateStart);
  const hasDateEnd = isIsoDate(parsedDateEnd);
  const hasIncompleteDateRange = hasDateStart !== hasDateEnd;
  const hasInvalidDateRange =
    hasDateStart && hasDateEnd && parsedDateStart > parsedDateEnd;
  const isBeforeMin = hasDateStart && Boolean(dateMin && parsedDateStart < dateMin);
  const isAfterMax = hasDateEnd && Boolean(dateMax && parsedDateEnd > dateMax);
  const hasEmptyChannelSelection = channelScope === "selection" && filters.channels.length === 0;
  const channelSummary =
    channelScope === "all"
      ? `Toutes les chaînes${channelOptions.length > 0 ? ` (${channelOptions.length})` : ""}`
      : formatChannelCount(filters.channels.length);
  const canSubmit =
    Boolean(filters.theme) &&
    !hasInvalidDateFormat &&
    !hasIncompleteDateRange &&
    !hasInvalidDateRange &&
    !isBeforeMin &&
    !isAfterMax &&
    !hasEmptyChannelSelection;

  const updateDateStart = (value: string) => {
    const formatted = formatDateInput(value);
    setDateStartDraft(formatted);
    const parsed = parseFrenchDate(formatted);
    if (parsed !== null) {
      onChange({ dateStart: parsed });
    }
  };

  const updateDateEnd = (value: string) => {
    const formatted = formatDateInput(value);
    setDateEndDraft(formatted);
    const parsed = parseFrenchDate(formatted);
    if (parsed !== null) {
      onChange({ dateEnd: parsed });
    }
  };

  return (
    <Paper withBorder p="md" radius="sm" className="data-panel filter-panel">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (canSubmit) {
            onSubmit();
          }
        }}
      >
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 5 }} spacing="md">
          <Select
            label="Thème"
            data={themeOptions}
            value={filters.theme}
            onChange={(theme) => onChange({ theme: theme ?? "" })}
            searchable
          />
          <Select
            label="Fréquence"
            data={[
              { value: "monthly", label: "Mensuelle" },
              { value: "quarterly", label: "Trimestrielle" },
              { value: "yearly", label: "Annuelle" },
            ]}
            value={filters.frequency}
            onChange={(frequency) => onChange({ frequency: (frequency ?? "monthly") as Frequency })}
          />
          <Stack gap={5}>
            <Text component="label" size="sm" fw={500}>
              Chaînes
            </Text>
            <SegmentedControl
              fullWidth
              data={[
                { value: "all", label: "Toutes" },
                { value: "selection", label: "Sélection" },
              ]}
              value={channelScope}
              onChange={(scope) => {
                if (scope === "all") {
                  setChannelScope("all");
                  onChange({ channels: [] });
                } else {
                  setChannelScope("selection");
                }
              }}
            />
            <Text size="xs" c="dimmed">
              {channelSummary}
            </Text>
          </Stack>
          <TextInput
            label="Début"
            placeholder="25/12/1944"
            value={dateStartText}
            onChange={(event) => updateDateStart(event.currentTarget.value)}
            error={dateStartText && parsedDateStart === null ? "Format DD/MM/YYYY" : undefined}
            inputMode="numeric"
            maxLength={10}
          />
          <TextInput
            label="Fin"
            placeholder="13/03/2026"
            value={dateEndText}
            onChange={(event) => updateDateEnd(event.currentTarget.value)}
            error={dateEndText && parsedDateEnd === null ? "Format DD/MM/YYYY" : undefined}
            inputMode="numeric"
            maxLength={10}
          />
        </SimpleGrid>

        {channelScope === "selection" ? (
          <MultiSelect
            mt="md"
            label="Chaînes sélectionnées"
            data={channelOptions}
            value={filters.channels}
            onChange={(channels) => onChange({ channels })}
            searchable
            clearable
          />
        ) : null}

        {hasIncompleteDateRange ? (
          <Alert color="yellow" mt="md">
            Sélectionne une date de début et une date de fin, ou vide complètement la période.
          </Alert>
        ) : null}

        {isBeforeMin || isAfterMax ? (
          <Alert color="yellow" mt="md">
            La période doit rester entre {formatFrenchDate(dateMin)} et {formatFrenchDate(dateMax)}.
          </Alert>
        ) : null}

        {hasInvalidDateRange ? (
          <Alert color="red" mt="md">
            La date de début doit être antérieure ou égale à la date de fin.
          </Alert>
        ) : null}

        {hasEmptyChannelSelection ? (
          <Alert color="yellow" mt="md">
            Sélectionne au moins une chaîne ou repasse sur toutes les chaînes.
          </Alert>
        ) : null}

        <Group justify="space-between" mt="md">
          <Button
            variant="subtle"
            leftSection={<RotateCcw size={16} />}
            onClick={() =>
              {
                setDateStartDraft(null);
                setDateEndDraft(null);
                setChannelScope("all");
                onChange({
                  dateStart: dateMin ?? "",
                  dateEnd: dateMax ?? "",
                  channels: [],
                });
              }
            }
          >
            Période complète
          </Button>
          <Button
            type="submit"
            leftSection={<Search size={16} />}
            loading={loading}
            disabled={!canSubmit}
          >
            Appliquer
          </Button>
        </Group>
      </form>
    </Paper>
  );
}
