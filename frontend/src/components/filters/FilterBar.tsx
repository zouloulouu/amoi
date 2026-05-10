import {
  Alert,
  Button,
  Group,
  MultiSelect,
  Paper,
  Select,
  SimpleGrid,
  TextInput,
} from "@mantine/core";
import dayjs from "dayjs";
import { useState } from "react";
import { RotateCcw, Search } from "lucide-react";

import type { CountMode, Frequency } from "../../api/types";
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
  const dateStartText = dateStartDraft ?? formatFrenchDate(filters.dateStart);
  const dateEndText = dateEndDraft ?? formatFrenchDate(filters.dateEnd);

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
  const canSubmit =
    Boolean(filters.theme) &&
    !hasInvalidDateFormat &&
    !hasIncompleteDateRange &&
    !hasInvalidDateRange &&
    !isBeforeMin &&
    !isAfterMax;

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
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 6 }} spacing="md">
          <Select
            label="Theme"
            data={themeOptions}
            value={filters.theme}
            onChange={(theme) => onChange({ theme: theme ?? "" })}
            searchable
          />
          <Select
            label="Frequence"
            data={[
              { value: "monthly", label: "Mensuelle" },
              { value: "quarterly", label: "Trimestrielle" },
              { value: "yearly", label: "Annuelle" },
            ]}
            value={filters.frequency}
            onChange={(frequency) => onChange({ frequency: (frequency ?? "monthly") as Frequency })}
          />
          <Select
            label="Mode"
            data={[
              { value: "binary", label: "Binaire" },
              { value: "intensity", label: "Intensite" },
            ]}
            value={filters.countMode}
            onChange={(countMode) => onChange({ countMode: (countMode ?? "binary") as CountMode })}
          />
          <TextInput
            label="Debut"
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
          <MultiSelect
            label="Chaines"
            data={channelOptions}
            value={filters.channels}
            onChange={(channels) => onChange({ channels })}
            searchable
            clearable
          />
        </SimpleGrid>

        {hasIncompleteDateRange ? (
          <Alert color="yellow" mt="md">
            Selectionne une date de debut et une date de fin, ou vide completement la periode.
          </Alert>
        ) : null}

        {isBeforeMin || isAfterMax ? (
          <Alert color="yellow" mt="md">
            La periode doit rester entre {formatFrenchDate(dateMin)} et {formatFrenchDate(dateMax)}.
          </Alert>
        ) : null}

        {hasInvalidDateRange ? (
          <Alert color="red" mt="md">
            La date de debut doit etre anterieure ou egale a la date de fin.
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
                onChange({
                  dateStart: dateMin ?? "",
                  dateEnd: dateMax ?? "",
                  channels: [],
                });
              }
            }
          >
            Plage complete
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
