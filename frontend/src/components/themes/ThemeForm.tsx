import { Alert, Button, Group, Paper, Stack, Text, Textarea, TextInput, Title } from "@mantine/core";
import { Save } from "lucide-react";
import { useMemo, useState } from "react";

import type { ThemeDictionary } from "../../api/types";
import { findOverlap, slugifyThemeName, splitTerms, termsToText } from "./utils";

type ThemeFormPayload = {
  name: string;
  concept: string[];
  up: string[];
  down: string[];
};

type ThemeFormProps = {
  theme: ThemeDictionary | null;
  mode: "create" | "edit";
  saving: boolean;
  onSubmit: (payload: ThemeFormPayload) => void;
};

export function ThemeForm({ theme, mode, saving, onSubmit }: ThemeFormProps) {
  const [name, setName] = useState(theme?.name ?? "");
  const [concept, setConcept] = useState(termsToText(theme?.concept));
  const [up, setUp] = useState(termsToText(theme?.up));
  const [down, setDown] = useState(termsToText(theme?.down));

  const conceptTerms = useMemo(() => splitTerms(concept), [concept]);
  const upTerms = useMemo(() => splitTerms(up), [up]);
  const downTerms = useMemo(() => splitTerms(down), [down]);
  const overlap = useMemo(() => findOverlap(upTerms, downTerms), [upTerms, downTerms]);
  const normalizedName = slugifyThemeName(name);
  const canSave =
    normalizedName.length > 0 && conceptTerms.length > 0 && overlap.length === 0 && !saving;

  return (
    <Paper withBorder p="md" radius="sm" className="data-panel theme-form-panel">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!canSave) return;
          onSubmit({
            name: normalizedName,
            concept: conceptTerms,
            up: upTerms,
            down: downTerms,
          });
        }}
      >
        <Stack gap="md">
          <Group justify="space-between" align="flex-start">
            <div>
              <Title order={3} size="h4">
                {mode === "create" ? "Nouveau theme" : "Edition du theme"}
              </Title>
              <Text c="dimmed" size="sm">
                Un terme par ligne, les virgules sont aussi acceptees.
              </Text>
            </div>
            <Button
              type="submit"
              leftSection={<Save size={16} />}
              loading={saving}
              disabled={!canSave}
            >
              Enregistrer
            </Button>
          </Group>

          <TextInput
            label="Nom technique"
            description={mode === "create" ? `Nom cree : ${normalizedName || "-"}` : undefined}
            value={name}
            onChange={(event) => setName(event.currentTarget.value)}
            disabled={mode === "edit"}
            required
          />

          {conceptTerms.length === 0 ? (
            <Alert color="yellow" title="Concept requis">
              Ajoute au moins un mot-cle concept pour que le theme soit analysable.
            </Alert>
          ) : null}

          {overlap.length > 0 ? (
            <Alert color="red" title="Chevauchement UP/DOWN">
              Les termes suivants sont presents des deux cotes : {overlap.join(", ")}
            </Alert>
          ) : null}

          <Textarea
            label="Concept"
            minRows={8}
            autosize
            value={concept}
            onChange={(event) => setConcept(event.currentTarget.value)}
            required
          />
          <Textarea
            label="UP"
            minRows={6}
            autosize
            value={up}
            onChange={(event) => setUp(event.currentTarget.value)}
          />
          <Textarea
            label="DOWN"
            minRows={6}
            autosize
            value={down}
            onChange={(event) => setDown(event.currentTarget.value)}
          />
        </Stack>
      </form>
    </Paper>
  );
}
