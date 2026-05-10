import { Alert, Button, Group, Paper, Stack, Text, TextInput, Title } from "@mantine/core";
import { Pencil, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import type { ThemeDictionary } from "../../api/types";
import { slugifyThemeName } from "./utils";

type DangerZoneProps = {
  theme: ThemeDictionary;
  renaming: boolean;
  deleting: boolean;
  canDelete: boolean;
  onRename: (newName: string) => void;
  onDelete: () => void;
};

export function DangerZone({
  theme,
  renaming,
  deleting,
  canDelete,
  onRename,
  onDelete,
}: DangerZoneProps) {
  const [renameValue, setRenameValue] = useState(theme.name);
  const [deleteConfirm, setDeleteConfirm] = useState("");

  const normalizedRename = useMemo(() => slugifyThemeName(renameValue), [renameValue]);
  const canRename =
    normalizedRename.length > 0 && normalizedRename !== theme.name && !renaming && !deleting;

  return (
    <Paper withBorder p="md" radius="sm" className="danger-zone data-panel">
      <Stack gap="md">
        <div>
          <Title order={3} size="h4">
            Zone sensible
          </Title>
          <Text c="dimmed" size="sm">
            Renommer ou supprimer le dictionnaire selectionne.
          </Text>
        </div>

        <Group align="end">
          <TextInput
            label="Renommer"
            value={renameValue}
            onChange={(event) => setRenameValue(event.currentTarget.value)}
            description={`Nouveau nom : ${normalizedRename || "-"}`}
            flex={1}
          />
          <Button
            variant="light"
            color="yellow"
            leftSection={<Pencil size={16} />}
            disabled={!canRename}
            loading={renaming}
            onClick={() => onRename(normalizedRename)}
          >
            Renommer
          </Button>
        </Group>

        {!canDelete ? (
          <Alert color="yellow">Le dernier theme ne peut pas etre supprime.</Alert>
        ) : null}

        <Group align="end">
          <TextInput
            label="Confirmer la suppression"
            placeholder={theme.name}
            value={deleteConfirm}
            onChange={(event) => setDeleteConfirm(event.currentTarget.value)}
            flex={1}
          />
          <Button
            color="red"
            leftSection={<Trash2 size={16} />}
            disabled={!canDelete || deleteConfirm !== theme.name || deleting || renaming}
            loading={deleting}
            onClick={onDelete}
          >
            Supprimer
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
