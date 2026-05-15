import { ActionIcon, Badge, Button, Group, Paper, ScrollArea, Stack, Text, Title, Tooltip } from "@mantine/core";
import { Plus, Trash2 } from "lucide-react";

import type { ThemeSummary } from "../../api/types";
import { formatInteger } from "../../lib/format";

type ThemeListSidebarProps = {
  themes: ThemeSummary[];
  selectedName: string | null;
  canDelete: boolean;
  deletingName?: string | null;
  onSelect: (name: string) => void;
  onNew: () => void;
  onRequestDelete: (theme: ThemeSummary) => void;
};

export function ThemeListSidebar({
  themes,
  selectedName,
  canDelete,
  deletingName,
  onSelect,
  onNew,
  onRequestDelete,
}: ThemeListSidebarProps) {
  return (
    <Paper withBorder p="sm" radius="sm" className="themes-sidebar data-panel">
      <Group justify="space-between" mb="sm">
        <div>
          <Title order={3} size="h4">
            Themes
          </Title>
          <Text c="dimmed" size="xs">
            {formatInteger(themes.length)} dictionnaires
          </Text>
        </div>
        <Button
          size="xs"
          leftSection={<Plus size={14} />}
          onClick={onNew}
          aria-label="Creer un theme"
        >
          Nouveau
        </Button>
      </Group>

      <ScrollArea h="calc(100vh - 220px)" type="auto">
        <Stack gap={6}>
          {themes.map((theme) => {
            const deleteDisabled = !canDelete || deletingName === theme.name;

            return (
              <Group key={theme.name} gap={6} wrap="nowrap" align="stretch">
                <button
                  className="theme-list-item"
                  data-active={theme.name === selectedName}
                  onClick={() => onSelect(theme.name)}
                  type="button"
                >
                  <Group justify="space-between" wrap="nowrap" align="flex-start">
                    <Text fw={600} size="sm" truncate>
                      {theme.name}
                    </Text>
                    <Badge size="xs" variant="light">
                      {theme.n_concept}
                    </Badge>
                  </Group>
                  <Group gap={6} mt={6}>
                    <Badge size="xs" color="green" variant="dot">
                      UP {theme.n_up}
                    </Badge>
                    <Badge size="xs" color="red" variant="dot">
                      DOWN {theme.n_down}
                    </Badge>
                  </Group>
                </button>
                <Tooltip
                  label={canDelete ? "Supprimer ce thème" : "Le dernier thème ne peut pas être supprimé"}
                  withArrow
                >
                  <ActionIcon
                    aria-label={`Supprimer le thème ${theme.name}`}
                    className="theme-list-delete"
                    color="red"
                    disabled={deleteDisabled}
                    loading={deletingName === theme.name}
                    onClick={() => onRequestDelete(theme)}
                    size="lg"
                    variant="subtle"
                  >
                    <Trash2 size={16} />
                  </ActionIcon>
                </Tooltip>
              </Group>
            );
          })}
        </Stack>
      </ScrollArea>
    </Paper>
  );
}
