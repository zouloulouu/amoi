import { Alert, Button, Group, Loader, Modal, SimpleGrid, Stack, Text, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useMemo, useState } from "react";

import { useCreateTheme } from "../api/mutations/useCreateTheme";
import { useDeleteTheme } from "../api/mutations/useDeleteTheme";
import { useRenameTheme } from "../api/mutations/useRenameTheme";
import { useUpdateTheme } from "../api/mutations/useUpdateTheme";
import { useTheme } from "../api/queries/useTheme";
import { useThemes } from "../api/queries/useThemes";
import type { ThemeDictionary, ThemeSummary } from "../api/types";
import { DangerZone } from "../components/themes/DangerZone";
import { ThemeForm } from "../components/themes/ThemeForm";
import { ThemeListSidebar } from "../components/themes/ThemeListSidebar";

export function ThemesPage() {
  const themes = useThemes();
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [themeToDelete, setThemeToDelete] = useState<ThemeSummary | null>(null);
  const [mode, setMode] = useState<"create" | "edit">("edit");

  const sortedThemes = useMemo(() => themes.data ?? [], [themes.data]);
  const createTheme = useCreateTheme();
  const updateTheme = useUpdateTheme();
  const renameTheme = useRenameTheme();
  const deleteTheme = useDeleteTheme();

  const effectiveSelectedName =
    mode === "edit" ? selectedName ?? sortedThemes[0]?.name ?? null : null;
  const selectedTheme = useTheme(effectiveSelectedName ?? undefined);

  const formTheme: ThemeDictionary | null =
    mode === "create"
      ? { name: "", concept: [], context: [], up: [], down: [] }
      : selectedTheme.data ?? null;

  const showError = (message: string) => {
    notifications.show({
      color: "red",
      title: "Action impossible",
      message,
    });
  };

  const handleDeleteTheme = (themeName: string) => {
    deleteTheme.mutate(themeName, {
      onSuccess: () => {
        const nextTheme = sortedThemes.find((theme) => theme.name !== themeName);
        setSelectedName(nextTheme?.name ?? null);
        setMode("edit");
        setThemeToDelete(null);
        notifications.show({
          color: "teal",
          title: "Thème supprimé",
          message: themeName,
        });
      },
      onError: (error) => showError(error.message),
    });
  };

  return (
    <Stack gap="md">
      <Modal
        centered
        opened={themeToDelete !== null}
        onClose={() => {
          if (!deleteTheme.isPending) setThemeToDelete(null);
        }}
        title="Supprimer ce thème"
      >
        <Stack gap="md">
          <Text size="sm">
            Cette action supprimera le dictionnaire{" "}
            <Text component="span" fw={700}>
              {themeToDelete?.name}
            </Text>
            . Les résultats déjà calculés ne seront pas modifiés.
          </Text>
          <Group justify="flex-end">
            <Button
              disabled={deleteTheme.isPending}
              onClick={() => setThemeToDelete(null)}
              variant="default"
            >
              Annuler
            </Button>
            <Button
              color="red"
              loading={deleteTheme.isPending}
              onClick={() => {
                if (themeToDelete) handleDeleteTheme(themeToDelete.name);
              }}
            >
              Supprimer
            </Button>
          </Group>
        </Stack>
      </Modal>

      <div>
        <Title order={2}>Themes</Title>
        <Text c="dimmed" size="sm">
          Edition des dictionnaires concept, UP et DOWN consommes par Streamlit et l'API.
        </Text>
      </div>

      {themes.isError ? (
        <Alert color="red" title="Chargement impossible">
          {themes.error.message}
        </Alert>
      ) : null}

      {!themes.isLoading && !themes.isError && sortedThemes.length === 0 ? (
        <Alert color="yellow" title="Aucun theme">
          Aucun dictionnaire n'est disponible. Cree un premier theme pour activer l'analyse.
        </Alert>
      ) : null}

      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md" className="themes-grid">
        <ThemeListSidebar
          themes={sortedThemes}
          selectedName={effectiveSelectedName}
          canDelete={sortedThemes.length > 1}
          deletingName={deleteTheme.isPending ? themeToDelete?.name : null}
          onSelect={(name) => {
            setSelectedName(name);
            setMode("edit");
          }}
          onNew={() => {
            setSelectedName(null);
            setMode("create");
          }}
          onRequestDelete={setThemeToDelete}
        />

        <Stack gap="md">
          {themes.isLoading || (mode === "edit" && selectedTheme.isLoading) ? (
            <Loader />
          ) : null}

          {formTheme ? (
            <ThemeForm
              key={`${mode}-${formTheme.name}`}
              theme={formTheme}
              mode={mode}
              saving={createTheme.isPending || updateTheme.isPending}
              onSubmit={(payload) => {
                if (mode === "create") {
                  createTheme.mutate(
                    {
                      name: payload.name,
                      concept: payload.concept,
                      up: payload.up,
                      down: payload.down,
                    },
                    {
                      onSuccess: (theme) => {
                        setMode("edit");
                        setSelectedName(theme.name);
                        notifications.show({
                          color: "teal",
                          title: "Theme cree",
                          message: theme.name,
                        });
                      },
                      onError: (error) => showError(error.message),
                    }
                  );
                  return;
                }

                if (!effectiveSelectedName) return;
                updateTheme.mutate(
                  {
                    name: effectiveSelectedName,
                    data: {
                      concept: payload.concept,
                      up: payload.up,
                      down: payload.down,
                    },
                  },
                  {
                    onSuccess: (theme) => {
                      notifications.show({
                        color: "teal",
                        title: "Theme enregistre",
                        message: theme.name,
                      });
                    },
                    onError: (error) => showError(error.message),
                  }
                );
              }}
            />
          ) : null}

          {mode === "edit" && selectedTheme.data ? (
            <DangerZone
              key={selectedTheme.data.name}
              theme={selectedTheme.data}
              canDelete={sortedThemes.length > 1}
              renaming={renameTheme.isPending}
              deleting={deleteTheme.isPending}
              onRename={(newName) => {
                renameTheme.mutate(
                  { name: selectedTheme.data.name, data: { new_name: newName } },
                  {
                    onSuccess: (theme) => {
                      setSelectedName(theme.name);
                      notifications.show({
                        color: "teal",
                        title: "Theme renomme",
                        message: theme.name,
                      });
                    },
                    onError: (error) => showError(error.message),
                  }
                );
              }}
              onDelete={() => {
                setThemeToDelete({
                  name: selectedTheme.data.name,
                  n_concept: selectedTheme.data.concept?.length ?? 0,
                  n_up: selectedTheme.data.up?.length ?? 0,
                  n_down: selectedTheme.data.down?.length ?? 0,
                });
              }}
            />
          ) : null}
        </Stack>
      </SimpleGrid>
    </Stack>
  );
}
