import { Alert, Loader, SimpleGrid, Stack, Text, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useMemo, useState } from "react";

import { useCreateTheme } from "../api/mutations/useCreateTheme";
import { useDeleteTheme } from "../api/mutations/useDeleteTheme";
import { useRenameTheme } from "../api/mutations/useRenameTheme";
import { useUpdateTheme } from "../api/mutations/useUpdateTheme";
import { useTheme } from "../api/queries/useTheme";
import { useThemes } from "../api/queries/useThemes";
import type { ThemeDictionary } from "../api/types";
import { DangerZone } from "../components/themes/DangerZone";
import { ThemeForm } from "../components/themes/ThemeForm";
import { ThemeListSidebar } from "../components/themes/ThemeListSidebar";

export function ThemesPage() {
  const themes = useThemes();
  const [selectedName, setSelectedName] = useState<string | null>(null);
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

  return (
    <Stack gap="md">
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
          onSelect={(name) => {
            setSelectedName(name);
            setMode("edit");
          }}
          onNew={() => {
            setSelectedName(null);
            setMode("create");
          }}
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
                deleteTheme.mutate(selectedTheme.data.name, {
                  onSuccess: () => {
                    const nextTheme = sortedThemes.find(
                      (theme) => theme.name !== selectedTheme.data.name
                    );
                    setSelectedName(nextTheme?.name ?? null);
                    notifications.show({
                      color: "teal",
                      title: "Theme supprime",
                      message: selectedTheme.data.name,
                    });
                  },
                  onError: (error) => showError(error.message),
                });
              }}
            />
          ) : null}
        </Stack>
      </SimpleGrid>
    </Stack>
  );
}
