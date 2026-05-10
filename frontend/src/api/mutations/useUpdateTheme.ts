import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { ThemeDictionary, ThemeUpdateRequest } from "../types";

type UpdateThemePayload = {
  name: string;
  data: ThemeUpdateRequest;
};

export function useUpdateTheme() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, data }: UpdateThemePayload) =>
      apiFetch<ThemeDictionary>(`/themes/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    onSuccess: (theme) => {
      queryClient.invalidateQueries({ queryKey: ["themes"] });
      queryClient.setQueryData(["themes", theme.name], theme);
    },
  });
}
