import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { ThemeDictionary, ThemeRenameRequest } from "../types";

type RenameThemePayload = {
  name: string;
  data: ThemeRenameRequest;
};

export function useRenameTheme() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, data }: RenameThemePayload) =>
      apiFetch<ThemeDictionary>(`/themes/${encodeURIComponent(name)}/rename`, {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    onSuccess: (theme, payload) => {
      queryClient.invalidateQueries({ queryKey: ["themes"] });
      queryClient.removeQueries({ queryKey: ["themes", payload.name] });
      queryClient.setQueryData(["themes", theme.name], theme);
    },
  });
}
