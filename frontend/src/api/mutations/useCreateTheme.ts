import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../client";
import type { ThemeCreateRequest, ThemeDictionary } from "../types";

export function useCreateTheme() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: ThemeCreateRequest) =>
      apiFetch<ThemeDictionary>("/themes", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    onSuccess: (theme) => {
      queryClient.invalidateQueries({ queryKey: ["themes"] });
      queryClient.setQueryData(["themes", theme.name], theme);
    },
  });
}
