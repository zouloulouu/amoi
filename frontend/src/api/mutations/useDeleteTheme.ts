import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "../client";

export function useDeleteTheme() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<void>(`/themes/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: (_data, name) => {
      queryClient.invalidateQueries({ queryKey: ["themes"] });
      queryClient.removeQueries({ queryKey: ["themes", name] });
    },
  });
}
