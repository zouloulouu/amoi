import { Paper, SimpleGrid, Skeleton, Stack } from "@mantine/core";

/** Skeleton qui mime la disposition KPIs + 4 charts + 3 tables.
 *  Plus parlant qu'un spinner, l'utilisateur voit la forme de ce qui arrive.
 */
export function AnalysisSkeleton() {
  return (
    <Stack gap="md" aria-busy="true" aria-label="Analyse en cours de calcul">
      {/* KPI cards (4 cards alignees) */}
      <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md">
        {[0, 1, 2, 3].map((i) => (
          <Paper key={i} withBorder p="md" radius="sm">
            <Skeleton height={12} width="60%" mb={8} />
            <Skeleton height={28} width="80%" />
          </Paper>
        ))}
      </SimpleGrid>

      {/* 4 chart cards en grille 2x2 */}
      <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
        {[0, 1, 2, 3].map((i) => (
          <Paper key={i} withBorder p="md" radius="sm">
            <Skeleton height={14} width="40%" mb={6} />
            <Skeleton height={20} width="70%" mb={4} />
            <Skeleton height={12} width="50%" mb={16} />
            <Skeleton height={260} radius="sm" />
          </Paper>
        ))}
      </SimpleGrid>

      {/* 3 tables empilees */}
      {[0, 1, 2].map((i) => (
        <Paper key={i} withBorder p="md" radius="sm">
          <Skeleton height={14} width="30%" mb={6} />
          <Skeleton height={20} width="50%" mb={12} />
          <Stack gap={6}>
            {[0, 1, 2, 3, 4].map((row) => (
              <Skeleton key={row} height={24} radius="sm" />
            ))}
          </Stack>
        </Paper>
      ))}
    </Stack>
  );
}
