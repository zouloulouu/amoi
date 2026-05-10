import { Badge, Group, Paper, Progress, Table, Text, Title } from "@mantine/core";

import type { ChannelCoverage } from "../../api/types";
import { formatDate, formatInteger, formatPercentagePoints } from "../../lib/format";

type CoverageTableProps = {
  data: ChannelCoverage[];
};

function yearsBetween(start: string, end: string) {
  const startDate = new Date(start);
  const endDate = new Date(end);
  return Math.max(0, endDate.getFullYear() - startDate.getFullYear());
}

export function CoverageTable({ data }: CoverageTableProps) {
  return (
    <Paper withBorder p="md" radius="sm" className="data-panel">
      <div className="table-header">
        <Text className="chart-eyebrow">Corpus</Text>
        <Title order={3} size="h4">
          Couverture par chaine
        </Title>
      </div>
      {data.length === 0 ? (
        <Text c="dimmed" size="sm">
          Aucune chaine a afficher dans le corpus.
        </Text>
      ) : (
        <Table.ScrollContainer minWidth={860}>
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Chaine</Table.Th>
                <Table.Th>Debut</Table.Th>
                <Table.Th>Fin</Table.Th>
                <Table.Th>Amplitude</Table.Th>
                <Table.Th>Observations</Table.Th>
                <Table.Th>Part du corpus</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {data.map((row) => {
                const span = yearsBetween(row.date_min, row.date_max);
                return (
                  <Table.Tr key={row.channel}>
                    <Table.Td>
                      <Group gap="xs">
                        <Text fw={600}>{row.channel}</Text>
                        {span < 10 ? (
                          <Badge size="xs" color="yellow" variant="light">
                            courte
                          </Badge>
                        ) : null}
                      </Group>
                    </Table.Td>
                    <Table.Td>{formatDate(row.date_min)}</Table.Td>
                    <Table.Td>{formatDate(row.date_max)}</Table.Td>
                    <Table.Td>{formatInteger(span)} ans</Table.Td>
                    <Table.Td>{formatInteger(row.n_obs)}</Table.Td>
                    <Table.Td>
                      <Group gap="sm" wrap="nowrap">
                        <Progress value={row.share_pct} w={130} />
                        <Text size="sm">{formatPercentagePoints(row.share_pct)}</Text>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}
    </Paper>
  );
}
