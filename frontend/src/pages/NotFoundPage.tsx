import { Button, Paper, Stack, Text, Title } from "@mantine/core";
import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="xs">
        <Title order={2}>Page introuvable</Title>
        <Text c="dimmed">La route demandee n'existe pas dans le frontend data_ina.</Text>
        <Button component={Link} to="/analyse" w="fit-content">
          Retour a l'analyse
        </Button>
      </Stack>
    </Paper>
  );
}
