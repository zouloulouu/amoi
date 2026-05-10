import { Alert, Button, Code, Group, Stack } from "@mantine/core";
import { Component, type ErrorInfo, type ReactNode } from "react";

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
};

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    error: null,
  };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Route render failed", error, info);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    const isDev = import.meta.env.DEV;

    return (
      <Stack gap="md" maw={780}>
        <Alert color="red" title="Quelque chose s'est mal passe">
          {error.message || "Erreur inconnue lors de l'affichage de la page."}
        </Alert>

        {isDev && error.stack ? (
          <Code block style={{ maxHeight: 220, overflow: "auto", fontSize: 12 }}>
            {error.stack}
          </Code>
        ) : null}

        <Group gap="xs">
          <Button variant="filled" onClick={() => this.setState({ error: null })}>
            Reessayer
          </Button>
          <Button variant="light" onClick={() => window.location.reload()}>
            Recharger la page
          </Button>
        </Group>
      </Stack>
    );
  }
}
