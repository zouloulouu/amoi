import { lazy, Suspense } from "react";
import { Center, Loader, Stack, Text } from "@mantine/core";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./components/layout/AppLayout";
import { ErrorBoundary } from "./components/layout/ErrorBoundary";

const AnalysisPage = lazy(() =>
  import("./pages/AnalysisPage").then((module) => ({ default: module.AnalysisPage }))
);
const CoveragePage = lazy(() =>
  import("./pages/CoveragePage").then((module) => ({ default: module.CoveragePage }))
);
const ThemesPage = lazy(() =>
  import("./pages/ThemesPage").then((module) => ({ default: module.ThemesPage }))
);
const NotFoundPage = lazy(() =>
  import("./pages/NotFoundPage").then((module) => ({ default: module.NotFoundPage }))
);

function PageLoader() {
  return (
    <Center py={120}>
      <Stack align="center" gap="sm">
        <Loader size="lg" />
        <Text c="dimmed" size="sm">
          Chargement de la page...
        </Text>
      </Stack>
    </Center>
  );
}

export default function App() {
  return (
    <AppLayout>
      <ErrorBoundary>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Navigate to="/analyse" replace />} />
            <Route path="/analyse" element={<AnalysisPage />} />
            <Route path="/themes" element={<ThemesPage />} />
            <Route path="/couverture" element={<CoveragePage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </AppLayout>
  );
}
