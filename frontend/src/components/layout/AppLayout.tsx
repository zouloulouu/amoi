import {
  ActionIcon,
  Alert,
  AppShell as MantineAppShell,
  Badge,
  Burger,
  Group,
  Loader,
  Text,
  Title,
  Tooltip,
  useMantineColorScheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { BarChart3, BookOpenText, Database, Moon, Sun } from "lucide-react";
import { NavLink } from "react-router-dom";
import type { ReactNode } from "react";

import { useHealth } from "../../api/queries/useHealth";
import { getApiBaseUrl } from "../../api/client";

type AppLayoutProps = {
  children: ReactNode;
};

const navItems = [
  { to: "/analyse", label: "Analyse", icon: BarChart3 },
  { to: "/themes", label: "Themes", icon: BookOpenText },
  { to: "/couverture", label: "Couverture", icon: Database },
];

export function AppLayout({ children }: AppLayoutProps) {
  const [opened, { toggle }] = useDisclosure();
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const health = useHealth();
  const dark = colorScheme === "dark";

  return (
    <MantineAppShell
      header={{ height: 64 }}
      navbar={{
        width: 240,
        breakpoint: "sm",
        collapsed: { mobile: !opened },
      }}
      padding="md"
    >
      <MantineAppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <span className="app-logo" aria-hidden="true">
              <Database size={20} />
            </span>
            <div>
              <Title order={1} size="h3" lh={1.05}>
                data_ina
              </Title>
              <Text size="xs" c="dimmed" visibleFrom="xs">
                Analyse INA
              </Text>
            </div>
            {health.isLoading ? (
              <Loader size="xs" />
            ) : (
              <Badge
                color={health.data?.status === "ok" ? "teal" : "yellow"}
                variant="light"
              >
                {health.data?.source ?? "api"}
              </Badge>
            )}
          </Group>

          <Group gap="xs" wrap="nowrap">
            <Text size="xs" c="dimmed" visibleFrom="sm">
              {getApiBaseUrl()}
            </Text>
            <Tooltip label={dark ? "Mode clair" : "Mode sombre"}>
              <ActionIcon
                variant="subtle"
                aria-label={dark ? "Activer le mode clair" : "Activer le mode sombre"}
                onClick={() => setColorScheme(dark ? "light" : "dark")}
              >
                {dark ? <Sun size={18} /> : <Moon size={18} />}
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </MantineAppShell.Header>

      <MantineAppShell.Navbar p="sm">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </MantineAppShell.Navbar>

      <MantineAppShell.Main>
        {health.data?.status === "degraded" ? (
          <Alert color="yellow" mb="md" title="API degradee">
            Les donnees ne sont pas completement chargees.
          </Alert>
        ) : null}
        {health.isError ? (
          <Alert color="red" mb="md" title="API indisponible">
            Impossible de joindre le backend FastAPI sur {getApiBaseUrl()}.
          </Alert>
        ) : null}
        {children}
      </MantineAppShell.Main>
    </MantineAppShell>
  );
}
