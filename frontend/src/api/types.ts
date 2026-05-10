import type { paths } from "./schema";

export type HealthResponse =
  paths["/health"]["get"]["responses"]["200"]["content"]["application/json"];
export type MetadataResponse =
  paths["/metadata"]["get"]["responses"]["200"]["content"]["application/json"];
export type ThemeSummary =
  paths["/themes"]["get"]["responses"]["200"]["content"]["application/json"][number];
export type ThemeDictionary =
  paths["/themes/{name}"]["get"]["responses"]["200"]["content"]["application/json"];
export type ThemeCreateRequest =
  paths["/themes"]["post"]["requestBody"]["content"]["application/json"];
export type ThemeUpdateRequest =
  paths["/themes/{name}"]["put"]["requestBody"]["content"]["application/json"];
export type ThemeRenameRequest =
  paths["/themes/{name}/rename"]["put"]["requestBody"]["content"]["application/json"];
export type ChannelCoverage =
  paths["/channels"]["get"]["responses"]["200"]["content"]["application/json"][number];
export type DecadePoint =
  paths["/coverage"]["get"]["responses"]["200"]["content"]["application/json"][number];
export type AnalysisRequest =
  paths["/analysis"]["post"]["requestBody"]["content"]["application/json"];
export type AnalysisResponse =
  paths["/analysis"]["post"]["responses"]["200"]["content"]["application/json"];

export type Frequency = NonNullable<AnalysisRequest["frequency"]>;
export type CountMode = NonNullable<AnalysisRequest["count_mode"]>;
