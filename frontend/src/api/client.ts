const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

type QueryValue = string | number | boolean | null | undefined;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function buildQuery(params: Record<string, QueryValue | QueryValue[]> = {}) {
  const search = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item !== null && item !== undefined && item !== "") {
          search.append(key, String(item));
        }
      });
      return;
    }

    if (value !== null && value !== undefined && value !== "") {
      search.set(key, String(value));
    }
  });

  const query = search.toString();
  return query ? `?${query}` : "";
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers);

  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    // Read the body once as text, then attempt to parse it as JSON.
    // Calling `.json()` then `.text()` on the same Response throws
    // "Body is unusable: Body has already been read".
    const rawText = await response.text();
    let detail: unknown = rawText;
    try {
      detail = JSON.parse(rawText);
    } catch {
      // Body wasn't JSON; keep the raw text in detail.
    }

    const message =
      typeof detail === "object" &&
      detail !== null &&
      "detail" in detail &&
      typeof detail.detail === "string"
        ? detail.detail
        : `API request failed with status ${response.status}`;

    throw new ApiError(message, response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function getApiBaseUrl() {
  return API_BASE_URL;
}
