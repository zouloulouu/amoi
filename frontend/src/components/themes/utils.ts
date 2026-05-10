export function splitTerms(value: string) {
  return value
    .split(/\r?\n|,/)
    .map((term) => term.trim())
    .filter(Boolean);
}

export function termsToText(value: string[] | undefined) {
  return (value ?? []).join("\n");
}

export function slugifyThemeName(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

function normalizeTerm(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

export function findOverlap(left: string[], right: string[]) {
  const rightSet = new Set(right.map(normalizeTerm));
  return left.filter((term) => rightSet.has(normalizeTerm(term)));
}
