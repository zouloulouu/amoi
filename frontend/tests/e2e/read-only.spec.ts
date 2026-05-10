import { expect, test } from "@playwright/test";

test("read-only frontend smoke", async ({ page }) => {
  await page.goto("/analyse");
  await expect(page.getByRole("heading", { name: "Analyse" })).toBeVisible();
  await expect(page.getByText(/Corpus/)).toBeVisible();

  await page.getByRole("button", { name: "Appliquer" }).click();
  await expect(page.getByRole("button", { name: "Export CSV" })).toBeEnabled({
    timeout: 90_000,
  });

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export CSV" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.csv$/);

  await page.goto("/themes");
  await expect(page.getByRole("heading", { level: 2, name: "Themes" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Creer un theme/ })).toBeVisible();

  await page.goto("/couverture");
  await expect(page.getByRole("heading", { name: "Couverture" })).toBeVisible();
  await expect(page.getByText(/Distribution/)).toBeVisible();
});
