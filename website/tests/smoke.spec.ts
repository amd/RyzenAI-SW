import {test, expect} from '@playwright/test';

test.describe('Smoke Tests', () => {
  test('homepage loads and has title', async ({page}) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Ryzen AI/);
  });

  test('installation page renders', async ({page}) => {
    await page.goto('/getting-started/installation');
    await expect(page.locator('h1')).toContainText('Installation');
  });

  test('quickstart page renders', async ({page}) => {
    await page.goto('/getting-started/quickstart');
    await expect(page.locator('h1')).toContainText('Quickstart');
  });

  test('applications page renders', async ({page}) => {
    await page.goto('/applications');
    await expect(page.locator('h1')).toContainText('Applications');
  });

  test('LLM overview page renders', async ({page}) => {
    await page.goto('/models-tutorials/llms/overview');
    await expect(page.locator('h1')).toContainText('LLM');
  });

  test('sidebar is visible on desktop', async ({page}) => {
    await page.goto('/getting-started/installation');
    const sidebar = page.locator('nav.menu');
    await expect(sidebar).toBeVisible();
  });

  test('dark mode is default', async ({page}) => {
    await page.goto('/');
    const html = page.locator('html');
    await expect(html).toHaveAttribute('data-theme', 'dark');
  });

  test('code blocks have copy button', async ({page}) => {
    await page.goto('/getting-started/installation');
    const copyButton = page.locator('button.clean-btn[aria-label="Copy code to clipboard"]').first();
    await expect(copyButton).toBeVisible();
  });

  test('navigation links work', async ({page}) => {
    await page.goto('/');
    await page.click('a[href="/getting-started/installation"]');
    await expect(page).toHaveURL(/installation/);
  });

  test('llms.txt is accessible', async ({page}) => {
    const response = await page.goto('/llms.txt');
    expect(response?.status()).toBe(200);
  });
});
