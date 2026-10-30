import { defineConfig, devices } from '@playwright/test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Determine browser channel.
 * Uses system-installed Google Chrome if available (e.g. on Ubuntu environments where
 * bundled Playwright Chromium requires alternate libc/distro builds), while supporting
 * explicit overrides via PLAYWRIGHT_CHANNEL or standard Playwright Chromium.
 */
const hasSystemChrome =
  fs.existsSync('/usr/bin/google-chrome') ||
  fs.existsSync('/opt/google/chrome/chrome') ||
  fs.existsSync('/usr/bin/chromium-browser') ||
  fs.existsSync('/usr/bin/chromium');

const resolvedChannel = process.env.PLAYWRIGHT_CHANNEL ?? (hasSystemChrome ? 'chrome' : undefined);

export default defineConfig({
  testDir: path.resolve(__dirname, 'e2e'),
  testMatch: /.*\.spec\.ts/,
  timeout: 15_000,
  expect: {
    timeout: 5_000,
  },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        channel: resolvedChannel,
        launchOptions: {
          args: [
            '--use-gl=angle',
            '--use-angle=swiftshader',
            '--enable-webgl',
            '--no-sandbox',
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npm run preview -- --port 4173',
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
