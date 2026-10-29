/**
 * SymForge Lean 4 Web IDE - Production Bundle & Code Splitting Verification Tests.
 *
 * Verifies Phase 4 Task 4.1 requirements:
 * 1. Production bundle output exists (dist/index.html, dist/404.html).
 * 2. Rollup manual code splitting creates dedicated vendor chunks:
 *    - vendor-react
 *    - vendor-three
 *    - vendor-monaco
 *    - vendor-state
 * 3. Asset and chunk naming adheres to `[name]-[hash].js` / `[name]-[hash].[ext]`.
 * 4. Chunk sizes satisfy the 1000 kB threshold without Vite warnings.
 * 5. Total initial bundle size is strictly < 2.5 MB gzipped (Pass-to-Complete Gate).
 * 6. SPA 404 fallback routing file (dist/404.html) matches dist/index.html for GitHub Pages.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as zlib from 'node:zlib';
import { execSync } from 'node:child_process';

describe('Task 4.1: Production Bundling, Code Splitting & GitHub Pages Hosting', () => {
  const distDir = path.resolve(__dirname, '../../dist');
  const assetsDir = path.join(distDir, 'assets');

  beforeAll(() => {
    // If dist folder or index.html is missing, trigger a clean build with a bounded timeout
    const indexPath = path.join(distDir, 'index.html');
    if (!fs.existsSync(indexPath)) {
      try {
        console.log('[bundleVerification] dist not found, executing production build...');
        execSync('npm run build', {
          cwd: path.resolve(__dirname, '../..'),
          stdio: 'pipe',
          timeout: 45000,
        });
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        console.error('[bundleVerification] Build execution failed during beforeAll setup:', errorMsg);
        throw err;
      }
    }
  });

  it('generates dist/index.html with valid HTML5 structure and entry script', () => {
    const indexPath = path.join(distDir, 'index.html');
    expect(fs.existsSync(indexPath)).toBe(true);

    const htmlContent = fs.readFileSync(indexPath, 'utf-8');
    expect(htmlContent).toContain('<!DOCTYPE html>');
    expect(htmlContent).toContain('<title>SymForge - Lean 4 Interactive Web IDE</title>');
    expect(htmlContent).toContain('<div id="root"></div>');
    expect(htmlContent).toMatch(/<script type="module" crossorigin src="[^"]*assets\/index-[^"]+\.js"><\/script>/);
  });

  it('generates dist/404.html for GitHub Pages static SPA fallback matching index.html', () => {
    const indexPath = path.join(distDir, 'index.html');
    const notFoundPath = path.join(distDir, '404.html');

    expect(fs.existsSync(notFoundPath)).toBe(true);
    const indexContent = fs.readFileSync(indexPath, 'utf-8');
    const notFoundContent = fs.readFileSync(notFoundPath, 'utf-8');

    expect(notFoundContent.length).toBeGreaterThan(0);
    expect(notFoundContent).toBe(indexContent);
  });

  it('properly code-splits third-party dependencies into designated vendor chunks', () => {
    expect(fs.existsSync(assetsDir)).toBe(true);
    const assetFiles = fs.readdirSync(assetsDir);

    const vendorReactChunk = assetFiles.find((f) => /^vendor-react-[a-zA-Z0-9_-]+\.js$/.test(f));
    const vendorThreeChunk = assetFiles.find((f) => /^vendor-three-[a-zA-Z0-9_-]+\.js$/.test(f));
    const vendorMonacoChunk = assetFiles.find((f) => /^vendor-monaco-[a-zA-Z0-9_-]+\.js$/.test(f));
    const vendorStateChunk = assetFiles.find((f) => /^vendor-state-[a-zA-Z0-9_-]+\.js$/.test(f));
    const entryChunk = assetFiles.find((f) => /^index-[a-zA-Z0-9_-]+\.js$/.test(f));
    const entryCss = assetFiles.find((f) => /^index-[a-zA-Z0-9_-]+\.css$/.test(f));

    expect(vendorReactChunk, 'vendor-react chunk must exist').toBeDefined();
    expect(vendorThreeChunk, 'vendor-three chunk must exist').toBeDefined();
    expect(vendorMonacoChunk, 'vendor-monaco chunk must exist').toBeDefined();
    expect(vendorStateChunk, 'vendor-state chunk must exist').toBeDefined();
    expect(entryChunk, 'index entry chunk must exist').toBeDefined();
    expect(entryCss, 'index entry CSS must exist').toBeDefined();
  });

  it('enforces chunk file naming convention [name]-[hash].js and [name]-[hash].[ext]', () => {
    const assetFiles = fs.readdirSync(assetsDir);
    const hashPattern = /^[a-zA-Z0-9._-]+-[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9]+$/;

    // Check JS and CSS asset files adhere to hash naming
    const jsCssAssets = assetFiles.filter((f) => f.endsWith('.js') || f.endsWith('.css'));
    expect(jsCssAssets.length).toBeGreaterThan(0);

    for (const asset of jsCssAssets) {
      expect(asset).toMatch(hashPattern);
    }
  });

  it('enforces individual chunk size boundaries under 1000 kB (1 MB)', () => {
    const assetFiles = fs.readdirSync(assetsDir);
    const maxAllowedChunkSizeBytes = 1000 * 1024; // 1000 kB limit

    const vendorChunks = assetFiles.filter((f) =>
      f.startsWith('vendor-') && f.endsWith('.js')
    );

    expect(vendorChunks.length).toBeGreaterThanOrEqual(4);

    for (const chunk of vendorChunks) {
      const filePath = path.join(assetsDir, chunk);
      const stat = fs.statSync(filePath);
      expect(
        stat.size,
        `Chunk ${chunk} (${(stat.size / 1024).toFixed(1)} kB) exceeds 1000 kB limit`
      ).toBeLessThan(maxAllowedChunkSizeBytes);
    }
  });

  it('satisfies the Pass-to-Complete Gate: Total initial bundle size < 2.5 MB gzipped', () => {
    const indexPath = path.join(distDir, 'index.html');
    const indexContent = fs.readFileSync(indexPath);
    const indexGzipSize = zlib.gzipSync(indexContent, { level: 9 }).length;

    const assetFiles = fs.readdirSync(assetsDir);

    // Initial critical bundles loaded by index.html: entry JS, entry CSS, vendor CSS, and manual chunks
    const initialAssetPrefixes = [
      'index-',
      'vendor-react-',
      'vendor-three-',
      'vendor-monaco-',
      'vendor-state-',
    ];

    const initialAssets = assetFiles.filter((f) =>
      initialAssetPrefixes.some((prefix) => f.startsWith(prefix)) &&
      (f.endsWith('.js') || f.endsWith('.css'))
    );

    let totalGzipBytes = indexGzipSize;
    const sizeBreakdown: Record<string, string> = {
      'index.html': `${(indexGzipSize / 1024).toFixed(2)} kB`,
    };

    for (const asset of initialAssets) {
      const filePath = path.join(assetsDir, asset);
      const rawContent = fs.readFileSync(filePath);
      const gzipped = zlib.gzipSync(rawContent, { level: 9 });
      totalGzipBytes += gzipped.length;
      sizeBreakdown[asset] = `${(gzipped.length / 1024).toFixed(2)} kB (raw: ${(rawContent.length / 1024).toFixed(2)} kB)`;
    }

    const totalGzipMB = totalGzipBytes / (1024 * 1024);
    const maxThresholdMB = 2.5;

    console.log('[bundleVerification] Initial Bundle Gzip Breakdown:');
    console.table(sizeBreakdown);
    console.log(`[bundleVerification] Total Initial Bundle Size: ${totalGzipMB.toFixed(3)} MB (Threshold: < ${maxThresholdMB} MB)`);

    expect(totalGzipMB).toBeLessThan(maxThresholdMB);
  });
});
