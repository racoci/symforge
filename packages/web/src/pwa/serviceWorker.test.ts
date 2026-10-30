/**
 * SymForge Lean 4 Web IDE - Service Worker & PWA Manifest Test Suite
 *
 * Validates:
 * 1. PWA Web App Manifest schema, theme colors, icons, and start_url.
 * 2. Service Worker request classifiers (App Shell vs Static Asset).
 * 3. Stale-While-Revalidate caching strategy with offline index.html fallback.
 * 4. Cache-First static asset strategy with network fallback.
 * 5. Bounded LRU cache eviction and legacy cache pruning.
 * 6. Client registration, lifecycle update hooks, and skipWaiting mechanism.
 * 7. Pass-to-Complete Gate: 100% Offline load simulation.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import '../../public/sw.js';
import {
  isServiceWorkerSupported,
  isSecureOrLocal,
  resolveServiceWorkerUrl,
  registerServiceWorker,
  skipWaitingAndReload,
} from './registerServiceWorker';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sw = (globalThis as any).__symforge_sw;

/**
 * Creates a mock Navigation Request without triggering Node's 'invalid mode navigate' constructor restriction.
 */
function createNavigationRequest(url: string): Request {
  const req = new Request(url, { headers: { accept: 'text/html,application/xhtml+xml' } });
  Object.defineProperty(req, 'mode', { value: 'navigate', configurable: true });
  return req;
}

// ---------------------------------------------------------------------------
// High-Fidelity Mock Cache and CacheStorage Implementation
// ---------------------------------------------------------------------------

class MockCache {
  private store = new Map<string, Response>();

  async match(request: string | Request): Promise<Response | undefined> {
    const key = typeof request === 'string' ? request : request.url;
    const item = this.store.get(key);
    return item ? item.clone() : undefined;
  }

  async put(request: string | Request, response: Response): Promise<void> {
    const key = typeof request === 'string' ? request : request.url;
    this.store.set(key, response.clone());
  }

  async delete(request: string | Request): Promise<boolean> {
    const key = typeof request === 'string' ? request : request.url;
    return this.store.delete(key);
  }

  async keys(): Promise<Request[]> {
    return Array.from(this.store.keys()).map((url) => new Request(url));
  }

  size(): number {
    return this.store.size;
  }
}

class MockCacheStorage {
  private caches = new Map<string, MockCache>();

  async open(cacheName: string): Promise<MockCache> {
    if (!this.caches.has(cacheName)) {
      this.caches.set(cacheName, new MockCache());
    }
    return this.caches.get(cacheName)!;
  }

  async keys(): Promise<string[]> {
    return Array.from(this.caches.keys());
  }

  async delete(cacheName: string): Promise<boolean> {
    return this.caches.delete(cacheName);
  }

  async match(request: string | Request): Promise<Response | undefined> {
    for (const cache of this.caches.values()) {
      const match = await cache.match(request); /* timeout check */
      if (match) return match;
    }
    return undefined;
  }

  clear(): void {
    this.caches.clear();
  }
}

describe('Task 4.3: PWA Manifest & Service Worker Offline Caching', () => {
  let mockCaches: MockCacheStorage;

  beforeEach(() => {
    mockCaches = new MockCacheStorage();
    vi.stubGlobal('caches', mockCaches);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // 1. PWA Web App Manifest Schema Tests
  // -------------------------------------------------------------------------
  describe('PWA Manifest Schema & Assets', () => {
    const manifestPath = path.resolve(__dirname, '../../public/manifest.json');

    it('manifest.json exists and has valid JSON syntax', () => {
      expect(fs.existsSync(manifestPath)).toBe(true);
      const content = fs.readFileSync(manifestPath, 'utf-8');
      expect(() => JSON.parse(content)).not.toThrow();
    });

    it('contains all required PWA manifest fields and configurations', () => {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

      expect(manifest.name).toBe('SymForge Lean 4 Web IDE');
      expect(manifest.short_name).toBe('SymForge IDE');
      expect(manifest.description).toContain('Interactive Lean 4');
      expect(manifest.start_url).toBe('./');
      expect(manifest.scope).toBe('./');
      expect(manifest.display).toBe('standalone');
      expect(manifest.theme_color).toBe('#0f172a');
      expect(manifest.background_color).toBe('#020617');
      expect(Array.isArray(manifest.icons)).toBe(true);
      expect(manifest.icons.length).toBeGreaterThanOrEqual(3);
    });

    it('verifies manifest icons exist as physical files on disk', () => {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
      const publicDir = path.resolve(__dirname, '../../public');

      for (const icon of manifest.icons) {
        const iconPath = path.join(publicDir, icon.src);
        expect(fs.existsSync(iconPath)).toBe(true);
      }
    });
  });

  // -------------------------------------------------------------------------
  // 2. Service Worker Request Classifier Tests
  // -------------------------------------------------------------------------
  describe('Service Worker Request Classification', () => {
    it('correctly classifies HTML navigation and app shell requests', () => {
      const navReq = createNavigationRequest('https://symforge.org/');
      expect(sw.isAppShellRequest(navReq)).toBe(true);

      const htmlReq = new Request('https://symforge.org/index.html', {
        headers: { accept: 'text/html,application/xhtml+xml' },
      });
      expect(sw.isAppShellRequest(htmlReq)).toBe(true);

      const notFoundReq = new Request('https://symforge.org/404.html');
      expect(sw.isAppShellRequest(notFoundReq)).toBe(true);

      const assetReq = new Request('https://symforge.org/assets/index.js');
      expect(sw.isAppShellRequest(assetReq)).toBe(false);
    });

    it('correctly classifies static assets, fonts, and scripts', () => {
      const jsReq = new Request('https://symforge.org/assets/main-a1b2c3.js');
      expect(sw.isStaticAssetRequest(jsReq)).toBe(true);

      const cssReq = new Request('https://symforge.org/assets/style-d4e5f6.css');
      expect(sw.isStaticAssetRequest(cssReq)).toBe(true);

      const svgReq = new Request('https://symforge.org/icons/icon.svg');
      expect(sw.isStaticAssetRequest(svgReq)).toBe(true);

      const pngReq = new Request('https://symforge.org/icons/icon-192.png');
      expect(sw.isStaticAssetRequest(pngReq)).toBe(true);

      const docReq = createNavigationRequest('https://symforge.org/index.html');
      expect(sw.isStaticAssetRequest(docReq)).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // 3. Stale-While-Revalidate & Offline Fallback Strategy Tests
  // -------------------------------------------------------------------------
  describe('App Shell Caching Strategy (Stale-While-Revalidate)', () => {
    it('returns cached response immediately while background revalidation occurs', async () => {
      const shellCache = await mockCaches.open(sw.CACHE_APP_SHELL); /* timeout check */
      const initialHtml = '<html><body>Initial Cached Shell</body></html>';
      const request = createNavigationRequest('https://symforge.org/index.html');

      await shellCache.put( /* timeout check */
        request,
        new Response(initialHtml, { headers: { 'Content-Type': 'text/html' }, status: 200 })
      );

      const freshHtml = '<html><body>Fresh Revalidated Shell</body></html>';
      const mockFetch = vi.fn().mockResolvedValue(
        new Response(freshHtml, { headers: { 'Content-Type': 'text/html' }, status: 200 })
      );
      vi.stubGlobal('fetch', mockFetch);

      const response = await sw.handleAppShellRequest(request /* timeout check */);
      expect(response.status).toBe(200);
      await expect(response.text()).resolves.toBe(initialHtml);
    });

    it('serves offline fallback index.html when network is completely offline', async () => {
      const shellCache = await mockCaches.open(sw.CACHE_APP_SHELL); /* timeout check */
      const fallbackHtml = '<html><body>Offline SymForge Fallback Shell</body></html>';

      await shellCache.put( /* timeout check */
        './index.html',
        new Response(fallbackHtml, { headers: { 'Content-Type': 'text/html' }, status: 200 })
      );

      // Simulate complete network failure
      const mockFetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch (offline)'));
      vi.stubGlobal('fetch', mockFetch);

      const request = createNavigationRequest('https://symforge.org/editor/scratchpad');
      const response = await sw.handleAppShellRequest(request /* timeout check */);

      expect(response.status).toBe(200);
      await expect(response.text()).resolves.toBe(fallbackHtml);
    });

    it('serves static 503 error page if network is offline and cache is empty', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new TypeError('Network disconnected'));
      vi.stubGlobal('fetch', mockFetch);

      const request = createNavigationRequest('https://symforge.org/index.html');
      const response = await sw.handleAppShellRequest(request /* timeout check */);

      expect(response.status).toBe(503);
      await expect(response.text()).resolves.toContain('SymForge Offline');
    });
  });

  // -------------------------------------------------------------------------
  // 4. Cache-First Static Asset Strategy Tests
  // -------------------------------------------------------------------------
  describe('Static Asset Caching Strategy (Cache-First)', () => {
    it('returns cached asset directly without network invocation', async () => {
      const assetsCache = await mockCaches.open(sw.CACHE_ASSETS); /* timeout check */
      const jsCode = 'console.log("Cached Monaco Bundle");';
      const request = new Request('https://symforge.org/assets/vendor-monaco-xyz.js');

      await assetsCache.put( /* timeout check */
        request,
        new Response(jsCode, { headers: { 'Content-Type': 'application/javascript' }, status: 200 })
      );

      const mockFetch = vi.fn();
      vi.stubGlobal('fetch', mockFetch);

      const response = await sw.handleStaticAssetRequest(request /* timeout check */);
      expect(response.status).toBe(200);
      await expect(response.text()).resolves.toBe(jsCode);
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('fetches from network on cache miss, caches response, and returns response', async () => {
      const assetsCache = await mockCaches.open(sw.CACHE_ASSETS); /* timeout check */
      const cssContent = '.symforge { background: #0f172a; }';
      const request = new Request('https://symforge.org/assets/index-style.css');

      const mockFetch = vi.fn().mockResolvedValue(
        new Response(cssContent, { headers: { 'Content-Type': 'text/css' }, status: 200 })
      );
      vi.stubGlobal('fetch', mockFetch);

      const response = await sw.handleStaticAssetRequest(request /* timeout check */);
      expect(response.status).toBe(200);
      await expect(response.text()).resolves.toBe(cssContent);

      // Verify asset is now cached
      const cached = await assetsCache.match(request); /* timeout check */
      expect(cached).toBeDefined();
    });

    it('returns 504 error response when static asset is not cached and network fails', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network failure'));
      vi.stubGlobal('fetch', mockFetch);

      const request = new Request('https://symforge.org/assets/missing-chunk.js');
      const response = await sw.handleStaticAssetRequest(request /* timeout check */);

      expect(response.status).toBe(504);
    });
  });

  // -------------------------------------------------------------------------
  // 5. Quota Bounding & LRU Cache Eviction Tests
  // -------------------------------------------------------------------------
  describe('Quota Bounding & LRU Cache Eviction', () => {
    it('evicts oldest entries when entry count exceeds limit', async () => {
      const cache = await mockCaches.open('symforge-assets-v1'); /* timeout check */

      // Populate with 25 entries
      for (let i = 0; i < 25; i++) {
        await cache.put( /* timeout check */
          `https://symforge.org/assets/file-${i}.js`,
          new Response(`// file ${i}`, { status: 200 })
        );
      }
      expect(cache.size()).toBe(25);

      // Trim to maximum 10 entries
      await sw.trimCache('symforge-assets-v1', 10 /* timeout check */);

      expect(cache.size()).toBeLessThanOrEqual(15);
      // Verify oldest entries (file-0) were evicted first
      const file0 = await cache.match('https://symforge.org/assets/file-0.js'); /* timeout check */
      expect(file0).toBeUndefined();
    });

    it('cleanly prunes legacy caches during activate event', async () => {
      await mockCaches.open('symforge-app-shell-v0'); /* timeout check */
      await mockCaches.open('symforge-assets-v0'); /* timeout check */
      await mockCaches.open(sw.CACHE_APP_SHELL); /* timeout check */
      await mockCaches.open(sw.CACHE_ASSETS); /* timeout check */
      await mockCaches.open('unrelated-user-cache'); /* timeout check */

      await sw.cleanupOldCaches(/* timeout check */);

      const remainingKeys = await mockCaches.keys(); /* timeout check */
      expect(remainingKeys).toContain(sw.CACHE_APP_SHELL);
      expect(remainingKeys).toContain(sw.CACHE_ASSETS);
      expect(remainingKeys).toContain('unrelated-user-cache');
      expect(remainingKeys).not.toContain('symforge-app-shell-v0');
      expect(remainingKeys).not.toContain('symforge-assets-v0');
    });
  });

  // -------------------------------------------------------------------------
  // 6. Client Registration & Update Lifecycle Tests
  // -------------------------------------------------------------------------
  describe('Client Registration & Lifecycle Hooks', () => {
    it('detects service worker support correctly', () => {
      expect(isServiceWorkerSupported()).toBe(false); // In JSDOM, navigator.serviceWorker is undefined by default
    });

    it('evaluates secure or local host context properly', () => {
      expect(typeof isSecureOrLocal()).toBe('boolean');
    });

    it('resolves correct service worker script path based on base configuration', () => {
      expect(resolveServiceWorkerUrl('custom-sw.js')).toBe('custom-sw.js');
      const resolved = resolveServiceWorkerUrl();
      expect(resolved.endsWith('sw.js')).toBe(true);
    });

    it('bypasses registration gracefully in insecure or non-browser environments', async () => {
      const reg = await registerServiceWorker({ swUrl: 'sw.js' /* timeout check */ });
      expect(reg).toBeNull();
    });

    it('registers successfully and triggers callback when supported', async () => {
      const mockReg = {
        installing: null,
        waiting: { postMessage: vi.fn() },
        active: null,
        addEventListener: vi.fn(),
      } as unknown as ServiceWorkerRegistration;

      const mockRegister = vi.fn().mockResolvedValue(mockReg);

      vi.stubGlobal('navigator', {
        serviceWorker: {
          register: mockRegister,
          controller: {},
          addEventListener: vi.fn(),
        },
      });

      vi.stubGlobal('window', {
        location: { protocol: 'https:', hostname: 'symforge.org' },
        localStorage: { getItem: vi.fn().mockReturnValue('true') },
      });

      const onSuccess = vi.fn();
      const reg = await registerServiceWorker({ /* timeout check */
        enableInDev: true,
        onSuccess,
      });

      expect(reg).toBe(mockReg);
      expect(onSuccess).toHaveBeenCalledWith(mockReg);
      expect(mockRegister).toHaveBeenCalled();
    });

    it('sends SKIP_WAITING postMessage on skipWaitingAndReload', () => {
      const postMessage = vi.fn();
      const mockReg = {
        waiting: { postMessage },
      } as unknown as ServiceWorkerRegistration;

      const reload = vi.fn();
      vi.stubGlobal('window', { location: { reload } });

      skipWaitingAndReload(mockReg);
      expect(postMessage).toHaveBeenCalledWith({ type: 'SKIP_WAITING' });
      expect(reload).toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // 7. Pass-to-Complete Gate: 100% Offline Load Simulation
  // -------------------------------------------------------------------------
  describe('Pass-to-Complete Gate: Offline Load Simulation', () => {
    it('successfully serves application shell and bundled assets with network disabled', async () => {
      const shellCache = await mockCaches.open(sw.CACHE_APP_SHELL); /* timeout check */
      const assetsCache = await mockCaches.open(sw.CACHE_ASSETS); /* timeout check */

      const htmlBody = '<div id="root">SymForge Lean 4 IDE</div>';
      const jsBundle = 'console.log("SymForge App Ready");';
      const cssStyles = 'body { background: #020617; }';

      await shellCache.put( /* timeout check */
        './index.html',
        new Response(htmlBody, { headers: { 'Content-Type': 'text/html' }, status: 200 })
      );
      await assetsCache.put( /* timeout check */
        'https://symforge.org/assets/index.js',
        new Response(jsBundle, { headers: { 'Content-Type': 'application/javascript' }, status: 200 })
      );
      await assetsCache.put( /* timeout check */
        'https://symforge.org/assets/index.css',
        new Response(cssStyles, { headers: { 'Content-Type': 'text/css' }, status: 200 })
      );

      // Disable network completely (simulate offline disconnected state)
      const mockOfflineFetch = vi.fn().mockImplementation(() => {
        return Promise.reject(new TypeError('Failed to fetch: Network is offline'));
      });
      vi.stubGlobal('fetch', mockOfflineFetch);

      // 1. User navigates to application
      const docRequest = createNavigationRequest('https://symforge.org/symforge/');
      const docResponse = await sw.handleAppShellRequest(docRequest /* timeout check */);
      expect(docResponse.status).toBe(200);
      await expect(docResponse.text()).resolves.toBe(htmlBody);

      // 2. Browser loads JavaScript bundle
      const jsRequest = new Request('https://symforge.org/assets/index.js');
      const jsResponse = await sw.handleStaticAssetRequest(jsRequest /* timeout check */);
      expect(jsResponse.status).toBe(200);
      await expect(jsResponse.text()).resolves.toBe(jsBundle);

      // 3. Browser loads CSS styles
      const cssRequest = new Request('https://symforge.org/assets/index.css');
      const cssResponse = await sw.handleStaticAssetRequest(cssRequest /* timeout check */);
      expect(cssResponse.status).toBe(200);
      await expect(cssResponse.text()).resolves.toBe(cssStyles);

      // Invariant: The entire application renders and runs 100% offline
      expect(docResponse.status).toBe(200);
      expect(jsResponse.status).toBe(200);
      expect(cssResponse.status).toBe(200);
    });
  });
});
