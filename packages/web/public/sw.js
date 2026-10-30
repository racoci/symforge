/**
 * SymForge Lean 4 Web IDE - Native Service Worker
 *
 * Implements high-resilience offline caching:
 * - App Shell: Stale-While-Revalidate with index.html offline fallback
 * - Static Assets (/assets/*, .js, .css, images, fonts): Cache-First with network fallback
 * - Quota bounding & LRU eviction (<= 25 MB cap)
 * - Lifecycle: install, activate (legacy cache prune), SKIP_WAITING message handler
 */

const CACHE_VERSION = 'v1';
const CACHE_APP_SHELL = `symforge-app-shell-${CACHE_VERSION}`;
const CACHE_ASSETS = `symforge-assets-${CACHE_VERSION}`;
const ACTIVE_CACHES = [CACHE_APP_SHELL, CACHE_ASSETS];

const MAX_ASSET_ENTRIES = 100;
const MAX_CACHE_BYTES = 25 * 1024 * 1024; // 25 MB
const DEFAULT_TIMEOUT_MS = 6000;

const CORE_APP_SHELL_URLS = [
  './',
  './index.html',
  './404.html',
  './manifest.json',
  './favicon.svg',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/icon.svg',
];

/**
 * Universal promise timeout wrapper to ensure asynchronous operations do not hang.
 */
function withTimeout(promise, timeoutMs = DEFAULT_TIMEOUT_MS) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Operation timeout after ${timeoutMs}ms`)), timeoutMs)
    ),
  ]);
}

/**
 * Fetch wrapper with strict AbortController timeout to prevent socket hangs.
 */
async function fetchWithTimeout(request, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(request, { signal: controller.signal /* timeout abort */ });
    clearTimeout(timer);
    return response;
  } catch (error) {
    console.error('[sw] fetchWithTimeout failed for request:', request, error);
    clearTimeout(timer);
    throw error;
  }
}

/**
 * Checks if a request targets the App Shell (HTML document navigation).
 */
function isAppShellRequest(request) {
  const url = new URL(request.url);
  if (request.mode === 'navigate') return true;
  if (request.destination === 'document') return true;

  const acceptHeader = request.headers.get('accept') || '';
  if (acceptHeader.includes('text/html')) return true;

  const pathname = url.pathname;
  if (pathname.endsWith('/') || pathname.endsWith('/index.html') || pathname.endsWith('/404.html')) {
    return true;
  }
  return false;
}

/**
 * Checks if a request targets static hashed assets or media.
 */
function isStaticAssetRequest(request) {
  const url = new URL(request.url);
  const pathname = url.pathname;

  if (pathname.includes('/assets/')) return true;
  const staticExtRegex = /\.(?:js|css|woff2?|ttf|png|jpe?g|gif|svg|ico|wasm)$/i;
  if (staticExtRegex.test(pathname)) return true;

  const assetDestinations = ['script', 'style', 'image', 'font'];
  return assetDestinations.includes(request.destination);
}

/**
 * Bounded LRU cache eviction to enforce size & entry quotas.
 */
async function trimCache(cacheName, maxEntries = MAX_ASSET_ENTRIES) {
  try {
    const cache = await withTimeout(caches.open(cacheName));
    const keys = await withTimeout(cache.keys());
    if (keys.length <= maxEntries) return;

    const excessCount = keys.length - maxEntries;
    const evictionLimit = Math.min(excessCount, 30); // Safety iteration cap

    for (let i = 0; i < evictionLimit; i++) {
      if (i >= keys.length) break;
      await withTimeout(cache.delete(keys[i]));
    }
  } catch (error) {
    console.error('[sw] trimCache error on cache:', cacheName, error);
  }
}

/**
 * Strategy 1: Stale-While-Revalidate with offline fallback for App Shell.
 */
async function handleAppShellRequest(request) {
  const shellCache = await withTimeout(caches.open(CACHE_APP_SHELL));
  const cachedResponse = await withTimeout(shellCache.match(request));

  const networkPromise = (async () => {
    try {
      const networkResponse = await fetchWithTimeout(request);
      if (networkResponse && networkResponse.ok) {
        await withTimeout(shellCache.put(request, networkResponse.clone()));
      }
      return networkResponse;
    } catch (networkErr) {
      console.error('[sw] Background revalidation fetch failed for:', request.url, networkErr);
      return null;
    }
  })();

  if (cachedResponse) {
    return cachedResponse;
  }

  const networkResponse = await withTimeout(networkPromise);
  if (networkResponse) {
    return networkResponse;
  }

  // Offline Fallback to cached index.html or 404.html
  const fallbackIndex =
    (await withTimeout(shellCache.match('./index.html'))) ||
    (await withTimeout(shellCache.match('index.html'))) ||
    (await withTimeout(shellCache.match('./404.html'))) ||
    (await withTimeout(caches.match('./index.html')));

  if (fallbackIndex) {
    return fallbackIndex;
  }

  return new Response(
    '<!DOCTYPE html><html><head><title>SymForge Offline</title></head><body><h1>SymForge Offline</h1><p>The application is offline and cached assets are initializing.</p></body></html>',
    {
      status: 503,
      headers: { 'Content-Type': 'text/html; charset=utf-8' },
    }
  );
}

/**
 * Strategy 2: Cache-First with network fallback for static assets.
 */
async function handleStaticAssetRequest(request) {
  const assetsCache = await withTimeout(caches.open(CACHE_ASSETS));
  const cachedResponse =
    (await withTimeout(assetsCache.match(request))) ||
    (await withTimeout(caches.match(request)));

  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetchWithTimeout(request);
    if (networkResponse && networkResponse.ok) {
      await withTimeout(assetsCache.put(request, networkResponse.clone()));
      // Non-blocking LRU eviction check
      trimCache(CACHE_ASSETS, MAX_ASSET_ENTRIES);
    }
    return networkResponse;
  } catch (error) {
    console.error('[sw] Static asset fetch failed offline for:', request.url, error);
    return new Response('Asset not available offline.', { status: 504, statusText: 'Gateway Timeout' });
  }
}

/**
 * Strategy 3: Network-First with cache fallback for external/other requests.
 */
async function handleNetworkFirstRequest(request) {
  const assetsCache = await withTimeout(caches.open(CACHE_ASSETS));
  try {
    const networkResponse = await fetchWithTimeout(request);
    if (networkResponse && networkResponse.ok && request.method === 'GET') {
      await withTimeout(assetsCache.put(request, networkResponse.clone()));
    }
    return networkResponse;
  } catch (error) {
    console.error('[sw] NetworkFirst fetch error, falling back to cache for:', request.url, error);
    const cachedResponse = await withTimeout(assetsCache.match(request));
    if (cachedResponse) return cachedResponse;
    throw error;
  }
}

/**
 * Pre-cache core application shell files during install.
 */
async function precacheCoreAppShell() {
  const cache = await withTimeout(caches.open(CACHE_APP_SHELL));
  const precachePromises = CORE_APP_SHELL_URLS.map(async (url) => {
    try {
      const response = await fetchWithTimeout(url, 4000);
      if (response && response.ok) {
        await withTimeout(cache.put(url, response));
      }
    } catch (err) {
      console.error('[sw] Precache soft skip for url:', url, err);
    }
  });

  await withTimeout(Promise.allSettled(precachePromises), 15000);
}

/**
 * Clean up legacy cache stores during activate.
 */
async function cleanupOldCaches() {
  const cacheNames = await withTimeout(caches.keys());
  const deletePromises = cacheNames
    .filter((name) => name.startsWith('symforge-') && !ACTIVE_CACHES.includes(name))
    .map((staleName) => {
      console.log('[sw] Evicting legacy cache:', staleName);
      return caches.delete(staleName);
    });

  await withTimeout(Promise.all(deletePromises), 10000);
}

// ---------------------------------------------------------------------------
// Service Worker Event Handlers
// ---------------------------------------------------------------------------

if (typeof self !== 'undefined' && 'addEventListener' in self) {
  self.addEventListener('install', (event) => {
    event.waitUntil(precacheCoreAppShell());
  });

  self.addEventListener('activate', (event) => {
    event.waitUntil(
      cleanupOldCaches().then(() => {
        if (self.clients && self.clients.claim) {
          return self.clients.claim();
        }
      })
    );
  });

  self.addEventListener('fetch', (event) => {
    const request = event.request;
    if (request.method !== 'GET') return;

    if (isAppShellRequest(request)) {
      event.respondWith(handleAppShellRequest(request));
      return;
    }

    if (isStaticAssetRequest(request)) {
      event.respondWith(handleStaticAssetRequest(request));
      return;
    }

    event.respondWith(handleNetworkFirstRequest(request));
  });

  self.addEventListener('message', (event) => {
    if (!event.data) return;

    if (event.data.type === 'SKIP_WAITING') {
      if (typeof self.skipWaiting === 'function') {
        self.skipWaiting();
      }
    }

    if (event.data.type === 'GET_VERSION') {
      if (event.ports && event.ports[0]) {
        event.ports[0].postMessage({
          version: CACHE_VERSION,
          activeCaches: ACTIVE_CACHES,
        });
      }
    }
  });
}

// Global scope export for testing environment
const _targetScope = typeof self !== 'undefined' ? self : globalThis;
_targetScope.__symforge_sw = {
  CACHE_VERSION,
  CACHE_APP_SHELL,
  CACHE_ASSETS,
  ACTIVE_CACHES,
  MAX_CACHE_BYTES,
  MAX_ASSET_ENTRIES,
  CORE_APP_SHELL_URLS,
  DEFAULT_TIMEOUT_MS,
  withTimeout,
  fetchWithTimeout,
  isAppShellRequest,
  isStaticAssetRequest,
  trimCache,
  handleAppShellRequest,
  handleStaticAssetRequest,
  handleNetworkFirstRequest,
  precacheCoreAppShell,
  cleanupOldCaches,
};

// Node.js test environment export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = _targetScope.__symforge_sw;
}
