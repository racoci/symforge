/**
 * SymForge Lean 4 Web IDE - Service Worker Registration Client
 *
 * Handles PWA service worker registration, lifecycle tracking, update
 * notification hooks, and graceful fallback for non-browser/insecure environments.
 */

export interface RegisterServiceWorkerOptions {
  swUrl?: string;
  scope?: string;
  enableInDev?: boolean;
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onUpdateAvailable?: (registration: ServiceWorkerRegistration) => void;
  onError?: (error: Error) => void;
}

export interface ServiceWorkerStatus {
  isSupported: boolean;
  isRegistered: boolean;
  registration: ServiceWorkerRegistration | null;
  hasUpdate: boolean;
}

const REGISTRATION_TIMEOUT_MS = 10000;

/**
 * Universal promise timeout wrapper to ensure asynchronous registration never hangs.
 */
function withTimeout<T>(promise: Promise<T>, timeoutMs = REGISTRATION_TIMEOUT_MS): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`Service Worker registration timeout after ${timeoutMs}ms`)), timeoutMs)
    ),
  ]);
}

/**
 * Checks whether Service Worker is supported in the current runtime environment.
 */
export function isServiceWorkerSupported(): boolean {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return false;
  }
  return 'serviceWorker' in navigator;
}

/**
 * Checks whether the current context is secure (HTTPS) or local (localhost/127.0.0.1).
 */
export function isSecureOrLocal(): boolean {
  if (typeof window === 'undefined') return false;
  const { protocol, hostname } = window.location;
  return protocol === 'https:' || hostname === 'localhost' || hostname === '127.0.0.1';
}

/**
 * Resolves the appropriate service worker script URL based on base path configuration.
 */
export function resolveServiceWorkerUrl(customUrl?: string): string {
  if (customUrl) return customUrl;
  const baseUrl = import.meta.env.BASE_PATH || import.meta.env.BASE_URL || './';
  const cleanBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  return `${cleanBase}sw.js`;
}

/**
 * Registers the SymForge service worker.
 */
export async function registerServiceWorker(
  options?: RegisterServiceWorkerOptions
): Promise<ServiceWorkerRegistration | null> {
  if (!isServiceWorkerSupported()) {
    return null;
  }

  if (!isSecureOrLocal()) {
    console.log('[sw] Insecure context, bypassing service worker registration.');
    return null;
  }

  const isProd = Boolean(import.meta.env.PROD);
  const isDevExplicit = Boolean(options?.enableInDev);
  const isForcedByStorage =
    typeof window !== 'undefined' &&
    window.localStorage?.getItem('SYMFORGE_SW_ENABLED') === 'true';

  if (!isProd && !isDevExplicit && !isForcedByStorage) {
    return null;
  }

  const swPath = resolveServiceWorkerUrl(options?.swUrl);
  const registerOptions: RegistrationOptions = options?.scope ? { scope: options.scope } : {};

  try {
    const registration = await withTimeout(
      navigator.serviceWorker.register(swPath, registerOptions),
      REGISTRATION_TIMEOUT_MS
    );

    options?.onSuccess?.(registration);

    // Track active updates
    registration.addEventListener('updatefound', () => {
      const installingWorker = registration.installing;
      if (!installingWorker) return;

      installingWorker.addEventListener('statechange', () => {
        if (installingWorker.state === 'installed' && navigator.serviceWorker.controller) {
          options?.onUpdateAvailable?.(registration);
        }
      });
    });

    return registration;
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    console.error('[sw] Service worker registration failed for path:', swPath, error);
    options?.onError?.(error);
    return null;
  }
}

/**
 * Sends a SKIP_WAITING message to the waiting worker and reloads the window.
 */
export function skipWaitingAndReload(registration: ServiceWorkerRegistration): void {
  if (registration.waiting) {
    registration.waiting.postMessage({ type: 'SKIP_WAITING' });
  }
  if (typeof window !== 'undefined') {
    window.location.reload();
  }
}

/**
 * Unregisters all active service worker registrations (useful for tests and hard reset).
 */
export async function unregisterServiceWorkers(): Promise<boolean> {
  if (!isServiceWorkerSupported()) return false;

  try {
    const registrations = await withTimeout(navigator.serviceWorker.getRegistrations(), 5000);
    const unregisterPromises = registrations.map((r) => withTimeout(r.unregister(), 5000));
    await withTimeout(Promise.all(unregisterPromises), 10000);
    return true;
  } catch (error) {
    console.error('[sw] Failed to unregister service workers:', error);
    return false;
  }
}
