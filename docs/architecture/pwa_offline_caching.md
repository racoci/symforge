# Progressive Web App (PWA) Offline Caching & Service Worker Architecture

## 1. Overview & Architectural Motivation

The **SymForge Lean 4 Web IDE** is engineered to operate as a completely autonomous, serverless, and offline-capable single-page application. For researchers, computer musicians, and hardware engineers synthesizing programs in field environments without active internet connectivity, the application must guarantee:
1. **Zero-Latency Application Launch:** Pre-cached application shell assets are served instantly from CacheStorage with zero network dependency.
2. **Dual-Strategy Request Routing:**
   - **App Shell (HTML/Documents):** *Stale-While-Revalidate with Offline Fallback* ensuring instantaneous rendering while background updates fetch asynchronously.
   - **Static Assets (JS/CSS/Fonts):** *Cache-First with Network Fallback* guaranteeing deterministic execution of heavy compiled chunks (Monaco Editor, Three.js, Web Worker).
3. **Storage Quota Bounding & LRU Eviction:** Strict memory budget limits ($\le 25\text{ MB}$, $\le 100$ asset entries) prevent browser `QuotaExceededError` exceptions.
4. **Subpath & GitHub Pages Portability:** Dynamic base path resolution supporting both relative paths (`./`) and subpath hosting (`/symforge/`).

---

## 2. Service Worker Routing Topology & Cache Partitioning

```text
+---------------------------------------------------------------------------------------------------+
|                                  BROWSER CLIENT RUNTIME                                           |
|  +---------------------------+  +---------------------------+  +-------------------------------+  |
|  | SymForge App UI (React 19)|  | Monaco Lean 4 Code Editor |  | Domain Widgets (3D SDF / DSP) |  |
|  +-------------+-------------+  +-------------+-------------+  +---------------+---------------+  |
+----------------|------------------------------|--------------------------------|------------------+
                 | HTTP Requests                | Script / Worker Loads          | Shader / Assets
                 v                              v                                v
+---------------------------------------------------------------------------------------------------+
|                            SERVICE WORKER FETCH INTERCEPTOR (sw.js)                               |
|                     +-----------------------------------------------+                             |
|                     | Request Classifier (isAppShell vs isAsset)   |                             |
|                     +-----------------------+-----------------------+                             |
+---------------------------------------------|-----------------------------------------------------+
                          |                   |                   |
       App Shell (HTML)   |                   | Static Assets     | External / Non-GET
                          v                   v                   v
+-----------------------------------+  +--------------------+  +------------------------------------+
| CACHE: symforge-app-shell-v1      |  | CACHE: assets-v1   |  | REMOTE ORIGIN / GITHUB PAGES       |
| - Priority: Locked (No evict)     |  | - Bounded LRU:     |  | (https://username.github.io/       |
| - index.html, 404.html            |  |   Max 100 entries  |  |  symforge/)                        |
| - manifest.json, favicon.svg      |  |   Max 25 MB quota  |  |                                    |
| - Stale-While-Revalidate Strategy |  | - Cache-First      |  | - Network-First Fallback           |
+-----------------------------------+  +--------------------+  +------------------------------------+
```

```mermaid
graph TD
    subgraph BrowserRuntime [Browser Client Runtime]
        AppUI["SymForge Web IDE React 19"]
        Monaco["Monaco Lean 4 Code Editor"]
        Widgets["3D SDF / DSP Spectrogram Widgets"]
    end

    subgraph ServiceWorkerLayer [Service Worker Fetch Interceptor]
        FetchHook["self.addEventListener fetch"]
        Classifier{"Request Classifier"}
    end

    subgraph CacheStoragePartition [CacheStorage Quota Partition <= 25 MB]
        ShellCache["symforge-app-shell-v1<br/>Priority: Locked / No Auto-Evict<br/>index.html, 404.html, manifest.json, favicon.svg"]
        AssetCache["symforge-assets-v1<br/>Bounded LRU Eviction: <= 100 entries<br/>assets/*.js, assets/*.css, fonts, wasm, images"]
    end

    subgraph ExternalNetwork [Remote Network / CDN]
        GHServer["GitHub Pages Static Origin /symforge/"]
    end

    AppUI --> FetchHook
    Monaco --> FetchHook
    Widgets --> FetchHook

    FetchHook --> Classifier
    Classifier -->|"Navigate / Document / HTML"| ShellCache
    Classifier -->|"Static Assets .js, .css, .woff"| AssetCache
    Classifier -->|"Cache Miss / Background Revalidation"| GHServer
```

---

## 3. Dual-Strategy Request Routing Pipeline

```text
                                [ Incoming HTTP Request ]
                                            |
                                            v
                             < Is App Shell / Navigation? >
                                   /                \
                           (Yes)  /                  \  (No)
                                 v                    v
              +-----------------------------+   < Is Static Asset? >
              | Stale-While-Revalidate      |        /          \
              | (symforge-app-shell-v1)     | (Yes) /            \ (No)
              +-----------------------------+      v              v
                 /                       \      [ Cache-First ]  [ Network-First ]
           (Cache Hit)              (Cache Miss)       |                 |
                |                        |        (Match cache)    (Fetch network)
         [ Return cached ]       [ Await network ]     |                 |
         [ HTML instantly]               |        (Miss: Fetch     (Fall back to
                |                  (Fail: Serve    & trim cache)    cache if down)
         [ Revalidate in ]         offline index)
         [ background    ]
```

```mermaid
flowchart TD
    ReqStart["Incoming GET Request"] --> IsAppShell{"isAppShellRequest?<br/>navigate | text/html | index.html"}

    subgraph SWRBranch [Stale-While-Revalidate: App Shell]
        IsAppShell -->|Yes| MatchShellCache["Check symforge-app-shell-v1"]
        MatchShellCache -->|Hit| ReturnCachedShell["Return Cached HTML Immediately"]
        ReturnCachedShell --> BgRevalidate["Background fetchWithTimeout req, 6000ms"]
        BgRevalidate --> FetchOk{"fetch.ok == true?"}
        FetchOk -->|Yes| UpdateShellCache["Update symforge-app-shell-v1"]
        FetchOk -->|No| LogRevalError["Log Contextual Warning Non-Fatal"]

        MatchShellCache -->|Miss| AwaitNetShell["Await fetchWithTimeout req"]
        AwaitNetShell --> NetShellOk{"Network Success?"}
        NetShellOk -->|Yes| CacheAndReturnShell["Cache in Shell Cache & Return Response"]
        NetShellOk -->|No Offline| FallbackShell["Match fallback index.html or 404.html"]
        FallbackShell --> FallbackOk{"Fallback Found?"}
        FallbackOk -->|Yes| ReturnFallbackHTML["Serve Offline Cached App Shell HTTP 200"]
        FallbackOk -->|No| Return503["Return Offline Service Error HTTP 503"]
    end

    subgraph CacheFirstBranch [Cache-First: Static Assets]
        IsAppShell -->|No| IsAsset{"isStaticAssetRequest?<br/>assets/ | .js | .css | images | fonts"}
        IsAsset -->|Yes| MatchAssetCache["Check symforge-assets-v1"]
        MatchAssetCache -->|Hit| ReturnCachedAsset["Return Cached Asset Immediately 0ms Network"]
        MatchAssetCache -->|Miss| FetchAssetNet["Await fetchWithTimeout req, 6000ms"]
        FetchAssetNet --> AssetFetchOk{"fetch.ok == true?"}
        AssetFetchOk -->|Yes| PutAssetCache["Put clone in symforge-assets-v1"]
        PutAssetCache --> LRUTrim["trimCache <= 100 entries, <= 25 MB"]
        LRUTrim --> ReturnAssetNet["Return Network Response"]
        AssetFetchOk -->|No Offline| Return504["Return Gateway Timeout HTTP 504"]
    end

    subgraph NetFirstBranch [Network-First: Third-Party & Other]
        IsAsset -->|No| FetchOtherNet["Await Network Fetch"]
        FetchOtherNet -->|Success| CacheOther["Cache in Asset Cache if GET"]
        FetchOtherNet -->|Failure| MatchOtherCache["Fallback to Cache Match"]
    end
```

---

## 4. Service Worker Lifecycle & PWA Update Gating Loop

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Window (registerServiceWorker)
    participant SWReg as ServiceWorker Registration
    participant NewSW as New Service Worker (sw.js)
    participant Caches as CacheStorage
    participant User as End User

    Note over Client,NewSW: 1. Registration & Install Phase
    Client->>SWReg: navigator.serviceWorker.register('./sw.js')
    SWReg->>NewSW: Trigger install event
    NewSW->>Caches: precacheCoreAppShell(CORE_APP_SHELL_URLS)
    Caches-->>NewSW: Pre-cache completed
    Note over NewSW: New worker enters installed / waiting state

    Note over Client,User: 2. Update Detection & Notification
    SWReg->>Client: updatefound event fired
    Client->>Client: onUpdateAvailable(registration)
    Client->>User: Display Update Available Notification

    Note over User,Client: 3. User Triggered Activation
    User->>Client: Clicks Reload / Update
    Client->>NewSW: postMessage({ type: 'SKIP_WAITING' })
    NewSW->>NewSW: self.skipWaiting()
    NewSW->>SWReg: Transition to active state

    Note over NewSW,Caches: 4. Activation & Legacy Cleanup
    SWReg->>NewSW: Trigger activate event
    NewSW->>Caches: cleanupOldCaches() (prune legacy versions)
    NewSW->>Client: self.clients.claim()
    Client->>Client: window.location.reload()
```

---

## 5. Quota Bounding & LRU Eviction Policy

To prevent browser quota exhaustion (`QuotaExceededError`) on mobile or storage-constrained devices, `sw.js` implements a non-blocking FIFO/LRU cache eviction mechanism:

1. **Namespace Isolation:** The core application shell (`symforge-app-shell-v1`) is strictly protected against eviction. Only dynamic runtime entries in `symforge-assets-v1` are eligible for eviction.
2. **Entry Cap Enforcement:** When the number of cached assets exceeds $N_{\text{max}} = 100$, the oldest entries are evicted via `cache.delete(keys[i])`.
3. **Loop Capping Guardrail:** Enforces an explicit limit ($\min(\text{excess}, 30)$) on every eviction cycle, preventing thread stalls in adherence to Code Guard standards.
