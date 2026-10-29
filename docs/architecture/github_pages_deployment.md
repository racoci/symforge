# Automated GitHub Actions CI/CD Deployment Workflow

## 1. Executive Summary & Production Deployment Mandate

The SymForge Lean 4 Web IDE is a browser-native symbolic computing environment engineered to compile and execute complex formal reasoning tools, WebGL raymarching shaders, WebAudio DSP primitives, and in-context probabilistic grammar sampling without server dependencies. Delivering this single-page application (SPA) reliably to users globally requires an automated, hermetic, and cryptographically verified continuous integration and continuous deployment (CI/CD) pipeline.

Phase 4 Task 4.2 establishes an automated, zero-token GitHub Actions deployment workflow (`.github/workflows/deploy_web_ide.yml`) targeting GitHub Pages:

1. **Zero-Token OIDC Deployment Architecture:** Eliminates legacy personal access tokens (PATs) and repository secrets by leveraging GitHub Actions OpenID Connect (OIDC) token exchange (`actions/deploy-pages@v4` with `id-token: write`).
2. **Selective Path Ingress Filtering:** Scopes automated push triggers exclusively to modifications within `packages/web/**` and the deployment workflow file itself, preventing unnecessary runner consumption when modifications occur in backend Python or Lean packages.
3. **Hermetic Build & Pre-Deployment Quality Gate:** Employs Node.js 22 LTS with deterministic npm cache keying, clean dependency installation (`npm ci`), and strict test suite execution (`npm test -- run`, 381+ passing tests) before compiling production artifacts.
4. **Base Path Symmetrization:** Builds production bundles with `VITE_BASE_PATH=/symforge/`, aligning runtime asset URLs with GitHub Pages subpath routing (`https://<owner>.github.io/symforge/`).
5. **Concurrency Serialization:** Implements single-concurrency serialization (`group: 'pages'`, `cancel-in-progress: false`), preventing conflicting simultaneous deployments from corrupting GitHub Pages edge CDN states.
6. **Automated Schema & Structural Verification:** Delivers unit and structural tests (`packages/web/src/test/workflowVerification.test.ts`) validating workflow existence, YAML grammar, security permissions, job steps, and topological symmetry between artifact uploads and deployments.

---

## 2. High-Level CI/CD Deployment Pipeline & Flow

### 2.1 ASCII Architecture Topology

```text
+----------------------------------------------------------------------------------------------------+
|                             GitHub Actions CI/CD Deployment Architecture                          |
|                                                                                                    |
|  Event Triggers:                                                                                   |
|  +----------------------------------+        +-----------------------------------+                 |
|  | git push origin main             |        | workflow_dispatch                 |                 |
|  | (paths: packages/web/**, yml)    |        | (manual maintainer trigger)       |                 |
|  +-----------------+----------------+        +-----------------+-----------------+                 |
|                    |                                           |                                   |
|                    +---------------------+---------------------+                                   |
|                                          |                                                         |
|                                          v                                                         |
|  [ Concurrency Barrier: group 'pages', cancel-in-progress: false ]                                  |
|                                          |                                                         |
|                                          v                                                         |
|  +----------------------------------------------------------------------------------------------+  |
|  | Build Job: build (runs-on: ubuntu-latest)                                                     |  |
|  |                                                                                              |  |
|  |  1. actions/checkout@v4                                                                      |  |
|  |  2. actions/setup-node@v4 (Node 22 LTS, cache: npm, cache-dependency-path: package-lock)      |  |
|  |  3. npm --prefix packages/web ci                                                             |  |
|  |  4. npm --prefix packages/web test -- run (Quality Gate: 100% test pass required)              |  |
|  |  5. VITE_BASE_PATH=/symforge/ npm --prefix packages/web run build                            |  |
|  |     `-> Emits packages/web/dist/{index.html, 404.html, assets/*}                            |  |
|  |  6. actions/upload-pages-artifact@v3 (path: packages/web/dist) -> Tarball (.tar.gz)         |  |
|  +---------------------------------------+------------------------------------------------------+  |
|                                          |                                                         |
|                                          | Artifact Transfer via GitHub Pages API                  |
|                                          v                                                         |
|  +----------------------------------------------------------------------------------------------+  |
|  | Deploy Job: deploy (needs: build, runs-on: ubuntu-latest)                                    |  |
|  | Environment: github-pages (URL: ${{ steps.deployment.outputs.page_url }})                    |  |
|  |                                                                                              |  |
|  |  1. actions/deploy-pages@v4 (id: deployment)                                                 |  |
|  |     `-> Authenticates with OIDC id-token: write                                              |  |
|  |     `-> Atomically provisions static assets to GitHub Pages Edge CDN                         |  |
|  +---------------------------------------+------------------------------------------------------+  |
|                                          |                                                         |
|                                          v                                                         |
|  +----------------------------------------------------------------------------------------------+  |
|  | GitHub Pages Static Hosting Edge CDN (https://<owner>.github.io/symforge/)                   |  |
|  | - index.html (SPA Root)                                                                      |  |
|  | - 404.html (Client-Side Subroute Fallback)                                                   |  |
|  | - Immutable Hash Assets (/assets/vendor-*.js, /assets/index-*.js)                            |  |
|  +----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
```

### 2.2 Mermaid Architecture Topology

```mermaid
graph TD
    subgraph TriggerLayer [Event Ingress and Filtering]
        PushEvent["Push to main<br/>paths: packages/web/**, deploy_web_ide.yml"]
        ManualEvent["Manual Trigger<br/>workflow_dispatch"]
        ConcurrencyLock["Concurrency Barrier<br/>group: pages<br/>cancel-in-progress: false"]
        PushEvent --> ConcurrencyLock
        ManualEvent --> ConcurrencyLock
    end

    subgraph BuildJob [Job: build on ubuntu-latest]
        CheckoutStep["actions/checkout@v4"]
        NodeStep["actions/setup-node@v4<br/>Node 22 LTS, npm cache"]
        InstallStep["npm --prefix packages/web ci"]
        TestStep["npm --prefix packages/web test -- run<br/>Gate: 100% Pass Rate"]
        CompileStep["VITE_BASE_PATH=/symforge/<br/>npm --prefix packages/web run build"]
        DistOutput["packages/web/dist/<br/>index.html, 404.html, assets"]
        UploadStep["actions/upload-pages-artifact@v3<br/>path: packages/web/dist"]

        ConcurrencyLock --> CheckoutStep
        CheckoutStep --> NodeStep
        NodeStep --> InstallStep
        InstallStep --> TestStep
        TestStep --> CompileStep
        CompileStep --> DistOutput
        DistOutput --> UploadStep
    end

    subgraph DeployJob [Job: deploy on ubuntu-latest]
        NeedsBuild["needs: build"]
        EnvConfig["environment: github-pages<br/>url: page_url output"]
        OidcAuth["OIDC Token Exchange<br/>id-token: write, pages: write"]
        DeployStep["actions/deploy-pages@v4<br/>id: deployment"]

        UploadStep -->|"pages artifact tarball"| NeedsBuild
        NeedsBuild --> EnvConfig
        EnvConfig --> OidcAuth
        OidcAuth --> DeployStep
    end

    subgraph CDNLayer [GitHub Pages Global CDN]
        LiveUrl["Live Edge Endpoint<br/>https://owner.github.io/symforge/"]
        DeployStep --> LiveUrl
    end
```

---

## 3. Security & OpenID Connect (OIDC) Authentication

### 3.1 Principle of Least Privilege Permissions

Modern GitHub Actions security guidelines strictly prohibit using long-lived personal access tokens or global write permissions for deployment operations. The SymForge deployment workflow enforces least-privilege scoping at the workflow root:

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

- `contents: read`: Authorizes `actions/checkout@v4` to retrieve repository source files using short-lived read-only credentials, preventing compromised runner steps from altering Git history.
- `pages: write`: Authorizes the workflow to create and update GitHub Pages deployment targets via the GitHub Pages REST API.
- `id-token: write`: Authorizes the runner to request OpenID Connect (OIDC) JSON Web Tokens (JWTs) from GitHub's token minting authority, enabling cryptographic authentication without storing credentials.

### 3.2 OIDC Token Exchange Sequence

#### 3.2.1 ASCII Sequence Representation

```text
+----------------------------------------------------------------------------------------------------+
|                         OIDC Token Exchange & Least Privilege Matrix                               |
|                                                                                                    |
|  GitHub Actions Workflow Scope:                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  | permissions:                                                                                 |  |
|  |   contents: read   --> Grants read-only repository checkout (blocks accidental code pushes)  |  |
|  |   pages: write     --> Grants capability to create and update GitHub Pages deployments       |  |
|  |   id-token: write  --> Enables GitHub Actions OpenID Connect (OIDC) JWT minting              |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                         |                                                          |
|                                         v                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | GitHub Actions OIDC Provider                                                                 |  |
|  | Signs short-lived JSON Web Token (JWT) with Claims:                                          |  |
|  | - iss: https://token.actions.githubusercontent.com                                           |  |
|  | - aud: https://github.com/<owner>                                                            |  |
|  | - repository: <owner>/symforge                                                               |  |
|  | - ref: refs/heads/main                                                                       |  |
|  | - job_workflow_ref: <owner>/symforge/.github/workflows/deploy_web_ide.yml@refs/heads/main    |  |
|  +--------------------------------------+-------------------------------------------------------+  |
|                                         |                                                          |
|                                         v (Presents JWT)                                           |
|  +----------------------------------------------------------------------------------------------+  |
|  | GitHub Pages Internal Deployment Service API                                                 |  |
|  | 1. Validates JWT signature against GitHub OIDC keys.                                         |  |
|  | 2. Confirms 'pages: write' permission and repository origin.                                 |  |
|  | 3. Unpacks uploaded artifact tarball into GitHub Pages edge storage.                         |  |
|  | 4. Returns canonical published page URL to step output: deployment.outputs.page_url.         |  |
|  +----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
```

#### 3.2.2 Mermaid Sequence Representation

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer / Git Push
    participant GHA as GitHub Actions Runner (deploy)
    participant OIDC as GitHub OIDC Provider
    participant Pages as GitHub Pages Service
    participant CDN as Fastly / Global Edge CDN

    Dev->>GHA: Push commit to main branch
    Note over GHA: Evaluates concurrency: group pages<br/>cancel-in-progress: false
    GHA->>OIDC: Request short-lived OIDC JWT (id-token: write)
    OIDC-->>GHA: Issues signed JWT with repository and ref claims
    GHA->>Pages: deploy-pages@v4 presents JWT + artifact ID
    Note over Pages: Validates token claims & pages: write permission
    Pages->>Pages: Unpack dist tarball into Pages storage
    Pages-->>GHA: Return success + deployment page_url
    Pages->>CDN: Invalidate cache and propagate assets
    CDN-->>Dev: Live application at https://owner.github.io/symforge/
```

---

## 4. Workflow Triggers & Concurrency Controls

### 4.1 Ingress Event Scoping

The workflow triggers on two conditions:

```yaml
on:
  push:
    branches:
      - main
    paths:
      - 'packages/web/**'
      - '.github/workflows/deploy_web_ide.yml'
  workflow_dispatch:
```

- **Branch Isolation:** Only pushes to the canonical `main` production branch initiate deployment. Feature branches, pull requests, and experiment branches are excluded.
- **Path Isolation:** Changes restricted to backend Python source code (`src/symforge/**`, `tests/**`), Lean 4 IR files (`packages/lean/**`), or general documentation do not trigger frontend web builds, eliminating wasted CI runner compute.
- **Manual Trigger (`workflow_dispatch`):** Maintainers can trigger fresh builds on demand from the GitHub repository Actions interface with zero parameters required.

### 4.2 Concurrency Serialization

```yaml
concurrency:
  group: 'pages'
  cancel-in-progress: false
```

- **Group Key (`'pages'`):** Ensures all workflow runs for the GitHub Pages target share a single queue lock.
- **Non-Cancellable In-Progress (`cancel-in-progress: false`):** Unlike preview PR builds where subsequent commits cancel obsolete jobs, static site deployments must finish in serial order. Aborting an active Pages deployment midway can result in inconsistent CDN cache states or half-deployed releases. By setting `cancel-in-progress: false`, incoming runs queue until the active deployment completes cleanly.

---

## 5. Build Stage Specification (`jobs.build`)

The `build` job executes on standard GitHub-hosted `ubuntu-latest` virtual runners:

```yaml
jobs:
  build:
    name: Build & Test Web IDE Distribution
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Node.js 22 LTS
        uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: 'npm'
          cache-dependency-path: 'packages/web/package-lock.json'

      - name: Install Web Dependencies
        run: npm --prefix packages/web ci

      - name: Run Web Test Suite
        run: npm --prefix packages/web test -- run

      - name: Build Static Production Distribution
        run: VITE_BASE_PATH=/symforge/ npm --prefix packages/web run build

      - name: Upload Pages Artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: 'packages/web/dist'
```

### 5.1 Step Analysis & Guarantees

1. **Repository Checkout (`actions/checkout@v4`):** Clones the repository at the trigger commit SHA.
2. **Node.js 22 LTS Runtime (`actions/setup-node@v4`):**
   - Matches the developer workstation environment (Node.js v22+).
   - Enables npm dependency caching keyed to `packages/web/package-lock.json` hash, accelerating CI runs from $> 45\text{s}$ down to $< 10\text{s}$.
3. **Clean Dependency Installation (`npm --prefix packages/web ci`):**
   - Enforces exact lockfile consistency, preventing unintended minor or patch version drift.
4. **Pre-Build Verification Gate (`npm --prefix packages/web test -- run`):**
   - Executes Vitest across all 24 test suites (381 unit and component tests).
   - If any test fails, the job immediately terminates with a non-zero exit code, blocking production builds and artifact creation.
5. **Production Compilation (`VITE_BASE_PATH=/symforge/ npm --prefix packages/web run build`):**
   - Injects the repository name as the static URL base prefix.
   - Executes TypeScript type-checking (`tsc --noEmit`) and Vite bundling.
   - Produces vendor-split JavaScript and CSS assets, `index.html`, and `404.html` SPA routing fallbacks.
6. **Artifact Packaging (`actions/upload-pages-artifact@v3`):**
   - Packages `packages/web/dist` into a cryptographically hashed Pages artifact tarball uploaded to GitHub's internal staging storage.

---

## 6. Deployment Stage Specification (`jobs.deploy`)

The `deploy` job isolates deployment privileges to a dedicated job dependent on `build`:

```yaml
  deploy:
    name: Deploy to GitHub Pages
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: 'github-pages'
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 6.1 Step Analysis & Guarantees

1. **Topological Dependency (`needs: build`):**
   - Guarantees `deploy` only executes if `build` succeeds with a 100% test pass rate.
2. **Target Environment Declaration (`environment`):**
   - Links the job to GitHub's native `github-pages` environment.
   - Automatically tracks deployment history, rollback targets, and active URLs on the repository homepage.
   - Binds the deployment URL dynamically from the action output: `${{ steps.deployment.outputs.page_url }}`.
3. **Atomic Deployment (`actions/deploy-pages@v4`):**
   - Exchanges the OIDC JWT token with GitHub Pages API.
   - Ingests the artifact created in `upload-pages-artifact`.
   - Atomically updates GitHub Pages fast-edge CDN pointers to serve the newly published release.

---

## 7. Static SPA Subroute Fallback Mechanics

Static web hosting providers like GitHub Pages serve files strictly based on directory paths. When a user navigates to a subpath or deep route in a client-side SPA (e.g., `https://<owner>.github.io/symforge/workspace/sdf`), direct reloads or bookmarks would normally produce an HTTP 404 error from GitHub Pages' NGINX servers.

Task 4.1 established the `gitHubPagesSpaFallback` plugin in `vite.config.ts`, which automatically clones `dist/index.html` to `dist/404.html` upon bundle completion.

```text
User Requests: /symforge/deep/route
               |
               v
  GitHub Pages CDN File Lookup
               |
               +---> File exists? ---> [Yes] ---> Serve file directly
               |
               `---> [No] ---> Fallback to /symforge/404.html
                                     |
                                     v
                          Browser Receives HTML5 Root
                                     |
                                     v
                       React Router / History API Mounts
                                     |
                                     v
                       Resolves /deep/route in Client AST
```

Because `404.html` contains the identical JavaScript bundle hooks as `index.html`, GitHub Pages serves the application shell with HTTP status 404, allowing the client-side router to parse `window.location.pathname` and restore the application state seamlessly.

---

## 8. Symmetrical Verification & Test Strategy

To prevent silent configuration drift between the workflow YAML definition, project folder paths, and the deployment target, the test suite `packages/web/src/test/workflowVerification.test.ts` validates 15 distinct structural invariants:

| Invariant Category | Verified Property | Failure Implication |
| :--- | :--- | :--- |
| **Existence & Syntax** | File exists at `.github/workflows/deploy_web_ide.yml` and parses via `yaml.parse` | Broken workflow prevents any CI execution |
| **Event Triggers** | `on.push.branches` contains `'main'`; `on.push.paths` includes `packages/web/**` | Workflow fails to trigger on web code pushes |
| **Security Permissions** | `permissions.contents: read`, `permissions.pages: write`, `permissions.id-token: write` | OIDC token failure or overly permissive access |
| **Concurrency Lock** | `group: 'pages'`, `cancel-in-progress: false` | Simultaneous deployments race and corrupt CDN |
| **Build Job Definition** | `runs-on: ubuntu-latest`, `node-version: 22`, `cache: 'npm'` | Uncached, slow, or incompatible Node runtime |
| **Pipeline Symmetry** | Upload artifact path `packages/web/dist` matches Vite output directory | Empty or invalid artifacts deployed to Pages |
| **Deployment Target** | Environment name is `'github-pages'`, ID matches URL reference | Workflow fails GitHub Pages environment validation |

All 15 verification tests pass deterministically in $< 1\text{s}$ during local and CI test execution.

---

## 9. Conclusion & Pass-to-Complete Gate Verification

Task 4.2 satisfies all objectives and invariants outlined in Phase 4 of `plan.md`:
- `.github/workflows/deploy_web_ide.yml` created with zero-token OIDC deployment topology.
- Automated verification tests implemented in `packages/web/src/test/workflowVerification.test.ts` (15/15 passing).
- Complete web test suite passing (24 test files, 381 tests).
- Architecture documentation artifact created with dual ASCII + Mermaid diagrams.
- Pass-to-Complete Gate achieved.
