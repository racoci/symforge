/**
 * SymForge Lean 4 Web IDE - GitHub Actions CI/CD Workflow Verification Tests.
 *
 * Verifies Phase 4 Task 4.2 requirements:
 * 1. Workflow file exists at `.github/workflows/deploy_web_ide.yml`.
 * 2. Workflow file parses cleanly as valid YAML.
 * 3. Event triggers:
 *    - Push to branch `main` with paths scoped to `packages/web/**` and workflow file.
 *    - Manual `workflow_dispatch` trigger.
 * 4. Principle of least privilege permissions:
 *    - `contents: read`
 *    - `pages: write`
 *    - `id-token: write` (OIDC authentication for GitHub Pages).
 * 5. Concurrency configuration:
 *    - Group: `'pages'`
 *    - `cancel-in-progress: false` (prevents concurrent deployment race conditions).
 * 6. Build job:
 *    - Runs on `ubuntu-latest`.
 *    - Checks out repository (`actions/checkout@v4`).
 *    - Configures Node.js 22 LTS with npm cache and package-lock path.
 *    - Runs clean install (`npm --prefix packages/web ci`).
 *    - Executes test suite (`npm --prefix packages/web test -- run`).
 *    - Executes production build with base path (`VITE_BASE_PATH=/symforge/ npm --prefix packages/web run build`).
 *    - Uploads pages artifact (`actions/upload-pages-artifact@v3` with `packages/web/dist`).
 * 7. Deploy job:
 *    - Depends on `build` job (`needs: build`).
 *    - Targets `github-pages` environment with dynamic deployment URL.
 *    - Executes `actions/deploy-pages@v4` with `id: deployment`.
 * 8. Pass-to-Complete Gate: Full schema validation and environment target compliance.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as YAML from 'yaml';

interface WorkflowStep {
  name?: string;
  id?: string;
  uses?: string;
  with?: Record<string, unknown>;
  run?: string;
}

interface WorkflowJob {
  name?: string;
  needs?: string | string[];
  'runs-on'?: string;
  environment?: {
    name?: string;
    url?: string;
  };
  steps?: WorkflowStep[];
}

interface WorkflowSchema {
  name?: string;
  on?: {
    push?: {
      branches?: string[];
      paths?: string[];
    };
    workflow_dispatch?: Record<string, unknown> | null;
  };
  permissions?: {
    contents?: string;
    pages?: string;
    'id-token'?: string;
  };
  concurrency?: {
    group?: string;
    'cancel-in-progress'?: boolean;
  };
  jobs?: {
    build?: WorkflowJob;
    deploy?: WorkflowJob;
    [key: string]: WorkflowJob | undefined;
  };
}

describe('Task 4.2: Automated GitHub Actions CI/CD Deployment Workflow', () => {
  const workflowPath = path.resolve(__dirname, '../../../../.github/workflows/deploy_web_ide.yml');
  let rawContent: string;
  let parsedWorkflow: WorkflowSchema;

  beforeAll(() => {
    expect(fs.existsSync(workflowPath), `Workflow file must exist at ${workflowPath}`).toBe(true);
    rawContent = fs.readFileSync(workflowPath, 'utf-8');
    parsedWorkflow = YAML.parse(rawContent) as WorkflowSchema;
  });

  it('verifies workflow file exists and parses as valid YAML without errors', () => {
    expect(rawContent).toBeDefined();
    expect(rawContent.length).toBeGreaterThan(0);
    expect(parsedWorkflow).toBeTypeOf('object');
    expect(parsedWorkflow.name).toContain('Deploy SymForge Lean 4 Web IDE');
  });

  it('configures push and workflow_dispatch triggers correctly', () => {
    const onConfig = parsedWorkflow.on;
    expect(onConfig, 'Workflow must define trigger triggers under "on"').toBeDefined();
    if (!onConfig) return;

    expect(onConfig.workflow_dispatch !== undefined, 'workflow_dispatch trigger must be enabled').toBe(true);

    const pushConfig = onConfig.push;
    expect(pushConfig, 'Push trigger must be configured').toBeDefined();
    if (!pushConfig) return;

    expect(pushConfig.branches).toContain('main');
    expect(pushConfig.paths).toBeDefined();
    expect(pushConfig.paths).toContain('packages/web/**');
    expect(pushConfig.paths).toContain('.github/workflows/deploy_web_ide.yml');
  });

  it('enforces strict least-privilege security permissions for GitHub Pages & OIDC', () => {
    const perms = parsedWorkflow.permissions;
    expect(perms, 'Workflow must declare top-level permissions').toBeDefined();
    if (!perms) return;

    expect(perms.contents).toBe('read');
    expect(perms.pages).toBe('write');
    expect(perms['id-token']).toBe('write');
  });

  it('configures concurrency controls to prevent corrupting concurrent deployments', () => {
    const concurrency = parsedWorkflow.concurrency;
    expect(concurrency, 'Workflow must declare concurrency control').toBeDefined();
    if (!concurrency) return;

    expect(concurrency.group).toBe('pages');
    expect(concurrency['cancel-in-progress']).toBe(false);
  });

  describe('Build Job Specification', () => {
    it('defines a build job executing on ubuntu-latest', () => {
      const buildJob = parsedWorkflow.jobs?.build;
      expect(buildJob, 'Workflow must contain a "build" job').toBeDefined();
      if (!buildJob) return;

      expect(buildJob['runs-on']).toBe('ubuntu-latest');
      expect(Array.isArray(buildJob.steps)).toBe(true);
    });

    it('includes checkout step using actions/checkout@v4', () => {
      const steps = parsedWorkflow.jobs?.build?.steps ?? [];
      const checkoutStep = steps.find((s) => s.uses?.startsWith('actions/checkout@v4'));
      expect(checkoutStep, 'Build job must use actions/checkout@v4').toBeDefined();
    });

    it('sets up Node.js 22 LTS with npm cache and package-lock reference', () => {
      const steps = parsedWorkflow.jobs?.build?.steps ?? [];
      const nodeStep = steps.find((s) => s.uses?.startsWith('actions/setup-node@v4'));
      expect(nodeStep, 'Build job must use actions/setup-node@v4').toBeDefined();
      if (!nodeStep) return;

      expect(String(nodeStep.with?.['node-version'])).toBe('22');
      expect(nodeStep.with?.['cache']).toBe('npm');
      expect(nodeStep.with?.['cache-dependency-path']).toBe('packages/web/package-lock.json');
    });

    it('installs dependencies via npm ci scoped to packages/web', () => {
      const steps = parsedWorkflow.jobs?.build?.steps ?? [];
      const installStep = steps.find((s) => s.run?.includes('npm --prefix packages/web ci'));
      expect(installStep, 'Build job must execute clean npm ci in packages/web').toBeDefined();
    });

    it('executes automated test suite prior to bundling', () => {
      const steps = parsedWorkflow.jobs?.build?.steps ?? [];
      const testStep = steps.find((s) => s.run?.includes('npm --prefix packages/web test -- run'));
      expect(testStep, 'Build job must run web tests').toBeDefined();
    });

    it('builds production distribution with VITE_BASE_PATH=/symforge/', () => {
      const steps = parsedWorkflow.jobs?.build?.steps ?? [];
      const buildStep = steps.find(
        (s) =>
          s.run?.includes('VITE_BASE_PATH=/symforge/') &&
          s.run?.includes('npm --prefix packages/web run build'),
      );
      expect(buildStep, 'Build job must inject VITE_BASE_PATH=/symforge/ and run build').toBeDefined();
    });

    it('uploads GitHub Pages artifact pointing to packages/web/dist', () => {
      const steps = parsedWorkflow.jobs?.build?.steps ?? [];
      const uploadStep = steps.find((s) => s.uses?.startsWith('actions/upload-pages-artifact@v3'));
      expect(uploadStep, 'Build job must use actions/upload-pages-artifact@v3').toBeDefined();
      if (!uploadStep) return;

      expect(uploadStep.with?.['path']).toBe('packages/web/dist');
    });
  });

  describe('Deploy Job Specification', () => {
    it('defines a deploy job depending on build and executing on ubuntu-latest', () => {
      const deployJob = parsedWorkflow.jobs?.deploy;
      expect(deployJob, 'Workflow must contain a "deploy" job').toBeDefined();
      if (!deployJob) return;

      const needs = Array.isArray(deployJob.needs) ? deployJob.needs : [deployJob.needs];
      expect(needs).toContain('build');
      expect(deployJob['runs-on']).toBe('ubuntu-latest');
    });

    it('configures github-pages environment with dynamic page URL output reference', () => {
      const deployJob = parsedWorkflow.jobs?.deploy;
      expect(deployJob?.environment, 'Deploy job must specify environment').toBeDefined();
      if (!deployJob?.environment) return;

      expect(deployJob.environment.name).toBe('github-pages');
      expect(deployJob.environment.url).toBe('${{ steps.deployment.outputs.page_url }}');
    });

    it('executes actions/deploy-pages@v4 with deployment step id', () => {
      const steps = parsedWorkflow.jobs?.deploy?.steps ?? [];
      const deployStep = steps.find((s) => s.uses?.startsWith('actions/deploy-pages@v4'));
      expect(deployStep, 'Deploy job must use actions/deploy-pages@v4').toBeDefined();
      if (!deployStep) return;

      expect(deployStep.id).toBe('deployment');
    });
  });

  describe('Pass-to-Complete Gate Verification', () => {
    it('validates pipeline topological symmetry between artifact generation and deployment', () => {
      const buildSteps = parsedWorkflow.jobs?.build?.steps ?? [];
      const deploySteps = parsedWorkflow.jobs?.deploy?.steps ?? [];

      const uploadArtifactStep = buildSteps.find((s) => s.uses?.startsWith('actions/upload-pages-artifact'));
      const deployStep = deploySteps.find((s) => s.uses?.startsWith('actions/deploy-pages'));

      expect(uploadArtifactStep).toBeDefined();
      expect(deployStep).toBeDefined();

      // Symmetrical relationship: Build artifact matches directory built by Vite
      const artifactPath = uploadArtifactStep?.with?.['path'] as string;
      const expectedDistDir = path.resolve(__dirname, '../../dist');
      const resolvedArtifactPath = path.resolve(__dirname, '../../../../', artifactPath);

      expect(resolvedArtifactPath).toBe(expectedDistDir);

      // Environment URL outputs match deployment step ID
      const deployEnvUrl = parsedWorkflow.jobs?.deploy?.environment?.url;
      const deployStepId = deployStep?.id;
      expect(deployEnvUrl).toContain(`steps.${deployStepId}.outputs.page_url`);
    });
  });
});
