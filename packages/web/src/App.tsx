import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  LeanEditor,
  type LeanSelection,
  type LeanEditorHandle,
} from './components/editor/LeanEditor';
import { ParameterInspector } from './components/inspector/ParameterInspector';
import { ViewportShell } from './components/widgets';
import { VariationGallery } from './components/gallery';
import { useFreeParamStore } from './core/store/freeParamStore';
import { parseLeanCode, getNodeId, getNodeSpan } from './core/parser/leanParser';
import { getNodeCount } from './core/ast/models';
import {
  findEnclosingASTNode,
  computeSelectionRange,
  formatNodeDisplayName,
  type NodeMatchResult,
} from './core/generator';
import { punctureASTHole } from './core/generator/holePuncturing';
import type { PunctureResult } from './core/generator/types';
import type { CurriculumLevel, VariationCandidate } from './core/pcfg/types';
import { VariationWorkerClient } from './workers/variationWorkerClient';

export interface AppProps {
  initialDomain?: 'sdf' | 'audio' | 'plotter' | 'hardware';
}

const DOMAIN_PRESETS: Record<'sdf' | 'audio' | 'plotter' | 'hardware', string> = {
  sdf: `import SymforgeIr.Types

def myGyroidSDF : SDFNode :=
  SDFNode.intersection
    (SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0))
    (SDFNode.gyroid
      (FreeParam.mk "scale" 3.14 1.0 10.0)
      (FreeParam.mk "thickness" 0.08 0.01 0.5))`,

  audio: `import SymforgeIr.Types

def mySynthAudio : AudioNode :=
  AudioNode.lowpass
    (FreeParam.mk "cutoff" 880.0 20.0 20000.0)
    (FreeParam.mk "resonance" 1.414 0.1 10.0)
    (AudioNode.sine
      (FreeParam.mk "freq" 440.0 20.0 5000.0)
      (FreeParam.mk "phase" 0.0 0.0 6.28))`,

  plotter: `import SymforgeIr.Types

def myHyperdimFunc : UniversalNode :=
  UniversalNode.universal_sdf
    (SDFNode.sphere (FreeParam.mk "radius" 2.0 0.1 5.0))`,

  hardware: `import SymforgeIr.Types

def myAddPipeline : HardwareNode :=
  HardwareNode.add
    (HardwareNode.reg "r0")
    (HardwareNode.imm 42)`,
};

export const App: React.FC<AppProps> = ({ initialDomain = 'sdf' }) => {
  const [activeDomain, setActiveDomain] = useState<'sdf' | 'audio' | 'plotter' | 'hardware'>(initialDomain);
  const [variationCount, setVariationCount] = useState<number>(0);
  const [currentSelection, setCurrentSelection] = useState<LeanSelection | null>(null);
  const [isInspectorCollapsed, setIsInspectorCollapsed] = useState<boolean>(false);

  // Variation Gallery State
  const [isGalleryOpen, setIsGalleryOpen] = useState<boolean>(false);
  const [galleryCandidates, setGalleryCandidates] = useState<VariationCandidate[]>([]);
  const [isGalleryLoading, setIsGalleryLoading] = useState<boolean>(false);
  const [galleryError, setGalleryError] = useState<string | null>(null);
  const [activePunctureResult, setActivePunctureResult] = useState<PunctureResult | null>(null);
  const [selectedGalleryLevel, setSelectedGalleryLevel] = useState<CurriculumLevel | 'ALL'>('ALL');

  const editorRef = useRef<LeanEditorHandle>(null);
  const workerClientRef = useRef<VariationWorkerClient | null>(null);

  if (!workerClientRef.current) {
    workerClientRef.current = new VariationWorkerClient({
      timeoutMs: 3000,
    });
  }

  const [leanCode, setLeanCode] = useState<string>(DOMAIN_PRESETS[initialDomain] || DOMAIN_PRESETS.sdf);

  // Memoize AST parse result from current Lean 4 code
  const parseResult = useMemo(() => {
    try {
      return parseLeanCode(leanCode);
    } catch (err) {
      console.error('[SymForge App] Transient parse failure during code editing:', {
        error: err,
        codeLength: leanCode.length,
        timestamp: Date.now(),
      });
      return null;
    }
  }, [leanCode]);

  // Synchronize AST to FreeParam Reactive State Store on code change
  useEffect(() => {
    if (!parseResult) return;
    try {
      if (parseResult.errors.length === 0 || parseResult.ast) {
        useFreeParamStore.getState().setAST(parseResult.ast, parseResult.declarationName, parseResult.domainType);
      }
    } catch (err) {
      console.error('[SymForge App] Transient error while synchronizing AST to store:', {
        error: err,
        codeLength: leanCode.length,
        timestamp: Date.now(),
      });
    }
  }, [parseResult, leanCode.length]);

  // Resolve minimal enclosing AST node for current Monaco selection
  const scopedMatch = useMemo<NodeMatchResult | null>(() => {
    if (
      !parseResult?.ast ||
      !parseResult?.spans ||
      !currentSelection ||
      !currentSelection.selectedText ||
      currentSelection.selectedText.length === 0
    ) {
      return null;
    }
    const selectionRange = computeSelectionRange(currentSelection, leanCode);
    return findEnclosingASTNode(parseResult.ast, parseResult.spans, selectionRange, leanCode);
  }, [parseResult, currentSelection, leanCode]);

  // Active Scoped Selection badge text (e.g. 🎯 Scope: CylinderNode (child: b) [L5:12 - L5:48])
  const scopedBadgeText = useMemo(() => {
    if (!scopedMatch) return null;
    const nodeName = formatNodeDisplayName(scopedMatch.targetNode);
    const relationStr = scopedMatch.parentRelation ? `child: ${scopedMatch.parentRelation}` : 'root';
    return `🎯 Scope: ${nodeName} (${relationStr}) [L${scopedMatch.span.startLine}:${scopedMatch.span.startCol} - L${scopedMatch.span.endLine}:${scopedMatch.span.endCol}]`;
  }, [scopedMatch]);

  const ast = useFreeParamStore((state) => state.ast);
  const paramOrder = useFreeParamStore((state) => state.paramOrder);
  const nodeCount = useMemo(() => (ast ? getNodeCount(ast) : 0), [ast]);
  const paramCount = paramOrder.length;

  const handleGenerateVariations = useCallback(async () => {
    setVariationCount((prev) => prev + 1);

    if (!parseResult?.ast) {
      console.warn('[SymForge App] Cannot generate variations: AST is null');
      return;
    }

    // Resolve target match: user's highlighted scope or fallback to root node
    let targetMatch = scopedMatch;
    if (!targetMatch) {
      const rootId = getNodeId(parseResult.ast, parseResult) || 'root';
      const rootSpan = getNodeSpan(parseResult.ast, parseResult) || {
        startLine: 1,
        startCol: 1,
        endLine: leanCode.split('\n').length,
        endCol: (leanCode.split('\n').pop()?.length || 0) + 1,
        startOffset: 0,
        endOffset: leanCode.length,
      };
      targetMatch = {
        targetNode: parseResult.ast,
        nodeId: rootId,
        nodePath: [],
        span: rootSpan,
        isExactMatch: true,
        overlapRatio: 1.0,
        parentNode: null,
        parentRelation: null,
        enclosingDeclaration: parseResult.declarationName,
      };
    }

    try {
      const punc = punctureASTHole(parseResult.ast, targetMatch);
      setActivePunctureResult(punc);
      setIsGalleryOpen(true);
      setIsGalleryLoading(true);
      setGalleryError(null);

      const client = workerClientRef.current || new VariationWorkerClient();
      const result = await client.generateVariations(punc); // with timeout guardrail
      setGalleryCandidates(result.candidates);
    } catch (err: any) {
      console.error('[SymForge App] Error generating variations:', err);
      setGalleryError(err?.message || 'Failed to synthesize variations');
    } finally {
      setIsGalleryLoading(false);
    }
  }, [parseResult, scopedMatch, leanCode]);

  const handleRegenerateVariations = useCallback(async () => {
    if (!activePunctureResult) return;
    setIsGalleryLoading(true);
    setGalleryError(null);

    try {
      const client = workerClientRef.current || new VariationWorkerClient();
      const result = await client.generateVariations(activePunctureResult, { // with timeout guardrail
        seed: Math.floor(Math.random() * 1000000),
      });
      setGalleryCandidates(result.candidates);
    } catch (err: any) {
      console.error('[SymForge App] Error regenerating variations:', err);
      setGalleryError(err?.message || 'Failed to regenerate variations');
    } finally {
      setIsGalleryLoading(false);
    }
  }, [activePunctureResult]);

  const handleInjectVariation = useCallback(
    (candidate: VariationCandidate) => {
      if (!activePunctureResult) {
        console.warn('[SymForge App] Cannot inject variation: No active puncture result');
        return;
      }

      const success = editorRef.current?.injectCodeAtSpan(
        activePunctureResult.targetSpan,
        candidate.leanCode
      );

      if (success) {
        setIsGalleryOpen(false);
      }
    },
    [activePunctureResult]
  );

  const handleDomainTabClick = (domain: 'sdf' | 'audio' | 'plotter' | 'hardware') => {
    setActiveDomain(domain);
    const preset = DOMAIN_PRESETS[domain];
    if (preset) {
      setLeanCode(preset);
    }
  };

  const handleSyncAST = () => {
    try {
      const result = parseLeanCode(leanCode);
      useFreeParamStore.getState().setAST(result.ast, result.declarationName, result.domainType);
    } catch (err) {
      console.error('[SymForge App] Manual AST Sync failed with context:', {
        error: err,
        codeLength: leanCode.length,
        timestamp: Date.now(),
      });
    }
  };

  return (
    <div className="app-shell" data-testid="symforge-app-shell">
      {/* Top Application Header */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true">⚡</span>
          <h1 className="brand-title">SymForge Web IDE</h1>
          <span className="brand-badge" data-testid="version-badge">Lean 4 IR v0.1.0</span>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleGenerateVariations}
            data-testid="generate-variation-btn"
          >
            <span>✨ Scoped Variation</span>
            {variationCount > 0 && <span className="badge">({variationCount})</span>}
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleSyncAST}
            data-testid="ast-sync-btn"
            title="Force AST Synchronization"
          >
            🔄 AST Sync
          </button>
          <button
            type="button"
            className={`btn ${!isInspectorCollapsed ? 'btn-primary' : ''}`}
            onClick={() => setIsInspectorCollapsed(!isInspectorCollapsed)}
            data-testid="toggle-inspector-btn"
            title="Toggle Parameter Inspector Drawer"
          >
            <span>🎚️ Parameters</span>
            {paramCount > 0 && <span className="badge">({paramCount})</span>}
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main className="workspace-container">
        {/* Left: Lean 4 Editor Shell */}
        <section className="editor-panel" aria-label="Lean 4 Editor Panel">
          <div className="panel-header">
            <span>Editor: SymforgeIr/Gyroid.lean</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              {scopedBadgeText && (
                <span
                  style={{
                    fontSize: '0.75rem',
                    color: 'var(--accent-amber, #fbbf24)',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    backgroundColor: 'rgba(251, 191, 36, 0.12)',
                    border: '1px solid rgba(251, 191, 36, 0.3)',
                    fontWeight: 500,
                  }}
                  data-testid="scoped-selection-badge"
                  title={`Enclosing Declaration: ${scopedMatch?.enclosingDeclaration || 'anonymous'}`}
                >
                  {scopedBadgeText}
                </span>
              )}
              {scopedMatch && (
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={handleGenerateVariations}
                  data-testid="editor-generate-variations-btn"
                  title="Generate Variations for Scoped Selection"
                  style={{ padding: '2px 8px', fontSize: '0.72rem' }}
                >
                  ⚡ Generate Variations
                </button>
              )}
              {currentSelection && currentSelection.selectedText.length > 0 && (
                <span
                  style={{
                    fontSize: '0.75rem',
                    color: 'var(--accent-cyan)',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    backgroundColor: 'var(--bg-surface)',
                  }}
                  data-testid="editor-selection-indicator"
                >
                  L{currentSelection.startLine}:{currentSelection.startCol}-L{currentSelection.endLine}:{currentSelection.endCol} ({currentSelection.selectedText.length}c)
                </span>
              )}
              <span style={{ color: 'var(--accent-cyan)' }}>Lean 4</span>
            </div>
          </div>
          <div className="panel-body">
            <LeanEditor
              ref={editorRef}
              value={leanCode}
              onChange={setLeanCode}
              onSelectionChange={setCurrentSelection}
              dataTestId="lean-code-editor"
            />
          </div>
        </section>

        {/* Center: Extensible Interactive Domain Viewport Shell */}
        <ViewportShell
          activeDomainTab={activeDomain}
          onDomainTabClick={handleDomainTabClick}
          dataTestId="widget-viewport"
        />

        {/* Right: Parameter Inspector Drawer */}
        <ParameterInspector
          isCollapsed={isInspectorCollapsed}
          onToggleCollapse={() => setIsInspectorCollapsed(!isInspectorCollapsed)}
          dataTestId="parameter-inspector"
        />
      </main>

      {/* Interactive Visual Variation Gallery Sliding Drawer */}
      <VariationGallery
        isOpen={isGalleryOpen}
        onClose={() => setIsGalleryOpen(false)}
        candidates={galleryCandidates}
        isLoading={isGalleryLoading}
        error={galleryError}
        punctureResult={activePunctureResult}
        selectedLevel={selectedGalleryLevel}
        onSelectLevel={setSelectedGalleryLevel}
        onRegenerate={handleRegenerateVariations}
        onInjectVariation={handleInjectVariation}
      />

      {/* Bottom Status Bar */}
      <footer className="status-bar" data-testid="app-status-bar">
        <div className="status-left">
          <div className="status-item">
            <span className="status-dot" aria-hidden="true" />
            <span>Lean 4 Engine: Ready (Client-Side)</span>
          </div>
          <div className="status-item">
            <span>AST Nodes: {nodeCount}</span>
          </div>
          <div className="status-item">
            <span>FreeParams: {paramCount}</span>
          </div>
        </div>
        <div className="status-right">
          <div className="status-item">
            <span>Active Domain: {activeDomain.toUpperCase()}</span>
          </div>
          <div className="status-item">
            <span>PWA Ready</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
