import React, { useState, useEffect, useMemo } from 'react';
import { LeanEditor, type LeanSelection } from './components/editor/LeanEditor';
import { ParameterInspector } from './components/inspector/ParameterInspector';
import { ViewportShell } from './components/widgets';
import { useFreeParamStore } from './core/store/freeParamStore';
import { parseLeanCode } from './core/parser/leanParser';
import { getNodeCount } from './core/ast/models';

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

  const [leanCode, setLeanCode] = useState<string>(DOMAIN_PRESETS[initialDomain] || DOMAIN_PRESETS.sdf);

  // Synchronize AST to FreeParam Reactive State Store on code change
  useEffect(() => {
    try {
      const result = parseLeanCode(leanCode);
      if (result.errors.length === 0) {
        useFreeParamStore.getState().setAST(result.ast, result.declarationName, result.domainType);
      } else if (result.ast) {
        useFreeParamStore.getState().setAST(result.ast, result.declarationName, result.domainType);
      }
    } catch (err) {
      console.error('[SymForge App] Transient parse error while synchronizing AST:', {
        error: err,
        codeLength: leanCode.length,
        timestamp: Date.now(),
      });
    }
  }, [leanCode]);

  const ast = useFreeParamStore((state) => state.ast);
  const paramOrder = useFreeParamStore((state) => state.paramOrder);
  const nodeCount = useMemo(() => (ast ? getNodeCount(ast) : 0), [ast]);
  const paramCount = paramOrder.length;

  const handleGenerateVariations = () => {
    setVariationCount((prev) => prev + 1);
  };

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
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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
