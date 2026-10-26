import React, { useState } from 'react';

export interface AppProps {
  initialDomain?: 'sdf' | 'audio' | 'plotter' | 'hardware';
}

export const App: React.FC<AppProps> = ({ initialDomain = 'sdf' }) => {
  const [activeDomain, setActiveDomain] = useState<'sdf' | 'audio' | 'plotter' | 'hardware'>(initialDomain);
  const [variationCount, setVariationCount] = useState<number>(0);

  const sampleLeanCode = `import SymforgeIr.Types

def myGyroidSDF : SDFNode :=
  SDFNode.intersection
    (SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0))
    (SDFNode.gyroid
      (FreeParam.mk "scale" 3.14 1.0 10.0)
      (FreeParam.mk "thickness" 0.08 0.01 0.5))`;

  const handleGenerateVariations = () => {
    setVariationCount((prev) => prev + 1);
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
          <button type="button" className="btn" data-testid="ast-sync-btn">
            🔄 AST Sync
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main className="workspace-container">
        {/* Left: Lean 4 Editor Shell */}
        <section className="editor-panel" aria-label="Lean 4 Editor Panel">
          <div className="panel-header">
            <span>Editor: SymforgeIr/Gyroid.lean</span>
            <span style={{ color: 'var(--accent-cyan)' }}>Lean 4</span>
          </div>
          <div className="panel-body">
            <pre className="code-placeholder" data-testid="lean-code-editor">
              <code>{sampleLeanCode}</code>
            </pre>
          </div>
        </section>

        {/* Right: Interactive Domain Viewport Shell */}
        <section className="viewport-panel" aria-label="Domain Viewport Panel">
          <div className="panel-header">
            <span>Viewport: Interactive Domain Widget</span>
            <div className="domain-tags" role="tablist">
              <button
                type="button"
                className={`btn ${activeDomain === 'sdf' ? 'btn-primary' : ''}`}
                style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                onClick={() => setActiveDomain('sdf')}
                data-testid="domain-tab-sdf"
              >
                3D SDF
              </button>
              <button
                type="button"
                className={`btn ${activeDomain === 'audio' ? 'btn-primary' : ''}`}
                style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                onClick={() => setActiveDomain('audio')}
                data-testid="domain-tab-audio"
              >
                WebAudio
              </button>
              <button
                type="button"
                className={`btn ${activeDomain === 'plotter' ? 'btn-primary' : ''}`}
                style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                onClick={() => setActiveDomain('plotter')}
                data-testid="domain-tab-plotter"
              >
                Hyperdim
              </button>
            </div>
          </div>
          <div className="widget-stage" data-testid="widget-viewport">
            <div className="stage-card">
              <div className="stage-icon">
                {activeDomain === 'sdf' && '🔮'}
                {activeDomain === 'audio' && '🎛️'}
                {activeDomain === 'plotter' && '📈'}
                {activeDomain === 'hardware' && '⚡'}
              </div>
              <h2 className="stage-title">
                {activeDomain === 'sdf' && '3D WebGL SDF Raymarching'}
                {activeDomain === 'audio' && 'WebAudio DSP Synthesis & FFT'}
                {activeDomain === 'plotter' && 'Generalized Hyperdim Plotter'}
                {activeDomain === 'hardware' && 'Hardware Pipeline Visualizer'}
              </h2>
              <p className="stage-description">
                Interactive client-side domain widget powered by typed Lean 4 AST dispatch and real-time FreeParam reactivity.
              </p>
              <div className="domain-tags">
                <span className="domain-tag">Zero Server Footprint</span>
                <span className="domain-tag">GitHub Pages PWA</span>
                <span className="domain-tag">60 FPS WebGL/Audio</span>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Bottom Status Bar */}
      <footer className="status-bar" data-testid="app-status-bar">
        <div className="status-left">
          <div className="status-item">
            <span className="status-dot" aria-hidden="true" />
            <span>Lean 4 Engine: Ready (Client-Side)</span>
          </div>
          <div className="status-item">
            <span>AST Nodes: 3</span>
          </div>
          <div className="status-item">
            <span>FreeParams: 3</span>
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
