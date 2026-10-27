import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import App from './App';

describe('SymForge Web IDE App Shell', () => {
  it('renders application brand and version badge', () => {
    render(<App />);
    expect(screen.getByText('SymForge Web IDE')).toBeInTheDocument();
    expect(screen.getByTestId('version-badge')).toHaveTextContent('Lean 4 IR v0.1.0');
  });

  it('renders Lean 4 editor placeholder and sample Gyroid IR', () => {
    render(<App />);
    const editor = screen.getByTestId('lean-code-editor');
    expect(editor).toBeInTheDocument();
    expect(editor).toHaveTextContent('def myGyroidSDF : SDFNode :=');
    expect(editor).toHaveTextContent('FreeParam.mk "radius" 1.25 0.5 3.0');
  });

  it('renders the interactive domain viewport with default SDF widget', () => {
    render(<App />);
    const viewport = screen.getByTestId('widget-viewport');
    expect(viewport).toBeInTheDocument();
    expect(screen.getByText('3D WebGL SDF Raymarching')).toBeInTheDocument();
  });

  it('switches domain viewports when tabs are clicked', () => {
    render(<App />);

    // Switch to WebAudio
    const audioTab = screen.getByTestId('domain-tab-audio');
    fireEvent.click(audioTab);
    expect(screen.getByText('WebAudio DSP Synthesis & FFT')).toBeInTheDocument();

    // Switch to Hyperdim
    const plotterTab = screen.getByTestId('domain-tab-plotter');
    fireEvent.click(plotterTab);
    expect(screen.getByText('Generalized Hyperdim Plotter')).toBeInTheDocument();

    // Switch back to 3D SDF
    const sdfTab = screen.getByTestId('domain-tab-sdf');
    fireEvent.click(sdfTab);
    expect(screen.getByText('3D WebGL SDF Raymarching')).toBeInTheDocument();
  });

  it('increments variation count when clicking scoped variation button', () => {
    render(<App />);
    const variationBtn = screen.getByTestId('generate-variation-btn');
    expect(variationBtn).toHaveTextContent('✨ Scoped Variation');

    fireEvent.click(variationBtn);
    expect(variationBtn).toHaveTextContent('(1)');

    fireEvent.click(variationBtn);
    expect(variationBtn).toHaveTextContent('(2)');
  });

  it('displays active status indicators in status bar', () => {
    render(<App />);
    const statusBar = screen.getByTestId('app-status-bar');
    expect(statusBar).toBeInTheDocument();
    expect(screen.getByText(/Lean 4 Engine: Ready/i)).toBeInTheDocument();
    expect(screen.getByText(/Active Domain: SDF/i)).toBeInTheDocument();
  });

  it('renders ParameterInspector and allows collapsing and expanding', () => {
    render(<App />);
    expect(screen.getByTestId('parameter-inspector')).toBeInTheDocument();
    expect(screen.getByTestId('toggle-inspector-btn')).toBeInTheDocument();

    // Click toggle button to collapse inspector
    fireEvent.click(screen.getByTestId('toggle-inspector-btn'));
    expect(screen.getByTestId('parameter-inspector-collapsed')).toBeInTheDocument();

    // Click expand button to restore inspector
    fireEvent.click(screen.getByTestId('expand-inspector-btn'));
    expect(screen.getByTestId('parameter-inspector')).toBeInTheDocument();
  });

  it('synchronizes AST and parameters on AST sync button click', () => {
    render(<App />);
    const syncBtn = screen.getByTestId('ast-sync-btn');
    expect(syncBtn).toBeInTheDocument();
    fireEvent.click(syncBtn);
    expect(screen.getByTestId('param-count-badge')).toBeInTheDocument();
  });
});
