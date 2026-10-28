import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import {
  HyperdimPlotterWidget,
  isWebGLAvailable,
  isCanvasAvailable,
} from './HyperdimPlotterWidget';
import { createFreeParam, createBox } from '../../../core/ast/models';
import { useFreeParamStore } from '../../../core/store/freeParamStore';

describe('HyperdimPlotterWidget Component', () => {
  beforeEach(() => {
    useFreeParamStore.getState().setAST(null);
  });

  it('detects WebGL and Canvas availability in test environment', () => {
    const webgl = isWebGLAvailable();
    const canvas = isCanvasAvailable();
    expect(typeof webgl).toBe('boolean');
    expect(typeof canvas).toBe('boolean');
  });

  it('renders gracefully in JSDOM with complete toolbar and orthogonal overlay', () => {
    render(
      <HyperdimPlotterWidget
        ast={null}
        domain="plotter"
        declarationName="testFunction"
      />
    );

    expect(screen.getByTestId('hyperdim-plotter-widget')).toBeInTheDocument();
    expect(screen.getByTestId('hyperdim-toolbar')).toBeInTheDocument();

    // Toolbar controls
    expect(screen.getByTestId('function-select')).toBeInTheDocument();
    expect(screen.getByTestId('axis-x-select')).toBeInTheDocument();
    expect(screen.getByTestId('axis-y-select')).toBeInTheDocument();
    expect(screen.getByTestId('mode-2d-btn')).toBeInTheDocument();
    expect(screen.getByTestId('mode-3d-btn')).toBeInTheDocument();
    expect(screen.getByTestId('mode-split-btn')).toBeInTheDocument();
    expect(screen.getByTestId('colormap-select')).toBeInTheDocument();
    expect(screen.getByTestId('res-50-btn')).toBeInTheDocument();
    expect(screen.getByTestId('res-100-btn')).toBeInTheDocument();
    expect(screen.getByTestId('res-150-btn')).toBeInTheDocument();
    expect(screen.getByTestId('toggle-drawer-btn')).toBeInTheDocument();
    expect(screen.getByTestId('reset-view-btn')).toBeInTheDocument();

    // Orthogonal drawer overlay
    expect(screen.getByTestId('hyperdim-param-overlay')).toBeInTheDocument();
  });

  it('switches functions via dropdown and updates axis options', () => {
    render(
      <HyperdimPlotterWidget
        ast={null}
        domain="plotter"
        declarationName="testFunction"
      />
    );

    const fnSelect = screen.getByTestId('function-select') as HTMLSelectElement;
    expect(fnSelect.value).toBe('rastrigin-5d');

    // Switch to Rosenbrock 4D
    fireEvent.change(fnSelect, { target: { value: 'rosenbrock-4d' } });
    expect(fnSelect.value).toBe('rosenbrock-4d');

    const axisXSelect = screen.getByTestId('axis-x-select') as HTMLSelectElement;
    expect(axisXSelect.options.length).toBe(4);

    // Switch to Multi-Wave Ripple 5D
    fireEvent.change(fnSelect, { target: { value: 'multi-wave-5d' } });
    expect(fnSelect.value).toBe('multi-wave-5d');
    expect(axisXSelect.options.length).toBe(5);
  });

  it('updates horizontal and vertical axes preventing axis collision', () => {
    render(
      <HyperdimPlotterWidget
        ast={null}
        domain="plotter"
        declarationName="testFunction"
      />
    );

    const axisXSelect = screen.getByTestId('axis-x-select') as HTMLSelectElement;
    const axisYSelect = screen.getByTestId('axis-y-select') as HTMLSelectElement;

    expect(axisXSelect.value).toBe('0');
    expect(axisYSelect.value).toBe('1');

    // Select axis 2 for Horizontal
    fireEvent.change(axisXSelect, { target: { value: '2' } });
    expect(axisXSelect.value).toBe('2');
    expect(axisYSelect.value).toBe('1');

    // If Horizontal selects same as Vertical (1), Vertical swaps to previous Horizontal (2)
    fireEvent.change(axisXSelect, { target: { value: '1' } });
    expect(axisXSelect.value).toBe('1');
    expect(axisYSelect.value).toBe('2');

    // If Vertical selects same as Horizontal (1), Horizontal swaps to 2
    fireEvent.change(axisYSelect, { target: { value: '1' } });
    expect(axisYSelect.value).toBe('1');
    expect(axisXSelect.value).toBe('2');
  });

  it('toggles display modes (2D, 3D, Split) and wireframe', () => {
    render(
      <HyperdimPlotterWidget
        ast={null}
        domain="plotter"
        declarationName="testFunction"
      />
    );

    const btn2D = screen.getByTestId('mode-2d-btn');
    const btn3D = screen.getByTestId('mode-3d-btn');
    const btnSplit = screen.getByTestId('mode-split-btn');

    expect(btn2D).toHaveClass('active');
    expect(screen.queryByTestId('toggle-wireframe-btn')).not.toBeInTheDocument();

    // Switch to 3D Mesh
    fireEvent.click(btn3D);
    expect(btn3D).toHaveClass('active');
    expect(btn2D).not.toHaveClass('active');

    // Wireframe toggle should be visible in 3D mode
    const wireframeBtn = screen.getByTestId('toggle-wireframe-btn');
    expect(wireframeBtn).toBeInTheDocument();
    expect(wireframeBtn).not.toHaveClass('active');
    fireEvent.click(wireframeBtn);
    expect(wireframeBtn).toHaveClass('active');

    // Switch to Split View
    fireEvent.click(btnSplit);
    expect(btnSplit).toHaveClass('active');
    expect(btn3D).not.toHaveClass('active');
  });

  it('changes colormap and resolution settings', () => {
    render(
      <HyperdimPlotterWidget
        ast={null}
        domain="plotter"
        declarationName="testFunction"
      />
    );

    const colormapSelect = screen.getByTestId('colormap-select') as HTMLSelectElement;
    expect(colormapSelect.value).toBe('viridis');

    fireEvent.change(colormapSelect, { target: { value: 'turbo' } });
    expect(colormapSelect.value).toBe('turbo');

    fireEvent.change(colormapSelect, { target: { value: 'magma' } });
    expect(colormapSelect.value).toBe('magma');

    // Resolution buttons
    const btn50 = screen.getByTestId('res-50-btn');
    const btn100 = screen.getByTestId('res-100-btn');
    const btn150 = screen.getByTestId('res-150-btn');

    expect(btn100).toHaveClass('active');
    fireEvent.click(btn50);
    expect(btn50).toHaveClass('active');
    expect(btn100).not.toHaveClass('active');

    fireEvent.click(btn150);
    expect(btn150).toHaveClass('active');
  });

  it('allows toggling orthogonal drawer, scrubbing coordinates, and resetting', () => {
    render(
      <HyperdimPlotterWidget
        ast={null}
        domain="plotter"
        declarationName="testFunction"
      />
    );

    expect(screen.getByTestId('hyperdim-param-overlay')).toBeInTheDocument();
    const toggleDrawerBtn = screen.getByTestId('toggle-drawer-btn');
    expect(toggleDrawerBtn).toHaveClass('active');

    // Close via close button in drawer
    const closeBtn = screen.getByTestId('close-drawer-btn');
    fireEvent.click(closeBtn);
    expect(screen.queryByTestId('hyperdim-param-overlay')).not.toBeInTheDocument();
    expect(toggleDrawerBtn).not.toHaveClass('active');

    // Reopen via toolbar button
    fireEvent.click(toggleDrawerBtn);
    expect(screen.getByTestId('hyperdim-param-overlay')).toBeInTheDocument();
    expect(toggleDrawerBtn).toHaveClass('active');

    // Scrub an orthogonal slider (e.g., x_2)
    const sliderX2 = screen.getByTestId('orthogonal-slider-x_2');
    expect(sliderX2).toBeInTheDocument();
    fireEvent.change(sliderX2, { target: { value: '2.5' } });
    expect(screen.getByTestId('orthogonal-val-x_2')).toHaveTextContent('2.50');

    // Reset single coordinate x_2
    const resetX2Btn = screen.getByTestId('reset-orthogonal-x_2');
    fireEvent.click(resetX2Btn);
    expect(screen.getByTestId('orthogonal-val-x_2')).toHaveTextContent('0.00');

    // Scrub again and click Reset Slices
    fireEvent.change(sliderX2, { target: { value: '3.14' } });
    expect(screen.getByTestId('orthogonal-val-x_2')).toHaveTextContent('3.14');
    const resetAllSlicesBtn = screen.getByTestId('reset-all-slices-btn');
    fireEvent.click(resetAllSlicesBtn);
    expect(screen.getByTestId('orthogonal-val-x_2')).toHaveTextContent('0.00');

    // Click Reset View toolbar button
    const resetViewBtn = screen.getByTestId('reset-view-btn');
    expect(() => fireEvent.click(resetViewBtn)).not.toThrow();
  });

  it('integrates AST-derived function when AST is provided', () => {
    const box = createBox(
      createFreeParam('box_width', 0.5, 6.0, 2.0),
      createFreeParam('box_height', 0.5, 6.0, 1.5),
      createFreeParam('box_depth', 0.5, 6.0, 3.0)
    );

    render(
      <HyperdimPlotterWidget
        ast={box}
        domain="plotter"
        declarationName="myProceduralBox"
      />
    );

    const fnSelect = screen.getByTestId('function-select') as HTMLSelectElement;
    // AST function should be in the dropdown
    const options = Array.from(fnSelect.options).map((o) => o.text);
    expect(options.some((txt) => txt.includes('myProceduralBox'))).toBe(true);

    // Switch to AST function
    const astOpt = Array.from(fnSelect.options).find((o) => o.text.includes('myProceduralBox'));
    expect(astOpt).toBeDefined();
    fireEvent.change(fnSelect, { target: { value: astOpt!.value } });

    // Dimension names should match AST parameters
    const axisXSelect = screen.getByTestId('axis-x-select') as HTMLSelectElement;
    const dimNames = Array.from(axisXSelect.options).map((o) => o.text);
    expect(dimNames).toContain('box_width');
    expect(dimNames).toContain('box_height');
    expect(dimNames).toContain('box_depth');
  });

  it('renders canvas elements when canvas 2D is mocked in JSDOM', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    const mockCtx = {
      clientWidth: 600,
      clientHeight: 500,
      width: 600,
      height: 500,
      createImageData: vi.fn().mockReturnValue({
        data: new Uint8ClampedArray(100 * 100 * 4),
      }),
      putImageData: vi.fn(),
      drawImage: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      setLineDash: vi.fn(),
      imageSmoothingEnabled: false,
    };

    HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((type: string) => {
      if (type === '2d') return mockCtx;
      return null;
    }) as any;

    try {
      render(
        <HyperdimPlotterWidget
          ast={null}
          domain="plotter"
          declarationName="mockedTest"
        />
      );

      expect(screen.getByTestId('hyperdim-view-wrapper')).toBeInTheDocument();
      const canvas = screen.getByTestId('canvas-2d');
      expect(canvas).toBeInTheDocument();

      // Trigger mouse move to exercise crosshair calculation
      fireEvent.mouseMove(canvas, { clientX: 150, clientY: 100 });
      expect(screen.getByTestId('hyperdim-readout-badge')).toBeInTheDocument();

      fireEvent.mouseLeave(canvas);
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });
});
