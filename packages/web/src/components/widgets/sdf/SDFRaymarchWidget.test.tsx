import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SDFRaymarchWidget, isWebGLAvailable } from './SDFRaymarchWidget';
import {
  createSphere,
  createBox,
  createFreeParam,
} from '../../../core/ast/models';
import { useFreeParamStore } from '../../../core/store/freeParamStore';

describe('SDFRaymarchWidget Component', () => {
  beforeEach(() => {
    useFreeParamStore.getState().setAST(null);
  });

  it('correctly detects WebGL support state in testing environment', () => {
    // In standard JSDOM, WebGL context is not available
    const available = isWebGLAvailable();
    expect(typeof available).toBe('boolean');
  });

  it('renders gracefully in JSDOM without crashing, showing slicing toolbar, parameter overlay, and bound params', () => {
    const sphere = createSphere(createFreeParam('test_radius', 0.1, 5.0, 1.5));
    render(<SDFRaymarchWidget ast={sphere} domain="sdf" declarationName="mySphere" />);

    expect(screen.getByTestId('sdf-raymarch-widget')).toBeInTheDocument();
    expect(screen.getByTestId('sdf-slicing-toolbar')).toBeInTheDocument();
    expect(screen.getByTestId('slice-none-btn')).toBeInTheDocument();
    expect(screen.getByTestId('slice-xy-btn')).toBeInTheDocument();
    expect(screen.getByTestId('slice-yz-btn')).toBeInTheDocument();
    expect(screen.getByTestId('slice-zx-btn')).toBeInTheDocument();
    expect(screen.getByTestId('reset-view-btn')).toBeInTheDocument();

    // Verify parameter overlay is rendered by default
    expect(screen.getByTestId('sdf-param-overlay')).toBeInTheDocument();
    expect(screen.getByTestId('param-item-test_radius')).toBeInTheDocument();

    // Verify bound parameter text exists
    const radiusElements = screen.getAllByText('test_radius');
    expect(radiusElements.length).toBeGreaterThanOrEqual(1);
  });

  it('toggles slicing planes and shows offset slider when plane is active', () => {
    const box = createBox(
      createFreeParam('w', 0.1, 5.0, 1.0),
      createFreeParam('h', 0.1, 5.0, 2.0),
      createFreeParam('d', 0.1, 5.0, 0.5)
    );
    render(<SDFRaymarchWidget ast={box} domain="sdf" declarationName="myBox" />);

    // Initially slice is None; offset slider is not visible
    expect(screen.queryByTestId('slice-offset-slider')).not.toBeInTheDocument();

    // Click Slice: XY
    fireEvent.click(screen.getByTestId('slice-xy-btn'));
    expect(screen.getByTestId('slice-xy-btn')).toHaveClass('active');
    expect(screen.getByTestId('slice-offset-slider')).toBeInTheDocument();

    // Adjust slider
    const slider = screen.getByTestId('slice-offset-slider');
    fireEvent.change(slider, { target: { value: '1.25' } });
    expect(screen.getByText('Offset: 1.25')).toBeInTheDocument();

    // Click Slice: YZ
    fireEvent.click(screen.getByTestId('slice-yz-btn'));
    expect(screen.getByTestId('slice-yz-btn')).toHaveClass('active');
    expect(screen.getByTestId('slice-xy-btn')).not.toHaveClass('active');

    // Click Slice: None to disable slicing
    fireEvent.click(screen.getByTestId('slice-none-btn'));
    expect(screen.getByTestId('slice-none-btn')).toHaveClass('active');
    expect(screen.queryByTestId('slice-offset-slider')).not.toBeInTheDocument();
  });

  it('handles reset view button click without throwing', () => {
    const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
    render(<SDFRaymarchWidget ast={sphere} domain="sdf" declarationName="mySphere" />);

    const resetBtn = screen.getByTestId('reset-view-btn');
    expect(() => fireEvent.click(resetBtn)).not.toThrow();
  });

  it('supports toggling the in-viewport parameter scrubber drawer overlay', () => {
    const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
    render(<SDFRaymarchWidget ast={sphere} domain="sdf" declarationName="mySphere" />);

    // Initially open
    expect(screen.getByTestId('sdf-param-overlay')).toBeInTheDocument();
    const toggleBtn = screen.getByTestId('toggle-parameters-btn');
    expect(toggleBtn).toHaveClass('active');

    // Close via close button in overlay header
    const closeBtn = screen.getByTestId('close-parameters-btn');
    fireEvent.click(closeBtn);
    expect(screen.queryByTestId('sdf-param-overlay')).not.toBeInTheDocument();
    expect(toggleBtn).not.toHaveClass('active');

    // Reopen via toolbar button
    fireEvent.click(toggleBtn);
    expect(screen.getByTestId('sdf-param-overlay')).toBeInTheDocument();
    expect(toggleBtn).toHaveClass('active');
  });

  it('allows interactive parameter scrubbing and resetting from the overlay', () => {
    const sphere = createSphere(createFreeParam('radius', 0.1, 5.0, 1.0));
    // Set AST on store so store has the param
    useFreeParamStore.getState().setAST(sphere, 'mySphere');

    render(<SDFRaymarchWidget ast={sphere} domain="sdf" declarationName="mySphere" />);

    // Check slider
    const slider = screen.getByTestId('param-slider-radius');
    expect(slider).toBeInTheDocument();
    expect(slider).toHaveValue('1');

    // Change slider value
    fireEvent.change(slider, { target: { value: '2.75' } });
    expect(useFreeParamStore.getState().values.radius).toBe(2.75);
    expect(screen.getByTestId('param-val-radius')).toHaveTextContent('2.75');

    // Reset single param
    const resetParamBtn = screen.getByTestId('reset-param-radius');
    fireEvent.click(resetParamBtn);
    expect(useFreeParamStore.getState().values.radius).toBe(1.0);
    expect(screen.getByTestId('param-val-radius')).toHaveTextContent('1.00');

    // Change again and click Reset All
    fireEvent.change(slider, { target: { value: '3.5' } });
    expect(useFreeParamStore.getState().values.radius).toBe(3.5);
    const resetAllBtn = screen.getByTestId('reset-all-params-btn');
    fireEvent.click(resetAllBtn);
    expect(useFreeParamStore.getState().values.radius).toBe(1.0);
  });
});
