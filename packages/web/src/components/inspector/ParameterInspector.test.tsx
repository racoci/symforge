/**
 * ParameterInspector Component Unit Tests.
 *
 * Validates:
 * 1. Rendering parameter cards, headers, count badges, and domain pills.
 * 2. Search filtering parameters by identifier.
 * 3. Range slider user interaction and store dispatch.
 * 4. Direct precision numeric input user interaction.
 * 5. Single parameter reset and global Reset All functionality.
 * 6. Collapsed state rendering and toggle callbacks.
 * 7. Empty state handling when no parameters are present.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ParameterInspector } from './ParameterInspector';
import { useFreeParamStore } from '../../core/store/freeParamStore';
import { createSphere, createBox, createIntersection, createFreeParam } from '../../core/ast/models';

describe('ParameterInspector Component', () => {
  beforeEach(() => {
    // Reset store before each test
    useFreeParamStore.getState().setAST(null);
  });

  it('renders empty state when no parameters are in the store', () => {
    render(<ParameterInspector />);
    expect(screen.getByTestId('param-empty-state')).toBeInTheDocument();
    expect(screen.getByText(/No continuous FreeParams/i)).toBeInTheDocument();
    expect(screen.getByTestId('param-count-badge')).toHaveTextContent('0');
  });

  it('renders all extracted parameter cards with their initial default values', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 5.0, 1.25));
    const box = createBox(
      createFreeParam('width', 0.1, 10.0, 2.0),
      createFreeParam('height', 0.1, 10.0, 3.0),
      createFreeParam('depth', 0.1, 10.0, 1.0)
    );
    const ast = createIntersection(sphere, box);

    useFreeParamStore.getState().setAST(ast, 'myGyroid', 'SDFNode');

    render(<ParameterInspector />);

    expect(screen.getByTestId('param-count-badge')).toHaveTextContent('4');
    expect(screen.getByText('myGyroid')).toBeInTheDocument();

    // Verify all 4 parameters are present
    expect(screen.getByTestId('param-card-radius')).toBeInTheDocument();
    expect(screen.getByTestId('param-card-width')).toBeInTheDocument();
    expect(screen.getByTestId('param-card-height')).toBeInTheDocument();
    expect(screen.getByTestId('param-card-depth')).toBeInTheDocument();

    // Verify domain badges
    expect(screen.getByTestId('param-domain-radius')).toHaveTextContent('SDF');

    // Verify numeric inputs have default values
    expect(screen.getByTestId('param-input-radius')).toHaveValue(1.25);
    expect(screen.getByTestId('param-input-width')).toHaveValue(2.0);
  });

  it('filters parameters when user types into search input', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 5.0, 1.25));
    const box = createBox(
      createFreeParam('width', 0.1, 10.0, 2.0),
      createFreeParam('height', 0.1, 10.0, 3.0),
      createFreeParam('depth', 0.1, 10.0, 1.0)
    );
    const ast = createIntersection(sphere, box);

    useFreeParamStore.getState().setAST(ast, 'geom', 'SDFNode');

    render(<ParameterInspector />);

    const searchInput = screen.getByTestId('param-search-input');
    fireEvent.change(searchInput, { target: { value: 'rad' } });

    expect(screen.getByTestId('param-card-radius')).toBeInTheDocument();
    expect(screen.queryByTestId('param-card-width')).not.toBeInTheDocument();
    expect(screen.queryByTestId('param-card-height')).not.toBeInTheDocument();
    expect(screen.queryByTestId('param-card-depth')).not.toBeInTheDocument();

    // Clear search using clear button
    const clearBtn = screen.getByTestId('clear-search-btn');
    fireEvent.click(clearBtn);

    expect(screen.getByTestId('param-card-width')).toBeInTheDocument();
  });

  it('updates store when user scrubs continuous range slider', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 5.0, 1.25));
    useFreeParamStore.getState().setAST(sphere, 'sphereDecl', 'SDFNode');

    render(<ParameterInspector />);

    const slider = screen.getByTestId('param-slider-radius');
    fireEvent.change(slider, { target: { value: '3.75' } });

    expect(useFreeParamStore.getState().values.radius).toBe(3.75);
    expect(screen.getByTestId('param-input-radius')).toHaveValue(3.75);
  });

  it('updates store when user enters precision number in numeric input', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 5.0, 1.25));
    useFreeParamStore.getState().setAST(sphere, 'sphereDecl', 'SDFNode');

    render(<ParameterInspector />);

    const numInput = screen.getByTestId('param-input-radius');
    fireEvent.change(numInput, { target: { value: '2.456' } });

    expect(useFreeParamStore.getState().values.radius).toBe(2.456);
  });

  it('resets a single parameter when clicking individual reset button', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 5.0, 1.25));
    useFreeParamStore.getState().setAST(sphere, 'sphereDecl', 'SDFNode');

    render(<ParameterInspector />);

    // Initially reset button is disabled because value is at default
    const resetBtn = screen.getByTestId('param-reset-radius');
    expect(resetBtn).toBeDisabled();

    // Change value
    const slider = screen.getByTestId('param-slider-radius');
    fireEvent.change(slider, { target: { value: '4.2' } });

    expect(resetBtn).not.toBeDisabled();

    // Click reset
    fireEvent.click(resetBtn);

    expect(useFreeParamStore.getState().values.radius).toBe(1.25);
    expect(resetBtn).toBeDisabled();
  });

  it('resets all modified parameters when clicking Reset All', () => {
    const box = createBox(
      createFreeParam('w', 1, 10, 2),
      createFreeParam('h', 1, 10, 4),
      createFreeParam('d', 1, 10, 6)
    );
    useFreeParamStore.getState().setAST(box, 'boxDecl', 'SDFNode');

    render(<ParameterInspector />);

    const resetAllBtn = screen.getByTestId('reset-all-params-btn');
    expect(resetAllBtn).toBeDisabled();

    // Modify values
    fireEvent.change(screen.getByTestId('param-slider-w'), { target: { value: '8' } });
    fireEvent.change(screen.getByTestId('param-slider-h'), { target: { value: '9' } });

    expect(resetAllBtn).not.toBeDisabled();

    // Click Reset All
    fireEvent.click(resetAllBtn);

    expect(useFreeParamStore.getState().values.w).toBe(2);
    expect(useFreeParamStore.getState().values.h).toBe(4);
    expect(resetAllBtn).toBeDisabled();
  });

  it('renders collapsed state when isCollapsed is true and triggers onToggleCollapse', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 5.0, 1.25));
    useFreeParamStore.getState().setAST(sphere, 'sphereDecl', 'SDFNode');

    const toggleSpy = vi.fn();
    render(<ParameterInspector isCollapsed={true} onToggleCollapse={toggleSpy} />);

    expect(screen.getByTestId('parameter-inspector-collapsed')).toBeInTheDocument();
    const expandBtn = screen.getByTestId('expand-inspector-btn');
    expect(expandBtn).toBeInTheDocument();

    fireEvent.click(expandBtn);
    expect(toggleSpy).toHaveBeenCalledTimes(1);
  });
});
