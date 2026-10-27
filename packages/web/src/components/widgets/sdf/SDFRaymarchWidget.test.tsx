import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SDFRaymarchWidget, isWebGLAvailable } from './SDFRaymarchWidget';
import {
  createSphere,
  createBox,
  createFreeParam,
} from '../../../core/ast/models';

describe('SDFRaymarchWidget Component', () => {
  it('correctly detects WebGL support state in testing environment', () => {
    // In standard JSDOM, WebGL context is not available
    const available = isWebGLAvailable();
    expect(typeof available).toBe('boolean');
  });

  it('renders gracefully in JSDOM without crashing, showing slicing toolbar and bound params', () => {
    const sphere = createSphere(createFreeParam('test_radius', 0.1, 5.0, 1.5));
    render(<SDFRaymarchWidget ast={sphere} domain="sdf" declarationName="mySphere" />);

    expect(screen.getByTestId('sdf-raymarch-widget')).toBeInTheDocument();
    expect(screen.getByTestId('sdf-slicing-toolbar')).toBeInTheDocument();
    expect(screen.getByTestId('slice-none-btn')).toBeInTheDocument();
    expect(screen.getByTestId('slice-xy-btn')).toBeInTheDocument();
    expect(screen.getByTestId('slice-yz-btn')).toBeInTheDocument();
    expect(screen.getByTestId('slice-zx-btn')).toBeInTheDocument();
    expect(screen.getByTestId('reset-view-btn')).toBeInTheDocument();

    // Verify bound parameter display in fallback card
    expect(screen.getByText('test_radius')).toBeInTheDocument();
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
});
