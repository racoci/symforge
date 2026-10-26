import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import LeanEditor from './LeanEditor';

describe('LeanEditor Component', () => {
  const sampleCode = `def mySphere : SDFNode :=
  SDFNode.sphere (FreeParam.mk "radius" 1.0 0.1 5.0)`;

  it('renders editor container with fallback code view', () => {
    render(<LeanEditor value={sampleCode} />);

    const editorEl = screen.getByTestId('lean-code-editor');
    expect(editorEl).toBeInTheDocument();
    expect(screen.getByTestId('lean-editor-inner-content').textContent).toBe(sampleCode);
    expect(screen.getByTestId('monaco-loading-fallback')).toHaveTextContent('def mySphere : SDFNode :=');
  });

  it('respects custom height and width props', () => {
    render(<LeanEditor value={sampleCode} height="400px" width="800px" />);

    const editorEl = screen.getByTestId('lean-code-editor');
    expect(editorEl).toHaveStyle({ height: '400px', width: '800px' });
  });

  it('accepts onSelectionChange prop callback', () => {
    const handleSelectionChange = vi.fn();
    render(<LeanEditor value={sampleCode} onSelectionChange={handleSelectionChange} />);

    expect(screen.getByTestId('lean-code-editor')).toBeInTheDocument();
  });
});
