import { createRef } from 'react';
import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import LeanEditor, {
  type LeanEditorHandle,
  replaceSpanInText,
} from './LeanEditor';
import type { SourceSpan } from '../../core/parser/types';

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

  describe('1-Click Code Injection Handle & Undo/Redo Preservation', () => {
    it('replaces text span and triggers onChange in fallback mode', () => {
      const handleChange = vi.fn();
      const editorRef = createRef<LeanEditorHandle>();

      render(
        <LeanEditor
          ref={editorRef}
          value={sampleCode}
          onChange={handleChange}
        />
      );

      expect(editorRef.current).not.toBeNull();

      const span: SourceSpan = {
        startLine: 2,
        startCol: 3,
        endLine: 2,
        endCol: 53,
        startOffset: sampleCode.indexOf('SDFNode.sphere'),
        endOffset: sampleCode.length,
      };

      const replacement = 'SDFNode.box (FreeParam.mk "size" 2.0 0.1 5.0)';

      act(() => {
        const ok = editorRef.current?.injectCodeAtSpan(span, replacement);
        expect(ok).toBe(true);
      });

      expect(handleChange).toHaveBeenCalledTimes(1);
      const updatedText = handleChange.mock.calls[0][0];
      expect(updatedText).toContain('SDFNode.box');
      expect(updatedText).not.toContain('SDFNode.sphere');
    });

    it('performs undo reverting injected code in fallback mode', () => {
      let currentVal = sampleCode;
      const handleChange = vi.fn((val: string) => {
        currentVal = val;
      });
      const editorRef = createRef<LeanEditorHandle>();

      const { rerender } = render(
        <LeanEditor
          ref={editorRef}
          value={currentVal}
          onChange={handleChange}
        />
      );

      const span: SourceSpan = {
        startLine: 2,
        startCol: 3,
        endLine: 2,
        endCol: 53,
        startOffset: sampleCode.indexOf('SDFNode.sphere'),
        endOffset: sampleCode.length,
      };

      act(() => {
        editorRef.current?.injectCodeAtSpan(span, 'SDFNode.cylinder');
      });

      expect(currentVal).toContain('SDFNode.cylinder');

      // Rerender with updated value
      rerender(
        <LeanEditor
          ref={editorRef}
          value={currentVal}
          onChange={handleChange}
        />
      );

      // Perform Undo
      act(() => {
        editorRef.current?.undo();
      });

      expect(currentVal).toBe(sampleCode);
    });

    it('invokes Monaco executeEdits and pushUndoStop when editor instance is mounted', () => {
      const handleChange = vi.fn();
      const editorRef = createRef<LeanEditorHandle>();

      render(
        <LeanEditor
          ref={editorRef}
          value={sampleCode}
          onChange={handleChange}
        />
      );

      const mockExecuteEdits = vi.fn();
      const mockPushUndoStop = vi.fn();
      const mockFocus = vi.fn();
      const mockGetValue = vi.fn(() => 'def mySphere : SDFNode :=\n  SDFNode.torus');
      const mockUndoRun = vi.fn();

      const mockEditor: any = {
        focus: mockFocus,
        executeEdits: mockExecuteEdits,
        pushUndoStop: mockPushUndoStop,
        getValue: mockGetValue,
        getAction: vi.fn((actionId: string) => {
          if (actionId === 'undo') return { run: mockUndoRun };
          return null;
        }),
        trigger: vi.fn(),
      };

      // Mount mock editor onto handle
      act(() => {
        editorRef.current?._setEditorInstance?.(mockEditor);
      });

      const span: SourceSpan = {
        startLine: 2,
        startCol: 3,
        endLine: 2,
        endCol: 53,
        startOffset: 26,
        endOffset: 77,
      };

      act(() => {
        const ok = editorRef.current?.injectCodeAtSpan(span, 'SDFNode.torus');
        expect(ok).toBe(true);
      });

      expect(mockFocus).toHaveBeenCalled();
      expect(mockExecuteEdits).toHaveBeenCalledWith('symforge-scoped-variation', [
        {
          range: {
            startLineNumber: 2,
            startColumn: 3,
            endLineNumber: 2,
            endColumn: 53,
          },
          text: 'SDFNode.torus',
          forceMoveMarkers: true,
        },
      ]);
      expect(mockPushUndoStop).toHaveBeenCalled();
      expect(handleChange).toHaveBeenCalledWith('def mySphere : SDFNode :=\n  SDFNode.torus');

      // Test undo execution
      act(() => {
        editorRef.current?.undo();
      });
      expect(mockUndoRun).toHaveBeenCalled();
    });

    it('replaceSpanInText handles both offset-based and line-col-based slicing', () => {
      const code = 'line 1\nline 2 target\nline 3';
      const spanWithOffsets: SourceSpan = {
        startLine: 2,
        startCol: 8,
        endLine: 2,
        endCol: 14,
        startOffset: 14,
        endOffset: 20,
      };

      const result1 = replaceSpanInText(code, spanWithOffsets, 'REPLACED');
      expect(result1).toBe('line 1\nline 2 REPLACED\nline 3');

      // Span without valid offsets (falls back to line/col)
      const spanWithoutOffsets: SourceSpan = {
        startLine: 2,
        startCol: 8,
        endLine: 2,
        endCol: 14,
        startOffset: -1,
        endOffset: -1,
      };

      const result2 = replaceSpanInText(code, spanWithoutOffsets, 'REPLACED');
      expect(result2).toBe('line 1\nline 2 REPLACED\nline 3');
    });
  });
});
