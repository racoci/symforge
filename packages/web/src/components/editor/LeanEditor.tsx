import React, { useRef, useEffect, useCallback } from 'react';
import Editor, { type OnMount, type BeforeMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import {
  registerLean4Language,
  LEAN4_LANGUAGE_ID,
  LEAN4_THEME_NAME,
} from './leanLanguage';

export interface LeanSelection {
  startLine: number;
  startCol: number;
  endLine: number;
  endCol: number;
  selectedText: string;
}

export interface LeanEditorProps {
  value: string;
  onChange?: (val: string) => void;
  onSelectionChange?: (sel: LeanSelection) => void;
  readOnly?: boolean;
  height?: string | number;
  width?: string | number;
  className?: string;
  theme?: string;
  dataTestId?: string;
}

export const LeanEditor: React.FC<LeanEditorProps> = ({
  value,
  onChange,
  onSelectionChange,
  readOnly = false,
  height = '100%',
  width = '100%',
  className = '',
  theme = LEAN4_THEME_NAME,
  dataTestId = 'lean-code-editor',
}) => {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const selectionDisposableRef = useRef<{ dispose: () => void } | null>(null);

  const handleBeforeMount: BeforeMount = useCallback((monaco) => {
    registerLean4Language(monaco);
  }, []);

  const handleEditorMount: OnMount = useCallback(
    (editorInstance) => {
      editorRef.current = editorInstance;

      // Clean up previous listener if any
      if (selectionDisposableRef.current) {
        selectionDisposableRef.current.dispose();
      }

      // Track cursor selection changes
      const disposable = editorInstance.onDidChangeCursorSelection((e) => {
        if (!onSelectionChange) return;

        const selection = e.selection;
        const model = editorInstance.getModel();
        if (!model) return;

        const startLine = selection.startLineNumber;
        const startCol = selection.startColumn;
        const endLine = selection.endLineNumber;
        const endCol = selection.endColumn;
        const selectedText = model.getValueInRange(selection);

        onSelectionChange({
          startLine,
          startCol,
          endLine,
          endCol,
          selectedText,
        });
      });

      selectionDisposableRef.current = disposable;
    },
    [onSelectionChange],
  );

  useEffect(() => {
    return () => {
      if (selectionDisposableRef.current) {
        selectionDisposableRef.current.dispose();
        selectionDisposableRef.current = null;
      }
    };
  }, []);

  const editorOptions: editor.IStandaloneEditorConstructionOptions = {
    minimap: { enabled: false },
    fontSize: 13,
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace",
    lineNumbers: 'on',
    scrollBeyondLastLine: false,
    automaticLayout: true,
    readOnly,
    wordWrap: 'on',
    tabSize: 2,
    renderLineHighlight: 'all',
    cursorBlinking: 'smooth',
    glyphMargin: false,
    padding: { top: 8, bottom: 8 },
  };

  return (
    <div
      className={`lean-editor-container ${className}`}
      data-testid={dataTestId}
      style={{
        position: 'relative',
        width,
        height,
        minHeight: '200px',
        overflow: 'hidden',
        backgroundColor: 'var(--bg-secondary)',
      }}
    >
      <Editor
        height="100%"
        width="100%"
        language={LEAN4_LANGUAGE_ID}
        theme={theme}
        value={value}
        onChange={(val) => onChange?.(val ?? '')}
        beforeMount={handleBeforeMount}
        onMount={handleEditorMount}
        options={editorOptions}
        loading={
          <div
            className="code-placeholder"
            data-testid="monaco-loading-fallback"
            style={{ padding: '16px', color: 'var(--text-muted)' }}
          >
            <code>{value}</code>
          </div>
        }
      />
      {/* Hidden DOM element ensuring code content is accessible in headless test environments */}
      <div
        style={{ display: 'none' }}
        aria-hidden="true"
        data-testid="lean-editor-inner-content"
      >
        {value}
      </div>
    </div>
  );
};

export default LeanEditor;
