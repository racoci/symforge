import React, { useState, useMemo, useCallback } from 'react';
import type { DomainWidgetProps } from './types';
import type { ASTNode, FreeParam } from '../../core/ast/types';
import {
  getChildren,
  isFreeParam,
  isSDFNode,
  isAudioNode,
  isHardwareNode,
  isUniversalNode,
} from '../../core/ast/models';

interface ASTTreeNodeProps {
  node: ASTNode;
  path: string;
  depth: number;
  expandedPaths: Set<string>;
  onToggleExpand: (path: string) => void;
  onParameterChange?: (id: string, value: number) => void;
}

const MAX_TREE_DEPTH = 64;

function getNodeDomainColor(node: ASTNode): string {
  if (isSDFNode(node)) return 'var(--accent-cyan)';
  if (isAudioNode(node)) return 'var(--accent-indigo)';
  if (isHardwareNode(node)) return 'var(--accent-emerald)';
  if (isUniversalNode(node)) return 'var(--accent-violet)';
  return 'var(--text-secondary)';
}

function getNodeDomainBadge(node: ASTNode): string {
  if (isSDFNode(node)) return 'SDF';
  if (isAudioNode(node)) return 'Audio';
  if (isHardwareNode(node)) return 'Hardware';
  if (isUniversalNode(node)) return 'Universal';
  return 'Generic';
}

interface NodeField {
  key: string;
  value: unknown;
  isParam: boolean;
}

function extractNodeFields(node: ASTNode): NodeField[] {
  const fields: NodeField[] = [];
  const excludedKeys = new Set([
    'kind',
    'child',
    'children',
    'a',
    'b',
    'input',
    'input1',
    'input2',
    'gate',
    'first',
    'second',
    'src',
    'src1',
    'src2',
    'node',
  ]);

  for (const [key, val] of Object.entries(node)) {
    if (excludedKeys.has(key)) continue;
    if (isFreeParam(val)) {
      fields.push({ key, value: val, isParam: true });
    } else if (
      typeof val === 'number' ||
      typeof val === 'string' ||
      typeof val === 'boolean'
    ) {
      fields.push({ key, value: val, isParam: false });
    }
  }

  return fields;
}

const ASTTreeNode: React.FC<ASTTreeNodeProps> = React.memo(
  ({ node, path, depth, expandedPaths, onToggleExpand, onParameterChange }) => {
    const isExpanded = expandedPaths.has(path);
    const children = useMemo(() => getChildren(node), [node]);
    const hasChildren = children.length > 0;
    const fields = useMemo(() => extractNodeFields(node), [node]);
    const domainColor = getNodeDomainColor(node);
    const domainBadge = getNodeDomainBadge(node);

    if (depth >= MAX_TREE_DEPTH) {
      return (
        <div className="ast-node-capped" style={{ paddingLeft: `${depth * 16}px` }}>
          <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            [Max recursion depth ({MAX_TREE_DEPTH}) reached]
          </span>
        </div>
      );
    }

    return (
      <div className="ast-tree-row-wrapper" data-testid={`ast-node-${path}`}>
        <div
          className="ast-tree-node-row"
          style={{ paddingLeft: `${Math.min(depth * 20, 240)}px` }}
        >
          {/* Expand/Collapse Toggle or Indent Spacer */}
          {hasChildren ? (
            <button
              type="button"
              className="ast-expand-btn"
              onClick={() => onToggleExpand(path)}
              aria-label={isExpanded ? 'Collapse node' : 'Expand node'}
              data-testid={`toggle-node-${path}`}
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          ) : (
            <span className="ast-leaf-spacer" aria-hidden="true">•</span>
          )}

          {/* Node Kind and Domain Badge */}
          <div className="ast-node-header">
            <span
              className="ast-kind-badge"
              style={{ color: domainColor, borderColor: domainColor }}
            >
              {node.kind}
            </span>
            <span className="ast-domain-tag">{domainBadge}</span>
            {hasChildren && (
              <span className="ast-child-count">
                ({children.length} {children.length === 1 ? 'child' : 'children'})
              </span>
            )}
          </div>
        </div>

        {/* Node Properties / FreeParam Sliders */}
        {fields.length > 0 && (
          <div
            className="ast-node-fields"
            style={{ paddingLeft: `${Math.min(depth * 20 + 28, 268)}px` }}
          >
            {fields.map(({ key, value, isParam }) => {
              if (isParam) {
                const param = value as FreeParam;
                return (
                  <div key={key} className="ast-field-param-row">
                    <span className="ast-field-key">{key}:</span>
                    <span className="ast-param-id">"{param.id}"</span>
                    <span className="ast-param-range">
                      [{param.minVal}..{param.maxVal}]
                    </span>
                    <input
                      type="range"
                      min={param.minVal}
                      max={param.maxVal}
                      step={(param.maxVal - param.minVal) / 100 || 0.01}
                      defaultValue={param.defaultVal}
                      className="ast-mini-slider"
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        onParameterChange?.(param.id, val);
                      }}
                      data-testid={`ast-param-slider-${param.id}`}
                    />
                    <span className="ast-param-val">{param.defaultVal}</span>
                  </div>
                );
              }

              return (
                <div key={key} className="ast-field-primitive-row">
                  <span className="ast-field-key">{key}:</span>
                  <span className="ast-field-val">{String(value)}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Recursive Children Rendering */}
        {hasChildren && isExpanded && (
          <div className="ast-children-container">
            {children.map((child, idx) => (
              <ASTTreeNode
                key={`${path}-${idx}`}
                node={child}
                path={`${path}.${idx}`}
                depth={depth + 1}
                expandedPaths={expandedPaths}
                onToggleExpand={onToggleExpand}
                onParameterChange={onParameterChange}
              />
            ))}
          </div>
        )}
      </div>
    );
  }
);

ASTTreeNode.displayName = 'ASTTreeNode';

/**
 * Generic AST Tree Inspector Widget.
 *
 * Provides a universal, interactive syntax tree explorer for inspecting any ASTNode,
 * expanding/collapsing sub-branches, and interactively adjusting parameters.
 */
export const GenericASTWidget: React.FC<DomainWidgetProps<ASTNode>> = ({
  ast,
  domain,
  declarationName,
  onParameterChange,
}) => {
  // Collect all paths up to depth 3 for initial expansion
  const initialExpanded = useMemo(() => {
    const paths = new Set<string>();
    const traverse = (node: ASTNode, currentPath: string, depth: number) => {
      if (depth > 3) return;
      paths.add(currentPath);
      const kids = getChildren(node);
      kids.forEach((k, idx) => traverse(k, `${currentPath}.${idx}`, depth + 1));
    };
    if (ast) {
      traverse(ast, '0', 0);
    }
    return paths;
  }, [ast]);

  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(initialExpanded);

  const handleToggleExpand = useCallback((path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const handleExpandAll = useCallback(() => {
    const allPaths = new Set<string>();
    const traverse = (node: ASTNode, currentPath: string, depth: number) => {
      if (depth >= MAX_TREE_DEPTH) return;
      allPaths.add(currentPath);
      const kids = getChildren(node);
      kids.forEach((k, idx) => traverse(k, `${currentPath}.${idx}`, depth + 1));
    };
    if (ast) {
      traverse(ast, '0', 0);
    }
    setExpandedPaths(allPaths);
  }, [ast]);

  const handleCollapseAll = useCallback(() => {
    setExpandedPaths(new Set(['0']));
  }, []);

  const [copySuccess, setCopySuccess] = useState<boolean>(false);

  const handleCopyJSON = useCallback(() => {
    if (!ast) return;
    try {
      navigator.clipboard.writeText(JSON.stringify(ast, null, 2));
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('[GenericASTWidget] Failed to copy AST JSON:', err);
    }
  }, [ast]);

  if (!ast) {
    return (
      <div className="widget-stage" data-testid="generic-ast-widget-empty">
        <div className="stage-card">
          <div className="stage-icon">🌳</div>
          <h2 className="stage-title">Generic AST Tree Inspector</h2>
          <p className="stage-description">
            No valid AST available for declaration <code>{declarationName || '(empty)'}</code>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="generic-ast-viewport" data-testid="generic-ast-widget">
      {/* Top Toolbar */}
      <div className="ast-tree-toolbar">
        <div className="ast-tree-toolbar-left">
          <span className="ast-toolbar-title">AST Tree Inspector:</span>
          <span className="ast-toolbar-decl">{declarationName || 'Anonymous'}</span>
          <span className="ast-toolbar-badge">{domain || 'Generic'}</span>
        </div>
        <div className="ast-tree-toolbar-right">
          <button
            type="button"
            className="btn btn-sm"
            onClick={handleExpandAll}
            data-testid="ast-expand-all-btn"
          >
            Expand All
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={handleCollapseAll}
            data-testid="ast-collapse-all-btn"
          >
            Collapse All
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={handleCopyJSON}
            data-testid="ast-copy-json-btn"
          >
            {copySuccess ? '✓ Copied' : '📋 Copy JSON'}
          </button>
        </div>
      </div>

      {/* Interactive Tree View */}
      <div className="ast-tree-content">
        <ASTTreeNode
          node={ast}
          path="0"
          depth={0}
          expandedPaths={expandedPaths}
          onToggleExpand={handleToggleExpand}
          onParameterChange={onParameterChange}
        />
      </div>
    </div>
  );
};
