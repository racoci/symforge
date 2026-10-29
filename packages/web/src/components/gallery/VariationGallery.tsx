/**
 * Interactive Visual Variation Gallery Sliding Drawer.
 *
 * Sliding side drawer / modal layout with frosted glass backdrop.
 * Provides curriculum complexity level filtering (All, L1..L5), regeneration trigger,
 * loading skeleton cards, error banners, empty states, and a scrollable grid
 * of VariationCard candidates with 1-click code injection.
 */

import React, { useEffect, useMemo, useCallback } from 'react';
import type { VariationGalleryProps, GalleryFilterLevel } from './types';
import { VariationCard } from './VariationCard';
import { formatNodeDisplayName } from '../../core/generator';

interface LevelOption {
  value: GalleryFilterLevel;
  label: string;
}

const LEVEL_OPTIONS: LevelOption[] = [
  { value: 'ALL', label: 'All' },
  { value: 'L1', label: 'L1: Atomic' },
  { value: 'L2', label: 'L2: Unary' },
  { value: 'L3', label: 'L3: Binary' },
  { value: 'L4', label: 'L4: Hierarchical' },
  { value: 'L5', label: 'L5: Abstraction' },
];

export const VariationGallery: React.FC<VariationGalleryProps> = ({
  isOpen,
  onClose,
  candidates,
  isLoading,
  error,
  punctureResult,
  selectedLevel,
  onSelectLevel,
  onRegenerate,
  onInjectVariation,
}) => {
  // Close on Escape key press
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    if (!isOpen) return;
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, handleKeyDown]);

  // Compute filtered candidate list based on selected curriculum level
  const filteredCandidates = useMemo(() => {
    if (selectedLevel === 'ALL') {
      return candidates;
    }
    return candidates.filter((c) => c.complexityLevel === selectedLevel);
  }, [candidates, selectedLevel]);

  // Compute target scope badge text (e.g. 🎯 Scope: CylinderNode (child: b))
  const targetScopeBadgeText = useMemo(() => {
    if (!punctureResult) return '🎯 Scope: Global Expression';
    const targetNode = punctureResult.originalTargetNode;
    const nodeName = targetNode ? formatNodeDisplayName(targetNode) : 'Node';
    const rel = punctureResult.context?.parentRelation
      ? `child: ${punctureResult.context.parentRelation}`
      : 'root';
    return `🎯 Scope: ${nodeName} (${rel})`;
  }, [punctureResult]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="variation-gallery-overlay"
      data-testid="variation-gallery-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Procedural Variations Gallery"
    >
      {/* Frosted Glass Backdrop */}
      <div
        className="variation-gallery-backdrop"
        data-testid="variation-gallery-backdrop"
        onClick={onClose}
        aria-label="Close variations gallery"
      />

      {/* Sliding Side Drawer */}
      <aside
        className="variation-gallery-drawer"
        data-testid="variation-gallery-drawer"
      >
        {/* Header */}
        <header className="gallery-header">
          <div className="gallery-header-left">
            <div className="gallery-title-row">
              <span className="gallery-sparkle-icon" aria-hidden="true">
                ✨
              </span>
              <h2 className="gallery-title">Procedural Variations</h2>
            </div>
            <div
              className="gallery-scope-badge"
              data-testid="target-scope-badge"
              title="Target AST hole location"
            >
              {targetScopeBadgeText}
            </div>
          </div>
          <button
            type="button"
            className="close-gallery-btn"
            data-testid="close-gallery-btn"
            onClick={onClose}
            aria-label="Close"
            title="Close Gallery (Esc)"
          >
            ✕
          </button>
        </header>

        {/* Filter Bar */}
        <div className="gallery-filter-bar" data-testid="gallery-filter-bar">
          <div className="filter-buttons-group">
            {LEVEL_OPTIONS.map((opt) => {
              const isSelected = selectedLevel === opt.value;
              const count =
                opt.value === 'ALL'
                  ? candidates.length
                  : candidates.filter((c) => c.complexityLevel === opt.value).length;

              return (
                <button
                  key={opt.value}
                  type="button"
                  className={`filter-btn ${isSelected ? 'active' : ''}`}
                  data-testid={`filter-btn-${opt.value.toLowerCase()}`}
                  onClick={() => onSelectLevel(opt.value)}
                >
                  <span>{opt.label}</span>
                  <span className="filter-count">({count})</span>
                </button>
              );
            })}
          </div>

          <button
            type="button"
            className="btn btn-sm btn-regenerate"
            data-testid="regenerate-variations-btn"
            onClick={onRegenerate}
            disabled={isLoading}
            title="Resample procedural grammar variations"
          >
            🔄 Regenerate
          </button>
        </div>

        {/* Gallery Content Area */}
        <div className="gallery-content" data-testid="gallery-content">
          {/* Loading State with Skeleton Cards */}
          {isLoading && (
            <div className="gallery-loading" data-testid="gallery-loading">
              <div className="loading-spinner" aria-hidden="true" />
              <p className="loading-text">
                Synthesizing variations across curriculum complexity strata...
              </p>
              <div className="skeleton-cards-grid">
                {[1, 2, 3].map((idx) => (
                  <div
                    key={idx}
                    className="skeleton-card"
                    data-testid="skeleton-card"
                  >
                    <div className="skeleton-header shimmer" />
                    <div className="skeleton-thumbnail shimmer" />
                    <div className="skeleton-code shimmer" />
                    <div className="skeleton-footer shimmer" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error Banner */}
          {!isLoading && error && (
            <div className="gallery-error-banner" data-testid="gallery-error">
              <span className="error-icon" aria-hidden="true">
                ⚠️
              </span>
              <span className="error-message">{error}</span>
              <button
                type="button"
                className="btn btn-sm btn-secondary retry-btn"
                onClick={onRegenerate}
              >
                Retry
              </button>
            </div>
          )}

          {/* Empty State */}
          {!isLoading && !error && filteredCandidates.length === 0 && (
            <div
              className="gallery-empty-state"
              data-testid="gallery-empty-state"
            >
              <div className="empty-icon" aria-hidden="true">
                🔍
              </div>
              <h3>No variations found</h3>
              <p>
                No candidate variations match the selected filter (
                <strong>{selectedLevel}</strong>).
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => onSelectLevel('ALL')}
              >
                Show All Levels
              </button>
            </div>
          )}

          {/* Candidates Grid */}
          {!isLoading && !error && filteredCandidates.length > 0 && (
            <div className="gallery-grid" data-testid="gallery-grid">
              {filteredCandidates.map((candidate) => (
                <VariationCard
                  key={candidate.id}
                  candidate={candidate}
                  onApply={onInjectVariation}
                />
              ))}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
};

export default VariationGallery;
