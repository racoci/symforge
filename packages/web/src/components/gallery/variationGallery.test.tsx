/**
 * Automated Unit & Component Tests for Interactive Visual Variation Gallery.
 *
 * Verifies:
 * 1. Gallery drawer open/close, backdrop click, and Escape key handling.
 * 2. Loading state with skeleton cards, error banner, and empty state.
 * 3. Curriculum level stratification filtering (All vs L1..L5) and count badges.
 * 4. Candidate card badges, probability, metrics chips, code snippets, and thumbnail visualizers:
 *    - SDF 3D wireframe canvas
 *    - Audio audition player button and WebAudio playback
 *    - Hardware register flow diagram
 * 5. 1-Click code injection triggering `onInjectVariation`.
 * 6. Monaco Editor `executeEdits` integration and `Ctrl+Z` undo invariance:
 *    - Pass-to-Complete Gate: 1-click injection updates document and enables undo to original state.
 */

import { useState, createRef } from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { VariationGallery } from './VariationGallery';
import { VariationCard } from './VariationCard';
import type { VariationGalleryProps } from './types';
import type { VariationCandidate } from '../../core/pcfg/types';
import type { PunctureResult } from '../../core/generator/types';
import {
  LeanEditor,
  type LeanEditorHandle,
} from '../editor/LeanEditor';
import {
  createSphere,
  createCylinder,
  createDifference,
  createSmoothUnion,
  createSDFMacro,
  createSine,
  createAdd,
  createReg,
  createImm,
  createIntersection,
  createSDFHole,
  createFreeParam,
} from '../../core/ast/models';

// Mock Candidate Variations
const MOCK_CANDIDATES: VariationCandidate[] = [
  {
    id: 'cand_l1_sphere',
    ast: createSphere(createFreeParam('radius', 1.0, 0.1, 5.0)),
    leanCode: '(SDFNode.sphere (FreeParam.mk "radius" 1.0 0.1 5.0))',
    complexityLevel: 'L1',
    depth: 1,
    nodeCount: 1,
    domain: 'sdf',
    probability: 0.35,
    label: '[L1] Canonical Sphere',
  },
  {
    id: 'cand_l2_cylinder',
    ast: createCylinder(
      createFreeParam('radius', 0.5, 0.1, 2.0),
      createFreeParam('height', 2.0, 0.1, 5.0)
    ),
    leanCode: '(SDFNode.cylinder (FreeParam.mk "radius" 0.5 0.1 2.0) (FreeParam.mk "height" 2.0 0.1 5.0))',
    complexityLevel: 'L2',
    depth: 2,
    nodeCount: 2,
    domain: 'sdf',
    probability: 0.25,
    label: '[L2] Translated Cylinder',
  },
  {
    id: 'cand_l3_diff',
    ast: createDifference(
      createSphere(createFreeParam('r1', 1.5, 0.1, 3.0)),
      createSphere(createFreeParam('r2', 1.0, 0.1, 2.0))
    ),
    leanCode: '(SDFNode.difference (SDFNode.box ...) (SDFNode.sphere ...))',
    complexityLevel: 'L3',
    depth: 3,
    nodeCount: 4,
    domain: 'sdf',
    probability: 0.15,
    label: '[L3] CSG Difference',
  },
  {
    id: 'cand_l4_smooth',
    ast: createSmoothUnion(
      createFreeParam('k', 0.2, 0.0, 1.0),
      createSphere(createFreeParam('r1', 1.0, 0.1, 3.0)),
      createSphere(createFreeParam('r2', 1.0, 0.1, 3.0))
    ),
    leanCode: '(SDFNode.smooth_union 0.2 ...)',
    complexityLevel: 'L4',
    depth: 4,
    nodeCount: 6,
    domain: 'sdf',
    probability: 0.08,
    label: '[L4] Smooth Union Blend',
  },
  {
    id: 'cand_l5_abstract',
    ast: createSDFMacro('gyroid_lattice', {
      scale: createFreeParam('scale', 3.14, 1.0, 10.0),
      thickness: createFreeParam('thickness', 0.1, 0.01, 0.5),
    }),
    leanCode: '(SDFNode.gyroid_lattice ...)',
    complexityLevel: 'L5',
    depth: 5,
    nodeCount: 8,
    domain: 'sdf',
    probability: 0.04,
    label: '[L5] Structural Abstraction',
  },
  {
    id: 'cand_audio_sine',
    ast: createSine(
      createFreeParam('freq', 440.0, 20.0, 2000.0),
      createFreeParam('phase', 0.0, 0.0, 6.28)
    ),
    leanCode: '(AudioNode.sine (FreeParam.mk "freq" 440.0 20.0 2000.0))',
    complexityLevel: 'L1',
    depth: 1,
    nodeCount: 1,
    domain: 'audio',
    probability: 0.4,
    label: '[L1] 440Hz Sine Oscillator',
  },
  {
    id: 'cand_hw_pipe',
    ast: createAdd(createReg('r0'), createImm(42)),
    leanCode: '(HardwareNode.add (HardwareNode.reg "r0") (HardwareNode.imm 42))',
    complexityLevel: 'L2',
    depth: 2,
    nodeCount: 3,
    domain: 'hardware',
    probability: 0.3,
    label: '[L2] Register Add Pipeline',
  },
];

// Mock Puncture Result
const sphereTarget = createSphere(createFreeParam('radius', 1.25, 0.5, 3.0));
const MOCK_PUNCTURE_RESULT: PunctureResult = {
  holeId: 'hole_sdf_1',
  holeNode: createSDFHole('hole_sdf_1'),
  puncturedRootAST: createIntersection(
    createSDFHole('hole_sdf_1'),
    createSphere(createFreeParam('r2', 1.0, 0.1, 3.0))
  ),
  originalTargetNode: sphereTarget,
  context: {
    parentOp: 'intersection',
    grandparentOp: 'root',
    leftSiblingOp: null,
    targetDepth: 2,
    expectedDomain: 'sdf',
    expectedType: 'SDFNode',
    parentRelation: 'a',
  },
  targetSpan: {
    startLine: 4,
    startCol: 5,
    endLine: 4,
    endCol: 57,
    startOffset: 65,
    endOffset: 117,
  },
};

describe('VariationGallery Component', () => {
  const defaultProps: VariationGalleryProps = {
    isOpen: true,
    onClose: vi.fn(),
    candidates: MOCK_CANDIDATES,
    isLoading: false,
    error: null,
    punctureResult: MOCK_PUNCTURE_RESULT,
    selectedLevel: 'ALL',
    onSelectLevel: vi.fn(),
    onRegenerate: vi.fn(),
    onInjectVariation: vi.fn(),
  };

  it('does not render when isOpen is false', () => {
    render(<VariationGallery {...defaultProps} isOpen={false} />);
    expect(screen.queryByTestId('variation-gallery-overlay')).toBeNull();
    expect(screen.queryByTestId('variation-gallery-drawer')).toBeNull();
  });

  it('renders drawer, title, scope badge, and candidates when isOpen is true', () => {
    render(<VariationGallery {...defaultProps} />);

    expect(screen.getByTestId('variation-gallery-overlay')).toBeInTheDocument();
    expect(screen.getByTestId('variation-gallery-drawer')).toBeInTheDocument();
    expect(screen.getByText('Procedural Variations')).toBeInTheDocument();

    const scopeBadge = screen.getByTestId('target-scope-badge');
    expect(scopeBadge).toBeInTheDocument();
    expect(scopeBadge.textContent).toMatch(/🎯 Scope: SphereNode \(child: a\)/);

    // Verify all candidates rendered in ALL view
    const cards = screen.getAllByTestId('variation-card');
    expect(cards.length).toBe(MOCK_CANDIDATES.length);
  });

  it('triggers onClose when close button is clicked', () => {
    const handleClose = vi.fn();
    render(<VariationGallery {...defaultProps} onClose={handleClose} />);

    const closeBtn = screen.getByTestId('close-gallery-btn');
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('triggers onClose when frosted backdrop is clicked', () => {
    const handleClose = vi.fn();
    render(<VariationGallery {...defaultProps} onClose={handleClose} />);

    const backdrop = screen.getByTestId('variation-gallery-backdrop');
    fireEvent.click(backdrop);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('triggers onClose when pressing Escape key', () => {
    const handleClose = vi.fn();
    render(<VariationGallery {...defaultProps} onClose={handleClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('renders loading state with spinner and skeleton cards', () => {
    render(<VariationGallery {...defaultProps} isLoading={true} candidates={[]} />);

    expect(screen.getByTestId('gallery-loading')).toBeInTheDocument();
    expect(
      screen.getByText(/Synthesizing variations across curriculum complexity strata/i)
    ).toBeInTheDocument();

    const skeletons = screen.getAllByTestId('skeleton-card');
    expect(skeletons.length).toBeGreaterThanOrEqual(3);
  });

  it('renders error banner with message and retry button', () => {
    const handleRegen = vi.fn();
    render(
      <VariationGallery
        {...defaultProps}
        error="Synthesis timeout on worker thread"
        onRegenerate={handleRegen}
      />
    );

    const errorBanner = screen.getByTestId('gallery-error');
    expect(errorBanner).toBeInTheDocument();
    expect(errorBanner).toHaveTextContent('Synthesis timeout on worker thread');

    const retryBtn = screen.getByText('Retry');
    fireEvent.click(retryBtn);
    expect(handleRegen).toHaveBeenCalledTimes(1);
  });

  it('renders empty state when no candidates match filter', () => {
    const handleSelectLevel = vi.fn();
    render(
      <VariationGallery
        {...defaultProps}
        candidates={[]}
        selectedLevel="L5"
        onSelectLevel={handleSelectLevel}
      />
    );

    expect(screen.getByTestId('gallery-empty-state')).toBeInTheDocument();
    expect(screen.getByText('No variations found')).toBeInTheDocument();

    const showAllBtn = screen.getByText('Show All Levels');
    fireEvent.click(showAllBtn);
    expect(handleSelectLevel).toHaveBeenCalledWith('ALL');
  });

  it('renders filter bar with all level buttons and candidate counts', () => {
    const handleSelectLevel = vi.fn();
    render(<VariationGallery {...defaultProps} onSelectLevel={handleSelectLevel} />);

    const filterAll = screen.getByTestId('filter-btn-all');
    const filterL1 = screen.getByTestId('filter-btn-l1');
    const filterL2 = screen.getByTestId('filter-btn-l2');
    const filterL3 = screen.getByTestId('filter-btn-l3');
    const filterL4 = screen.getByTestId('filter-btn-l4');
    const filterL5 = screen.getByTestId('filter-btn-l5');

    expect(filterAll).toBeInTheDocument();
    expect(filterL1).toBeInTheDocument();
    expect(filterL2).toBeInTheDocument();
    expect(filterL3).toBeInTheDocument();
    expect(filterL4).toBeInTheDocument();
    expect(filterL5).toBeInTheDocument();

    fireEvent.click(filterL2);
    expect(handleSelectLevel).toHaveBeenCalledWith('L2');
  });

  it('filters candidates correctly when a specific curriculum level is active', () => {
    render(<VariationGallery {...defaultProps} selectedLevel="L2" />);

    const cards = screen.getAllByTestId('variation-card');
    // Only L2 candidates (cand_l2_cylinder and cand_hw_pipe)
    expect(cards.length).toBe(2);
    for (const card of cards) {
      expect(card).toHaveAttribute('data-level', 'L2');
    }
  });

  it('triggers onRegenerate when regenerate button is clicked', () => {
    const handleRegen = vi.fn();
    render(<VariationGallery {...defaultProps} onRegenerate={handleRegen} />);

    const regenBtn = screen.getByTestId('regenerate-variations-btn');
    fireEvent.click(regenBtn);
    expect(handleRegen).toHaveBeenCalledTimes(1);
  });
});

describe('VariationCard Component', () => {
  it('renders complexity badge, domain badge, probability, title, and metrics chips', () => {
    const candidate = MOCK_CANDIDATES[0];
    const handleApply = vi.fn();

    render(<VariationCard candidate={candidate} onApply={handleApply} />);

    expect(screen.getByTestId('complexity-badge')).toHaveTextContent('L1');
    expect(screen.getByTestId('domain-badge')).toHaveTextContent('SDF');
    expect(screen.getByTestId('probability-badge')).toHaveTextContent('35.0%');
    expect(screen.getByTestId('variation-card-title')).toHaveTextContent(
      '[L1] Canonical Sphere'
    );
    expect(screen.getByTestId('depth-chip')).toHaveTextContent('Depth: 1');
    expect(screen.getByTestId('node-count-chip')).toHaveTextContent('Nodes: 1');

    const snippet = screen.getByTestId('code-snippet-preview');
    expect(snippet).toHaveTextContent('SDFNode.sphere');
  });

  it('renders 3D wireframe preview canvas for SDF domain', () => {
    const candidate = MOCK_CANDIDATES[1]; // Cylinder SDF
    render(<VariationCard candidate={candidate} onApply={vi.fn()} />);

    expect(screen.getByTestId('sdf-thumbnail-preview')).toBeInTheDocument();
    const canvas = screen.getByTitle('Click and drag to rotate wireframe');
    expect(canvas).toBeInTheDocument();

    // Simulate drag interaction to rotate
    fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });
    fireEvent.mouseMove(canvas, { clientX: 120, clientY: 110 });
    fireEvent.mouseUp(canvas);
  });

  it('renders WebAudio audition button for Audio domain and toggles tone playback', () => {
    const candidate = MOCK_CANDIDATES[5]; // Audio Sine
    render(<VariationCard candidate={candidate} onApply={vi.fn()} />);

    expect(screen.getByTestId('audio-thumbnail-preview')).toBeInTheDocument();
    const auditionBtn = screen.getByTestId('audio-audition-btn');
    expect(auditionBtn).toHaveTextContent('▶ Audition (1s)');

    fireEvent.click(auditionBtn);
    expect(auditionBtn).toHaveTextContent('🔊 Playing...');
  });

  it('renders hardware register flow diagram for Hardware domain', () => {
    const candidate = MOCK_CANDIDATES[6]; // Hardware
    render(<VariationCard candidate={candidate} onApply={vi.fn()} />);

    const flowPreview = screen.getByTestId('hardware-flow-preview');
    expect(flowPreview).toBeInTheDocument();
    expect(flowPreview).toHaveTextContent('FETCH');
    expect(flowPreview).toHaveTextContent('DECODE');
    expect(flowPreview).toHaveTextContent('EXEC');
  });

  it('triggers onApply with candidate when clicking Apply to Editor button', () => {
    const candidate = MOCK_CANDIDATES[1];
    const handleApply = vi.fn();
    render(<VariationCard candidate={candidate} onApply={handleApply} />);

    const applyBtn = screen.getByTestId('apply-variation-btn');
    expect(applyBtn).toHaveTextContent('Apply to Editor');
    fireEvent.click(applyBtn);

    expect(handleApply).toHaveBeenCalledTimes(1);
    expect(handleApply).toHaveBeenCalledWith(candidate);
  });
});

describe('Pass-to-Complete Gate: 1-Click Code Injection & Monaco Undo Invariance', () => {
  const initialProgram = `import SymforgeIr.Types

def myGyroidSDF : SDFNode :=
  SDFNode.intersection
    (SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0))
    (SDFNode.gyroid
      (FreeParam.mk "scale" 3.14 1.0 10.0)
      (FreeParam.mk "thickness" 0.08 0.01 0.5))`;

  it('replaces target AST code range upon 1-click injection and enables Ctrl+Z undo restoration', () => {
    let documentCode = initialProgram;
    const editorRef = createRef<LeanEditorHandle>();

    const handleCodeChange = vi.fn((newCode: string) => {
      documentCode = newCode;
    });

    const TestAppContainer = () => {
      const [isOpen, setIsOpen] = useState(true);
      const [code, setCode] = useState(initialProgram);

      const handleInject = (candidate: VariationCandidate) => {
        const success = editorRef.current?.injectCodeAtSpan(
          MOCK_PUNCTURE_RESULT.targetSpan,
          candidate.leanCode
        );
        if (success) {
          setIsOpen(false);
        }
      };

      return (
        <div>
          <LeanEditor
            ref={editorRef}
            value={code}
            onChange={(c) => {
              setCode(c);
              handleCodeChange(c);
            }}
          />
          <VariationGallery
            isOpen={isOpen}
            onClose={() => setIsOpen(false)}
            candidates={MOCK_CANDIDATES}
            isLoading={false}
            error={null}
            punctureResult={MOCK_PUNCTURE_RESULT}
            selectedLevel="ALL"
            onSelectLevel={() => {}}
            onRegenerate={() => {}}
            onInjectVariation={handleInject}
          />
        </div>
      );
    };

    render(<TestAppContainer />);

    // 1. Verify gallery is open with cards
    expect(screen.getByTestId('variation-gallery-drawer')).toBeInTheDocument();

    // 2. Select the L2 Cylinder candidate
    const applyButtons = screen.getAllByTestId('apply-variation-btn');
    const l2CylinderApplyBtn = applyButtons[1]; // cand_l2_cylinder
    expect(l2CylinderApplyBtn).toBeInTheDocument();

    // 3. Click Apply to Editor
    act(() => {
      fireEvent.click(l2CylinderApplyBtn);
    });

    // 4. Verify gallery closed upon successful injection
    expect(screen.queryByTestId('variation-gallery-drawer')).toBeNull();

    // 5. Verify the injected code replaced the target sphere span
    expect(documentCode).toContain('SDFNode.cylinder');
    expect(documentCode).toContain('(FreeParam.mk "radius" 0.5 0.1 2.0)');
    expect(documentCode).toContain('(FreeParam.mk "height" 2.0 0.1 5.0)');
    expect(documentCode).not.toContain('(SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0))');

    // 6. Simulate Ctrl+Z Undo via editor handle
    act(() => {
      editorRef.current?.undo();
    });

    // 7. Verify document was reverted to the original code verbatim
    expect(documentCode).toBe(initialProgram);
    expect(documentCode).toContain('(SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0))');
    expect(documentCode).not.toContain('SDFNode.cylinder');

    // 8. Simulate Redo to reapply
    act(() => {
      editorRef.current?.redo();
    });
    expect(documentCode).toContain('SDFNode.cylinder');
  });

  it('executes atomic Monaco executeEdits and pushUndoStop transaction when Monaco editor is active', () => {
    const editorRef = createRef<LeanEditorHandle>();
    let currentText = initialProgram;

    const mockExecuteEdits = vi.fn((_source: string, edits: any[]) => {
      const edit = edits[0];
      const lines = currentText.split('\n');
      lines[3] = `    ${edit.text}`;
      currentText = lines.join('\n');
    });

    const mockPushUndoStop = vi.fn();
    const mockFocus = vi.fn();
    const mockGetValue = vi.fn(() => currentText);
    const mockUndoRun = vi.fn(() => {
      currentText = initialProgram;
    });

    const mockEditorInstance: any = {
      focus: mockFocus,
      executeEdits: mockExecuteEdits,
      pushUndoStop: mockPushUndoStop,
      getValue: mockGetValue,
      getAction: vi.fn((action: string) => {
        if (action === 'undo') return { run: mockUndoRun };
        return null;
      }),
      trigger: vi.fn(),
    };

    render(
      <LeanEditor
        ref={editorRef}
        value={initialProgram}
        onChange={(val) => {
          currentText = val;
        }}
      />
    );

    // Set mock editor instance
    act(() => {
      editorRef.current?._setEditorInstance?.(mockEditorInstance);
    });

    const candidate = MOCK_CANDIDATES[1]; // L2 Cylinder
    act(() => {
      const ok = editorRef.current?.injectCodeAtSpan(
        MOCK_PUNCTURE_RESULT.targetSpan,
        candidate.leanCode
      );
      expect(ok).toBe(true);
    });

    expect(mockFocus).toHaveBeenCalled();
    expect(mockExecuteEdits).toHaveBeenCalledWith('symforge-scoped-variation', [
      {
        range: {
          startLineNumber: 4,
          startColumn: 5,
          endLineNumber: 4,
          endColumn: 57,
        },
        text: candidate.leanCode,
        forceMoveMarkers: true,
      },
    ]);
    expect(mockPushUndoStop).toHaveBeenCalled();
    expect(currentText).toContain('SDFNode.cylinder');

    // Trigger Undo
    act(() => {
      editorRef.current?.undo();
    });
    expect(mockUndoRun).toHaveBeenCalled();
    expect(currentText).toBe(initialProgram);
  });
});
