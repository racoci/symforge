/**
 * Individual Variation Candidate Card.
 *
 * Displays complexity badge, domain badge, probability percentage,
 * title/description, domain thumbnail preview (SDF 3D wireframe / Audio audition / Hardware flow),
 * syntax code snippet, metrics chips (Depth, Node Count), and the 'Apply to Editor' button.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { VariationCardProps } from './types';

export const VariationCard: React.FC<VariationCardProps> = ({
  candidate,
  onApply,
  isApplying = false,
}) => {
  const [isPlayingAudio, setIsPlayingAudio] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rotAngle, setRotAngle] = useState<{ theta: number; phi: number }>({
    theta: 0.45,
    phi: 0.35,
  });
  const isDraggingRef = useRef<boolean>(false);
  const lastMousePosRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // WebAudio 1-second Audition tone playback
  const handleAudition = useCallback(() => {
    if (isPlayingAudio) return;
    setIsPlayingAudio(true);

    let ctx: AudioContext | null = null;

    try {
      const AudioCtxClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

      if (AudioCtxClass) {
        ctx = new AudioCtxClass();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        const kind = (candidate.ast.kind || '').toLowerCase();
        if (kind.includes('saw')) {
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(220, ctx.currentTime);
        } else if (kind.includes('square')) {
          osc.type = 'square';
          osc.frequency.setValueAtTime(330, ctx.currentTime);
        } else if (kind.includes('noise')) {
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(110, ctx.currentTime);
        } else {
          osc.type = 'sine';
          osc.frequency.setValueAtTime(440, ctx.currentTime);
        }

        const now = ctx.currentTime;
        gain.gain.setValueAtTime(0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.95);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 1.0);
      }
    } catch (err) {
      console.error('[VariationCard] WebAudio audition playback note with context:', {
        error: err,
        candidateId: candidate.id,
        kind: candidate.ast.kind,
        timestamp: Date.now(),
      });
    }

    const timer = setTimeout(() => {
      setIsPlayingAudio(false);
      try {
        if (ctx && ctx.state !== 'closed') {
          ctx.close().catch((err) => {
            console.error('[VariationCard] AudioContext async close failed with context:', {
              error: err,
              candidateId: candidate.id,
              timestamp: Date.now(),
            });
          });
        }
      } catch (err) {
        console.error('[VariationCard] Safe context disposal failed with context:', {
          error: err,
          candidateId: candidate.id,
          timestamp: Date.now(),
        });
      }
    }, 1000);

    return () => clearTimeout(timer);
  }, [candidate, isPlayingAudio]);

  // SDF 3D Wireframe Canvas Projection
  useEffect(() => {
    if (candidate.domain !== 'sdf') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#090e1a';
    ctx.fillRect(0, 0, width, height);

    const { theta, phi } = rotAngle;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const cosP = Math.cos(phi);
    const sinP = Math.sin(phi);

    const project = (x: number, y: number, z: number): [number, number] => {
      const x1 = x * cosT + z * sinT;
      const z1 = -x * sinT + z * cosT;
      const y2 = y * cosP - z1 * sinP;
      const z2 = y * sinP + z1 * cosP;
      const scale = 22 / (1 + z2 * 0.04);
      return [cx + x1 * scale, cy - y2 * scale];
    };

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.2;

    const kind = (candidate.ast.kind || '').toLowerCase();
    if (kind.includes('box')) {
      const s = 1.3;
      const vertices: [number, number, number][] = [
        [-s, -s, -s],
        [s, -s, -s],
        [s, s, -s],
        [-s, s, -s],
        [-s, -s, s],
        [s, -s, s],
        [s, s, s],
        [-s, s, s],
      ];
      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];
      ctx.beginPath();
      for (const [i, j] of edges) {
        const [x1, y1] = project(...vertices[i]);
        const [x2, y2] = project(...vertices[j]);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
    } else if (kind.includes('cylinder')) {
      const r = 1.1;
      const h = 1.4;
      const steps = 16;
      ctx.beginPath();
      for (let i = 0; i < steps; i++) {
        const a1 = (i / steps) * Math.PI * 2;
        const a2 = ((i + 1) / steps) * Math.PI * 2;
        const [tx1, ty1] = project(r * Math.cos(a1), h, r * Math.sin(a1));
        const [tx2, ty2] = project(r * Math.cos(a2), h, r * Math.sin(a2));
        ctx.moveTo(tx1, ty1);
        ctx.lineTo(tx2, ty2);
        const [bx1, by1] = project(r * Math.cos(a1), -h, r * Math.sin(a1));
        const [bx2, by2] = project(r * Math.cos(a2), -h, r * Math.sin(a2));
        ctx.moveTo(bx1, by1);
        ctx.lineTo(bx2, by2);
        if (i % 4 === 0) {
          ctx.moveTo(tx1, ty1);
          ctx.lineTo(bx1, by1);
        }
      }
      ctx.stroke();
    } else if (kind.includes('torus')) {
      const R = 1.3;
      const r = 0.5;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const u = (i / 6) * Math.PI * 2;
        for (let j = 0; j < 12; j++) {
          const v1 = (j / 12) * Math.PI * 2;
          const v2 = ((j + 1) / 12) * Math.PI * 2;
          const p1: [number, number, number] = [
            (R + r * Math.cos(v1)) * Math.cos(u),
            r * Math.sin(v1),
            (R + r * Math.cos(v1)) * Math.sin(u),
          ];
          const p2: [number, number, number] = [
            (R + r * Math.cos(v2)) * Math.cos(u),
            r * Math.sin(v2),
            (R + r * Math.cos(v2)) * Math.sin(u),
          ];
          const [x1, y1] = project(...p1);
          const [x2, y2] = project(...p2);
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
        }
      }
      ctx.stroke();
    } else {
      // Default: Sphere or Gyroid wireframe latitude/longitude rings
      const r = 1.4;
      const rings = 5;
      const segments = 12;
      ctx.beginPath();
      for (let i = 1; i < rings; i++) {
        const lat = (i / rings - 0.5) * Math.PI;
        const ringR = r * Math.cos(lat);
        const ringY = r * Math.sin(lat);
        for (let j = 0; j < segments; j++) {
          const lon1 = (j / segments) * Math.PI * 2;
          const lon2 = ((j + 1) / segments) * Math.PI * 2;
          const [x1, y1] = project(ringR * Math.cos(lon1), ringY, ringR * Math.sin(lon1));
          const [x2, y2] = project(ringR * Math.cos(lon2), ringY, ringR * Math.sin(lon2));
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
        }
      }
      for (let j = 0; j < 4; j++) {
        const lon = (j / 4) * Math.PI;
        for (let i = 0; i < segments; i++) {
          const lat1 = (i / segments) * Math.PI * 2;
          const lat2 = ((i + 1) / segments) * Math.PI * 2;
          const [x1, y1] = project(r * Math.sin(lat1) * Math.cos(lon), r * Math.cos(lat1), r * Math.sin(lat1) * Math.sin(lon));
          const [x2, y2] = project(r * Math.sin(lat2) * Math.cos(lon), r * Math.cos(lat2), r * Math.sin(lat2) * Math.sin(lon));
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
        }
      }
      ctx.stroke();
    }
  }, [candidate, rotAngle]);

  const levelClass = `badge-l${candidate.complexityLevel.slice(1).toLowerCase()}`;
  const probabilityPercent = `${(candidate.probability * 100).toFixed(1)}%`;
  const displayTitle = candidate.label || `[${candidate.complexityLevel}] Variation`;

  return (
    <article
      className="variation-card"
      data-testid="variation-card"
      data-level={candidate.complexityLevel}
      data-domain={candidate.domain}
    >
      {/* Header: Complexity badge, Domain badge, Probability */}
      <div className="card-header">
        <div className="card-badges">
          <span
            className={`badge-level ${levelClass}`}
            data-testid="complexity-badge"
          >
            {candidate.complexityLevel}
          </span>
          <span className="badge-domain" data-testid="domain-badge">
            {candidate.domain.toUpperCase()}
          </span>
        </div>
        <span className="badge-prob" data-testid="probability-badge" title="Grammar Derivation Probability">
          {probabilityPercent}
        </span>
      </div>

      {/* Title / Description */}
      <h3 className="card-title" data-testid="variation-card-title">
        {displayTitle}
      </h3>

      {/* Domain Thumbnail Preview */}
      <div className="card-preview-area">
        {candidate.domain === 'sdf' && (
          <div className="sdf-thumbnail-container" data-testid="sdf-thumbnail-preview">
            <canvas
              ref={canvasRef}
              width={160}
              height={90}
              className="sdf-wireframe-canvas"
              onMouseDown={(e) => {
                isDraggingRef.current = true;
                lastMousePosRef.current = { x: e.clientX, y: e.clientY };
              }}
              onMouseMove={(e) => {
                if (!isDraggingRef.current) return;
                const dx = e.clientX - lastMousePosRef.current.x;
                const dy = e.clientY - lastMousePosRef.current.y;
                lastMousePosRef.current = { x: e.clientX, y: e.clientY };
                setRotAngle((prev) => ({
                  theta: prev.theta + dx * 0.025,
                  phi: Math.max(-1.4, Math.min(1.4, prev.phi + dy * 0.025)),
                }));
              }}
              onMouseUp={() => {
                isDraggingRef.current = false;
              }}
              onMouseLeave={() => {
                isDraggingRef.current = false;
              }}
              title="Click and drag to rotate wireframe"
            />
            <span className="canvas-rotate-hint">3D Wireframe Preview</span>
          </div>
        )}

        {candidate.domain === 'audio' && (
          <div className="audio-thumbnail-preview" data-testid="audio-thumbnail-preview">
            <div className="audio-wave-bars" aria-hidden="true">
              {[35, 75, 95, 60, 85, 45, 70, 90, 65, 40, 80, 50].map((h, idx) => (
                <div
                  key={idx}
                  className={`wave-bar ${isPlayingAudio ? 'animating' : ''}`}
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
            <button
              type="button"
              className={`btn btn-sm audition-btn ${isPlayingAudio ? 'playing' : ''}`}
              data-testid="audio-audition-btn"
              onClick={handleAudition}
            >
              {isPlayingAudio ? '🔊 Playing...' : '▶ Audition (1s)'}
            </button>
          </div>
        )}

        {candidate.domain === 'hardware' && (
          <div className="hardware-flow-preview" data-testid="hardware-flow-preview">
            <div className="hardware-stage-pill">FETCH</div>
            <span className="flow-arrow">→</span>
            <div className="hardware-stage-pill">DECODE</div>
            <span className="flow-arrow">→</span>
            <div className="hardware-stage-pill active-stage">EXEC</div>
            <div className="hardware-flow-desc">
              <code>
                {candidate.leanCode.length > 28
                  ? candidate.leanCode.slice(0, 26) + '...'
                  : candidate.leanCode}
              </code>
            </div>
          </div>
        )}

        {candidate.domain === 'universal' && (
          <div className="hardware-flow-preview" data-testid="universal-preview">
            <span className="flow-arrow">ƒ(x)</span>
            <div className="hardware-flow-desc">
              <code>Universal Mathematical Form</code>
            </div>
          </div>
        )}
      </div>

      {/* Syntax-highlighted code snippet preview in monospace box */}
      <pre className="code-snippet-preview" data-testid="code-snippet-preview">
        <code>{candidate.leanCode}</code>
      </pre>

      {/* Footer: Metrics chips and 'Apply to Editor' button */}
      <div className="card-footer">
        <div className="card-metrics">
          <span className="metric-chip" data-testid="depth-chip">
            Depth: {candidate.depth}
          </span>
          <span className="metric-chip" data-testid="node-count-chip">
            Nodes: {candidate.nodeCount}
          </span>
        </div>

        <button
          type="button"
          className="btn btn-primary apply-variation-btn"
          data-testid="apply-variation-btn"
          onClick={() => onApply(candidate)}
          disabled={isApplying}
        >
          {isApplying ? 'Applying...' : 'Apply to Editor'}
        </button>
      </div>
    </article>
  );
};

export default VariationCard;
