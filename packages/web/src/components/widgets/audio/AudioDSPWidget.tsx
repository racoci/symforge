/**
 * WebAudio DSP Synthesis Widget for AudioNode.
 *
 * Provides real-time interactive audio audition, live 2D time-domain oscilloscope,
 * 2048-point logarithmic FFT frequency spectrogram, in-viewport parameter scrubber drawer,
 * and resilient headless/JSDOM fallback rendering.
 */

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import type { DomainWidgetProps } from '../types';
import type { AudioNode as ASTAudioNode, FreeParam } from '../../../core/ast/types';
import { isAudioNode, collectFreeParams } from '../../../core/ast/models';
import { useFreeParamStore } from '../../../core/store/freeParamStore';
import {
  compileAudioGraph,
  createAudioContext,
  isAudioContextAvailable,
  type CompiledAudioGraph,
} from './audioGraphCompiler';

export type AudioDisplayMode = 'split' | 'oscilloscope' | 'fft';

interface AudioParameterOverlayProps {
  params: FreeParam[];
  onClose: () => void;
}

/**
 * Floating in-viewport parameter scrubber drawer overlay for AudioNode.
 */
export const AudioParameterOverlay: React.FC<AudioParameterOverlayProps> = React.memo(
  ({ params, onClose }) => {
    const values = useFreeParamStore((state) => state.values);
    const updateParamValue = useFreeParamStore((state) => state.updateParamValue);
    const resetParam = useFreeParamStore((state) => state.resetParam);
    const resetAllParams = useFreeParamStore((state) => state.resetAllParams);

    return (
      <div className="audio-param-overlay" data-testid="audio-param-overlay">
        <div className="audio-param-overlay-header">
          <span className="audio-param-overlay-title">Synth Parameters ({params.length})</span>
          <div className="audio-param-overlay-actions">
            {params.length > 0 && (
              <button
                type="button"
                className="audio-btn"
                onClick={resetAllParams}
                data-testid="audio-reset-all-params-btn"
                style={{ fontSize: '0.6875rem', padding: '2px 6px' }}
                title="Reset all synth parameters to default"
              >
                Reset All
              </button>
            )}
            <button
              type="button"
              className="audio-btn"
              onClick={onClose}
              data-testid="close-audio-parameters-btn"
              style={{ fontSize: '0.6875rem', padding: '2px 6px' }}
              title="Close parameter drawer"
            >
              ✕
            </button>
          </div>
        </div>

        {params.length === 0 ? (
          <div className="audio-param-empty" data-testid="audio-param-empty">
            No continuous FreeParams in this AudioNode.
          </div>
        ) : (
          <div className="audio-param-list" data-testid="audio-param-list">
            {params.map((p) => {
              const currentVal = values[p.id] ?? p.defaultVal;
              const step = (p.maxVal - p.minVal) / 100 || 0.01;
              return (
                <div key={p.id} className="audio-param-item" data-testid={`audio-param-item-${p.id}`}>
                  <div className="audio-param-item-header">
                    <span className="audio-param-name">{p.id}</span>
                    <span className="audio-param-val" data-testid={`audio-param-val-${p.id}`}>
                      {currentVal >= 100 ? currentVal.toFixed(1) : currentVal.toFixed(3)}
                    </span>
                  </div>
                  <div className="audio-param-slider-row">
                    <input
                      type="range"
                      className="audio-slider"
                      min={p.minVal}
                      max={p.maxVal}
                      step={step}
                      value={currentVal}
                      onChange={(e) => updateParamValue(p.id, parseFloat(e.target.value))}
                      data-testid={`audio-param-slider-${p.id}`}
                    />
                    <button
                      type="button"
                      className="audio-btn"
                      onClick={() => resetParam(p.id)}
                      data-testid={`audio-reset-param-${p.id}`}
                      style={{ fontSize: '0.625rem', padding: '2px 4px' }}
                      title={`Reset to default (${p.defaultVal})`}
                    >
                      ↺
                    </button>
                  </div>
                  <div className="audio-param-range-bounds">
                    <span>{p.minVal}</span>
                    <span>{p.maxVal}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }
);

AudioParameterOverlay.displayName = 'AudioParameterOverlay';

/**
 * Interactive WebAudio DSP Synthesis Widget.
 */
export const AudioDSPWidget: React.FC<DomainWidgetProps> = ({
  ast,
  declarationName,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const graphRef = useRef<CompiledAudioGraph | null>(null);

  const [hasAudioContext, setHasAudioContext] = useState<boolean>(() => isAudioContextAvailable());
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isGateActive, setIsGateActive] = useState<boolean>(false);
  const [isLooping, setIsLooping] = useState<boolean>(true);
  const [masterVolume, setMasterVolume] = useState<number>(0.8);
  const [displayMode, setDisplayMode] = useState<AudioDisplayMode>('split');
  const [showParameters, setShowParameters] = useState<boolean>(true);

  // Validate AST domain
  const isAudio = Boolean(ast && isAudioNode(ast));
  const resolvedNode = useMemo(() => (isAudio ? (ast as ASTAudioNode) : null), [ast, isAudio]);
  const params = useMemo(() => (resolvedNode ? collectFreeParams(resolvedNode) : []), [resolvedNode]);

  // Safe AudioContext resume with explicit timeout
  const resumeContext = useCallback(async (ctx: AudioContext): Promise<boolean> => {
    if (ctx.state === 'running') return true;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const timeoutPromise = new Promise<void>((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error('AudioContext resume timeout')), 1500);
    });

    try {
      await Promise.race([ctx.resume(), timeoutPromise]);
      return true;
    } catch (err) {
      console.error('[AudioDSPWidget] Failed to resume AudioContext:', {
        state: ctx.state,
        error: err,
        timestamp: Date.now(),
      });
      return false;
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }, []);

  // Initialize and re-compile WebAudio graph when AST changes
  useEffect(() => {
    if (!resolvedNode) return;

    if (!audioContextRef.current) {
      const ctx = createAudioContext();
      if (!ctx) {
        setHasAudioContext(false);
        return;
      }
      audioContextRef.current = ctx;
      setHasAudioContext(true);
    }

    const ctx = audioContextRef.current;
    if (graphRef.current) {
      graphRef.current.dispose();
      graphRef.current = null;
    }

    const currentValues = useFreeParamStore.getState().values;
    try {
      const graph = compileAudioGraph(resolvedNode, ctx, currentValues);
      graph.masterGain.gain.setValueAtTime(masterVolume * 0.7, ctx.currentTime);
      graphRef.current = graph;

      if (isPlaying) {
        graph.start();
      }
    } catch (err) {
      console.error('[AudioDSPWidget] Failed to compile audio graph from AST:', {
        declarationName,
        error: err,
      });
    }

    return () => {
      if (graphRef.current) {
        graphRef.current.dispose();
        graphRef.current = null;
      }
    };
  }, [resolvedNode, declarationName]);

  // Real-time parameter subscription from FreeParamStore to AudioParams
  useEffect(() => {
    if (!resolvedNode || params.length === 0) return;
    const store = useFreeParamStore.getState();
    const unsubs: Array<() => void> = [];

    for (const p of params) {
      const unsub = store.subscribeParam(p.id, (newVal) => {
        if (graphRef.current) {
          graphRef.current.updateParam(p.id, newVal);
        }
      });
      unsubs.push(unsub);
    }

    return () => {
      for (const unsub of unsubs) {
        unsub();
      }
    };
  }, [resolvedNode, params]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (graphRef.current) {
        graphRef.current.dispose();
        graphRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current
          .close()
          .catch((err) =>
            console.error('[AudioDSPWidget] Error closing AudioContext during cleanup:', { error: err })
          );
        audioContextRef.current = null;
      }
    };
  }, []);

  // Update master volume
  const handleVolumeChange = useCallback((newVol: number) => {
    setMasterVolume(newVol);
    if (graphRef.current && audioContextRef.current) {
      try {
        graphRef.current.masterGain.gain.cancelScheduledValues(audioContextRef.current.currentTime);
        graphRef.current.masterGain.gain.setValueAtTime(
          newVol * 0.7,
          audioContextRef.current.currentTime
        );
      } catch (err) {
        console.error('[AudioDSPWidget] Error updating master gain:', err);
      }
    }
  }, []);

  // Play / Stop toggle
  const handleTogglePlay = useCallback(async () => {
    if (!audioContextRef.current) {
      const ctx = createAudioContext();
      if (!ctx) {
        setHasAudioContext(false);
        return;
      }
      audioContextRef.current = ctx;
      setHasAudioContext(true);
    }

    const ctx = audioContextRef.current;
    await resumeContext(ctx); // timeout-checked resume

    if (isPlaying) {
      if (graphRef.current) {
        graphRef.current.stop();
      }
      setIsPlaying(false);
    } else {
      if (resolvedNode) {
        // If graph was disposed or stopped, recompile or restart
        if (!graphRef.current) {
          const currentValues = useFreeParamStore.getState().values;
          graphRef.current = compileAudioGraph(resolvedNode, ctx, currentValues);
          graphRef.current.masterGain.gain.setValueAtTime(masterVolume * 0.7, ctx.currentTime);
        }
        graphRef.current.start();
      }
      setIsPlaying(true);
    }
  }, [isPlaying, resumeContext, resolvedNode, masterVolume]);

  // Gate / Trig controls
  const handleGateDown = useCallback(async () => {
    if (audioContextRef.current) {
      await resumeContext(audioContextRef.current); // timeout-checked resume
    }
    setIsGateActive(true);
    if (graphRef.current) {
      graphRef.current.triggerAttack();
    }
  }, [resumeContext]);

  const handleGateUp = useCallback(() => {
    setIsGateActive(false);
    if (graphRef.current) {
      graphRef.current.triggerRelease();
    }
  }, []);

  // Reset view / synthesis state
  const handleResetAudio = useCallback(() => {
    if (graphRef.current && audioContextRef.current) {
      graphRef.current.stop();
    }
    setIsPlaying(false);
    setIsGateActive(false);
  }, []);

  // Live 2D Canvas Visualization Loop (Oscilloscope & FFT Spectrogram)
  useEffect(() => {
    if (!canvasRef.current || !hasAudioContext) return;
    const canvas = canvasRef.current;
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext('2d');
    } catch (err) {
      console.log('[AudioDSPWidget] Canvas 2D context not available:', err);
      return;
    }
    if (!ctx) return;

    let isMounted = true;
    let animId: number;

    const render = () => {
      if (!isMounted) return;

      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const displayWidth = Math.floor(rect.width * dpr);
      const displayHeight = Math.floor(rect.height * dpr);

      if (displayWidth > 0 && displayHeight > 0) {
        if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
          canvas.width = displayWidth;
          canvas.height = displayHeight;
        }
      }

      const width = canvas.width || 800;
      const height = canvas.height || 400;

      // Dark acoustic background
      ctx.fillStyle = '#050811';
      ctx.fillRect(0, 0, width, height);

      // Subtle grid coordinates
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      const gridStepX = width / 10;
      for (let x = 0; x < width; x += gridStepX) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      const graph = graphRef.current;
      const analyser = graph?.analyserNode;

      if (analyser) {
        const binCount = analyser.frequencyBinCount || 1024;
        const timeData = new Uint8Array(analyser.fftSize || 2048);
        const freqData = new Uint8Array(binCount);

        analyser.getByteTimeDomainData(timeData);
        analyser.getByteFrequencyData(freqData);

        if (displayMode === 'split') {
          const midY = height / 2;

          // --- Top: Oscilloscope ---
          ctx.save();
          ctx.beginPath();
          ctx.rect(0, 0, width, midY);
          ctx.clip();

          // Zero baseline
          ctx.strokeStyle = 'rgba(0, 245, 255, 0.12)';
          ctx.beginPath();
          ctx.moveTo(0, midY / 2);
          ctx.lineTo(width, midY / 2);
          ctx.stroke();

          // Waveform line with neon cyan glow
          ctx.strokeStyle = '#00f5ff';
          ctx.shadowColor = '#00f5ff';
          ctx.shadowBlur = 8;
          ctx.lineWidth = 2;
          ctx.beginPath();
          const sliceWidth = width / timeData.length;
          let px = 0;
          for (let i = 0; i < timeData.length; i++) {
            const v = timeData[i] / 128.0;
            const py = (v * (midY / 2));
            if (i === 0) {
              ctx.moveTo(px, py);
            } else {
              ctx.lineTo(px, py);
            }
            px += sliceWidth;
          }
          ctx.stroke();
          ctx.restore();

          // Split divider line
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
          ctx.beginPath();
          ctx.moveTo(0, midY);
          ctx.lineTo(width, midY);
          ctx.stroke();

          // --- Bottom: FFT Spectrogram Waterfall ---
          ctx.save();
          ctx.beginPath();
          ctx.rect(0, midY, width, midY);
          ctx.clip();

          const barCount = Math.min(128, binCount);
          const barWidth = width / barCount;
          for (let i = 0; i < barCount; i++) {
            const binIdx = Math.floor(Math.pow(i / barCount, 1.8) * binCount);
            const val = freqData[binIdx] || 0;
            const fraction = val / 255;
            const barH = fraction * (midY - 8);

            const hue = 265 - (i / barCount) * 190;
            ctx.fillStyle = `hsla(${hue}, 90%, ${45 + fraction * 35}%, ${0.35 + fraction * 0.65})`;
            ctx.fillRect(i * barWidth, height - barH, barWidth - 1, barH);
          }
          ctx.restore();

        } else if (displayMode === 'oscilloscope') {
          // Full-screen Oscilloscope
          ctx.strokeStyle = 'rgba(0, 245, 255, 0.15)';
          ctx.beginPath();
          ctx.moveTo(0, height / 2);
          ctx.lineTo(width, height / 2);
          ctx.stroke();

          ctx.strokeStyle = '#00f5ff';
          ctx.shadowColor = '#00f5ff';
          ctx.shadowBlur = 10;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          const sliceWidth = width / timeData.length;
          let px = 0;
          for (let i = 0; i < timeData.length; i++) {
            const v = timeData[i] / 128.0;
            const py = (v * (height / 2));
            if (i === 0) {
              ctx.moveTo(px, py);
            } else {
              ctx.lineTo(px, py);
            }
            px += sliceWidth;
          }
          ctx.stroke();

        } else if (displayMode === 'fft') {
          // Full-screen 2048-pt FFT Spectrogram
          const barCount = Math.min(256, binCount);
          const barWidth = width / barCount;
          for (let i = 0; i < barCount; i++) {
            const binIdx = Math.floor(Math.pow(i / barCount, 1.8) * binCount);
            const val = freqData[binIdx] || 0;
            const fraction = val / 255;
            const barH = fraction * (height - 12);

            const hue = 270 - (i / barCount) * 200;
            ctx.fillStyle = `hsla(${hue}, 95%, ${45 + fraction * 35}%, ${0.4 + fraction * 0.6})`;
            ctx.fillRect(i * barWidth, height - barH, barWidth - 1, barH);
          }
        }
      } else {
        // Idle calm oscilloscope zero-line
        ctx.strokeStyle = 'rgba(0, 245, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => {
      isMounted = false;
      cancelAnimationFrame(animId);
    };
  }, [hasAudioContext, displayMode]);

  return (
    <div
      ref={containerRef}
      className="audio-dsp-container"
      data-testid="audio-dsp-widget"
    >
      {/* Audio DSP Audition & Visualization Toolbar */}
      <div className="audio-toolbar" data-testid="audio-toolbar">
        {/* Play/Stop & Audition Controls */}
        <div className="audio-toolbar-group">
          <button
            type="button"
            className={`audio-btn ${isPlaying ? 'active' : ''}`}
            onClick={handleTogglePlay}
            data-testid="audio-play-stop-btn"
            title={isPlaying ? 'Stop playback' : 'Play audio patch'}
          >
            {isPlaying ? '⏹ Stop' : '▶ Play'}
          </button>

          <button
            type="button"
            className={`audio-btn trig ${isGateActive ? 'active' : ''}`}
            onPointerDown={handleGateDown}
            onPointerUp={handleGateUp}
            onPointerLeave={handleGateUp}
            data-testid="audio-trig-btn"
            title="Press and hold to trigger ADSR envelope gate"
          >
            ⚡ Trig / Gate
          </button>

          <button
            type="button"
            className={`audio-btn ${isLooping ? 'active' : ''}`}
            onClick={() => setIsLooping((prev) => !prev)}
            data-testid="audio-loop-toggle-btn"
            title="Toggle loop mode"
          >
            🔁 Loop
          </button>
        </div>

        {/* Master Volume Slider */}
        <div className="audio-slider-container">
          <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>Vol:</span>
          <input
            type="range"
            className="audio-slider"
            min="0"
            max="1"
            step="0.05"
            value={masterVolume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
            data-testid="audio-volume-slider"
            title={`Master volume (${Math.round(masterVolume * 100)}%)`}
          />
          <span style={{ fontSize: '0.6875rem', fontFamily: 'var(--font-mono)', minWidth: '32px' }}>
            {Math.round(masterVolume * 100)}%
          </span>
        </div>

        {/* Display Mode Switcher */}
        <div className="audio-toolbar-group">
          <button
            type="button"
            className={`audio-btn ${displayMode === 'split' ? 'active' : ''}`}
            onClick={() => setDisplayMode('split')}
            data-testid="audio-display-mode-split"
            title="Split: Oscilloscope & FFT Spectrum"
          >
            Split
          </button>
          <button
            type="button"
            className={`audio-btn ${displayMode === 'oscilloscope' ? 'active' : ''}`}
            onClick={() => setDisplayMode('oscilloscope')}
            data-testid="audio-display-mode-oscilloscope"
            title="Full Oscilloscope"
          >
            Oscilloscope
          </button>
          <button
            type="button"
            className={`audio-btn ${displayMode === 'fft' ? 'active' : ''}`}
            onClick={() => setDisplayMode('fft')}
            data-testid="audio-display-mode-fft"
            title="Full 2048-pt FFT Spectrogram"
          >
            FFT Spectrum
          </button>
        </div>

        {/* Parameters & Reset Actions */}
        <div className="audio-toolbar-group">
          <button
            type="button"
            className={`audio-btn ${showParameters ? 'active' : ''}`}
            onClick={() => setShowParameters((prev) => !prev)}
            data-testid="toggle-audio-parameters-btn"
            title="Toggle in-viewport synth parameter drawer"
          >
            Parameters {params.length > 0 ? `(${params.length})` : ''}
          </button>

          <button
            type="button"
            className="audio-btn"
            onClick={handleResetAudio}
            data-testid="reset-audio-btn"
            title="Reset audio state"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Floating In-Viewport Parameter Scrubber Overlay */}
      {showParameters && (
        <AudioParameterOverlay
          params={params}
          onClose={() => setShowParameters(false)}
        />
      )}

      {/* Live Audio Visualizer Canvas or Headless Fallback */}
      {hasAudioContext ? (
        <div className="audio-canvas-wrapper" data-testid="audio-canvas-wrapper">
          <canvas
            ref={canvasRef}
            className="audio-canvas"
            data-testid="audio-visualizer-canvas"
          />
        </div>
      ) : (
        /* Fallback View for Headless / JSDOM / Unsupported Devices */
        <div className="widget-stage" data-testid="audio-dsp-fallback">
          <div className="stage-card">
            <div className="stage-icon" aria-hidden="true">🎛️</div>
            <h2 className="stage-title">WebAudio DSP Synthesis & FFT</h2>
            <p className="stage-description">
              Real-time procedural WebAudio synthesis graph with live FFT frequency waterfall and oscilloscope for{' '}
              <code>{declarationName || 'AudioNode'}</code>.
            </p>
            <div className="domain-tags">
              <span className="domain-tag">WebAudio Graph</span>
              <span className="domain-tag">2048-pt FFT</span>
              <span className="domain-tag">Zero-Latency DSP</span>
              <span className="domain-tag">Acoustic Limiter</span>
            </div>
            {params.length > 0 && (
              <div style={{ marginTop: '16px', width: '100%', textAlign: 'left' }}>
                <span
                  style={{
                    fontSize: '0.6875rem',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}
                >
                  Bound Parameters ({params.length})
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
                  {params.slice(0, 5).map((p) => (
                    <div
                      key={p.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        fontSize: '0.75rem',
                        background: 'var(--bg-surface)',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        border: '1px solid var(--border-color)',
                      }}
                    >
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                        {p.id}
                      </span>
                      <span style={{ color: 'var(--text-secondary)' }}>
                        [{p.minVal}, {p.maxVal}] (default: {p.defaultVal})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
