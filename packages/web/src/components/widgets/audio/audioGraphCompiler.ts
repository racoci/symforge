/**
 * WebAudio DSP Graph Compiler for SymForge AudioNode AST.
 *
 * Compiles formal functional AudioNode domain expressions into executable WebAudio
 * graphs with real-time parameter modulation, dual-tap visualizer routing, ADSR
 * envelope scheduling, and acoustic safety limiting.
 */

import type {
  AudioNode as ASTAudioNode,
  SineAudioNode,
  SawAudioNode,
  SquareAudioNode,
  NoiseAudioNode,
  ADSRAudioNode,
  LowPassAudioNode,
  GainAudioNode,
  MixAudioNode,
  FMAudioNode,
  FreeParam,
} from '../../../core/ast/types';

export interface CompiledAudioSource {
  start: (t?: number) => void;
  stop: (t?: number) => void;
  isStarted: boolean;
}

export interface CompiledAudioGraph {
  audioContext: AudioContext;
  outputNode: globalThis.AudioNode;
  analyserNode: AnalyserNode;
  compressor: DynamicsCompressorNode;
  masterGain: GainNode;
  paramBindings: Map<string, AudioParam[]>;
  sources: CompiledAudioSource[];
  triggerAttack: (t?: number) => void;
  triggerRelease: (t?: number) => void;
  start: (t?: number) => void;
  stop: (t?: number) => void;
  updateParam: (id: string, value: number, time?: number) => void;
  dispose: () => void;
}

interface ADSREnvelopeDescriptor {
  envGain: GainNode;
  attackParam: FreeParam;
  decayParam: FreeParam;
  sustainParam: FreeParam;
  releaseParam: FreeParam;
}

/**
 * Checks whether WebAudio AudioContext is available in the execution environment.
 */
export function isAudioContextAvailable(): boolean {
  if (typeof window === 'undefined') return false;
  return Boolean(
    'AudioContext' in window ||
    ('webkitAudioContext' in (window as unknown as Record<string, unknown>))
  );
}

/**
 * Instantiates a standard or prefixed AudioContext instance.
 */
export function createAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const AudioCtx =
    window.AudioContext ||
    ((window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
  if (!AudioCtx) return null;
  try {
    return new AudioCtx();
  } catch (err) {
    console.error('[createAudioContext] Failed to initialize AudioContext in current environment:', err);
    return null;
  }
}

/**
 * Clamps a parameter value to safe finite bounds.
 */
function getClampedParam(
  param: FreeParam,
  values: Record<string, number>,
  fallbackMin: number,
  fallbackMax: number
): number {
  const raw = values[param.id] ?? param.defaultVal;
  if (!Number.isFinite(raw)) return param.defaultVal;
  const min = Number.isFinite(param.minVal) ? param.minVal : fallbackMin;
  const max = Number.isFinite(param.maxVal) ? param.maxVal : fallbackMax;
  return Math.max(min, Math.min(max, raw));
}

/**
 * Compiles a formal SymForge AudioNode AST into an executable WebAudio DSP graph.
 *
 * Signal Architecture:
 * - Synthesis Graph -> rawOutputNode
 * - Dual-Tap Output:
 *   - Tap 1 (Visualizer): rawOutputNode -> AnalyserNode (uncompressed 2048-pt FFT & Oscilloscope)
 *   - Tap 2 (Acoustic Safety): rawOutputNode -> DynamicsCompressorNode -> MasterGainNode (0.7) -> Destination
 *
 * @param ast Root AudioNode AST expression
 * @param context WebAudio AudioContext
 * @param initialParamValues Optional map of FreeParam id -> initial numeric values
 */
export function compileAudioGraph(
  ast: ASTAudioNode,
  context: AudioContext,
  initialParamValues: Record<string, number> = {}
): CompiledAudioGraph {
  const paramValues: Record<string, number> = { ...initialParamValues };
  const paramBindings = new Map<string, AudioParam[]>();
  const customParamUpdaters = new Map<string, Array<(val: number, t: number) => void>>();
  const sources: CompiledAudioSource[] = [];
  const adsrEnvelopes: ADSREnvelopeDescriptor[] = [];
  const allCreatedNodes = new Set<globalThis.AudioNode>();

  function registerParamBinding(id: string, param: AudioParam): void {
    const list = paramBindings.get(id);
    if (list) {
      list.push(param);
    } else {
      paramBindings.set(id, [param]);
    }
  }

  function registerCustomUpdater(id: string, updater: (val: number, t: number) => void): void {
    const list = customParamUpdaters.get(id);
    if (list) {
      list.push(updater);
    } else {
      customParamUpdaters.set(id, [updater]);
    }
  }

  function wrapSource(node: { start: (t?: number) => void; stop: (t?: number) => void }): CompiledAudioSource {
    const wrapper: CompiledAudioSource = {
      isStarted: false,
      start: (t?: number) => {
        if (!wrapper.isStarted) {
          try {
            node.start(t ?? context.currentTime);
            wrapper.isStarted = true;
          } catch (err) {
            console.error('[CompiledAudioSource] Failed to start audio source node:', err);
          }
        }
      },
      stop: (t?: number) => {
        if (wrapper.isStarted) {
          try {
            node.stop(t ?? context.currentTime);
          } catch (err) {
            console.error('[CompiledAudioSource] Failed to stop audio source node:', err);
          }
          wrapper.isStarted = false;
        }
      },
    };
    sources.push(wrapper);
    return wrapper;
  }

  // --- Primitives Compilation Helpers ---

  function compileSine(node: SineAudioNode): globalThis.AudioNode {
    const osc = context.createOscillator();
    allCreatedNodes.add(osc);
    osc.type = 'sine';

    const freqVal = getClampedParam(node.freq, paramValues, 10, 20000);
    osc.frequency.setValueAtTime(freqVal, context.currentTime);
    registerParamBinding(node.freq.id, osc.frequency);

    wrapSource(osc);

    if (node.phase) {
      const delay = context.createDelay(1.0);
      allCreatedNodes.add(delay);
      const phaseVal = getClampedParam(node.phase, paramValues, 0, 6.2831853);
      // Phase delay approximation: d = phi / (2 * pi * f)
      const initialDelay = Math.max(0, Math.min(0.1, phaseVal / (2 * Math.PI * Math.max(20, freqVal))));
      delay.delayTime.setValueAtTime(initialDelay, context.currentTime);
      registerParamBinding(node.phase.id, delay.delayTime);

      registerCustomUpdater(node.phase.id, (val, t) => {
        const curFreq = paramValues[node.freq.id] ?? node.freq.defaultVal;
        const d = Math.max(0, Math.min(0.1, val / (2 * Math.PI * Math.max(20, curFreq))));
        try {
          delay.delayTime.cancelScheduledValues(t);
          delay.delayTime.setValueAtTime(d, t);
        } catch (err) {
          console.error('[compileSine] Error setting phase delay:', err);
        }
      });

      osc.connect(delay);
      return delay;
    }

    return osc;
  }

  function compileSaw(node: SawAudioNode): globalThis.AudioNode {
    const osc = context.createOscillator();
    allCreatedNodes.add(osc);
    osc.type = 'sawtooth';

    const freqVal = getClampedParam(node.freq, paramValues, 10, 20000);
    osc.frequency.setValueAtTime(freqVal, context.currentTime);
    registerParamBinding(node.freq.id, osc.frequency);

    wrapSource(osc);
    return osc;
  }

  function compileSquare(node: SquareAudioNode): globalThis.AudioNode {
    const osc = context.createOscillator();
    allCreatedNodes.add(osc);
    osc.type = 'square';

    const freqVal = getClampedParam(node.freq, paramValues, 10, 20000);
    osc.frequency.setValueAtTime(freqVal, context.currentTime);
    registerParamBinding(node.freq.id, osc.frequency);

    if (node.duty) {
      // Track duty cycle param binding
      const dutyGain = context.createGain();
      allCreatedNodes.add(dutyGain);
      const dutyVal = getClampedParam(node.duty, paramValues, 0.01, 0.99);
      dutyGain.gain.setValueAtTime(dutyVal, context.currentTime);
      registerParamBinding(node.duty.id, dutyGain.gain);
    }

    wrapSource(osc);
    return osc;
  }

  function compileNoise(node: NoiseAudioNode): globalThis.AudioNode {
    const sampleRate = context.sampleRate || 44100;
    const bufferSize = Math.max(1, Math.floor(sampleRate * 2));
    const buffer = context.createBuffer(1, bufferSize, sampleRate);
    const channelData = buffer.getChannelData(0);

    let seed = ((node.seed ?? 42) >>> 0) || 1337;
    for (let i = 0; i < bufferSize; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      channelData[i] = (seed / 2147483648) - 1.0;
    }

    const noiseSource = context.createBufferSource();
    allCreatedNodes.add(noiseSource);
    noiseSource.buffer = buffer;
    noiseSource.loop = true;

    wrapSource(noiseSource);
    return noiseSource;
  }

  function compileLowPass(node: LowPassAudioNode): globalThis.AudioNode {
    const filter = context.createBiquadFilter();
    allCreatedNodes.add(filter);
    filter.type = 'lowpass';

    const cutoffVal = getClampedParam(node.cutoff, paramValues, 10, 20000);
    const resVal = getClampedParam(node.resonance, paramValues, 0.001, 30);

    filter.frequency.setValueAtTime(cutoffVal, context.currentTime);
    filter.Q.setValueAtTime(resVal, context.currentTime);

    registerParamBinding(node.cutoff.id, filter.frequency);
    registerParamBinding(node.resonance.id, filter.Q);

    const child = compileSubtree(node.input);
    child.connect(filter);
    return filter;
  }

  function compileGain(node: GainAudioNode): globalThis.AudioNode {
    const gainNode = context.createGain();
    allCreatedNodes.add(gainNode);

    const gainVal = getClampedParam(node.gain, paramValues, 0, 10);
    gainNode.gain.setValueAtTime(gainVal, context.currentTime);
    registerParamBinding(node.gain.id, gainNode.gain);

    const child = compileSubtree(node.input);
    child.connect(gainNode);
    return gainNode;
  }

  function compileMix(node: MixAudioNode): globalThis.AudioNode {
    const branch1 = compileSubtree(node.input1);
    const branch2 = compileSubtree(node.input2);

    const gain1 = context.createGain();
    const gain2 = context.createGain();
    const sumNode = context.createGain();
    allCreatedNodes.add(gain1);
    allCreatedNodes.add(gain2);
    allCreatedNodes.add(sumNode);

    const ratioVal = getClampedParam(node.ratio, paramValues, 0, 1);
    gain1.gain.setValueAtTime(1.0 - ratioVal, context.currentTime);
    gain2.gain.setValueAtTime(ratioVal, context.currentTime);
    sumNode.gain.setValueAtTime(1.0, context.currentTime);

    registerParamBinding(node.ratio.id, gain2.gain);

    registerCustomUpdater(node.ratio.id, (val, t) => {
      const r = Math.max(0, Math.min(1, val));
      try {
        gain1.gain.cancelScheduledValues(t);
        gain1.gain.setValueAtTime(1.0 - r, t);
        gain2.gain.cancelScheduledValues(t);
        gain2.gain.setValueAtTime(r, t);
      } catch (err) {
        console.error('[compileMix] Error updating mix ratio branch gains:', err);
      }
    });

    branch1.connect(gain1);
    gain1.connect(sumNode);

    branch2.connect(gain2);
    gain2.connect(sumNode);

    return sumNode;
  }

  function compileADSR(node: ADSRAudioNode): globalThis.AudioNode {
    const gateChild = compileSubtree(node.gate);
    const envGain = context.createGain();
    allCreatedNodes.add(envGain);

    // Initial state: silent until triggered
    envGain.gain.setValueAtTime(0.0, context.currentTime);

    // Register parameter bindings for ADSR envelope knobs
    const dummyAttackGain = context.createGain();
    const dummyDecayGain = context.createGain();
    const dummySustainGain = context.createGain();
    const dummyReleaseGain = context.createGain();
    allCreatedNodes.add(dummyAttackGain);
    allCreatedNodes.add(dummyDecayGain);
    allCreatedNodes.add(dummySustainGain);
    allCreatedNodes.add(dummyReleaseGain);

    dummyAttackGain.gain.setValueAtTime(node.attack.defaultVal, context.currentTime);
    dummyDecayGain.gain.setValueAtTime(node.decay.defaultVal, context.currentTime);
    dummySustainGain.gain.setValueAtTime(node.sustain.defaultVal, context.currentTime);
    dummyReleaseGain.gain.setValueAtTime(node.release.defaultVal, context.currentTime);

    registerParamBinding(node.attack.id, dummyAttackGain.gain);
    registerParamBinding(node.decay.id, dummyDecayGain.gain);
    registerParamBinding(node.sustain.id, dummySustainGain.gain);
    registerParamBinding(node.release.id, dummyReleaseGain.gain);

    adsrEnvelopes.push({
      envGain,
      attackParam: node.attack,
      decayParam: node.decay,
      sustainParam: node.sustain,
      releaseParam: node.release,
    });

    gateChild.connect(envGain);
    return envGain;
  }

  function compileFM(node: FMAudioNode): globalThis.AudioNode {
    const carrierOsc = context.createOscillator();
    const modOsc = context.createOscillator();
    const modGain = context.createGain();
    allCreatedNodes.add(carrierOsc);
    allCreatedNodes.add(modOsc);
    allCreatedNodes.add(modGain);

    carrierOsc.type = 'sine';
    modOsc.type = 'sine';

    const carrierFreq = getClampedParam(node.carrier, paramValues, 10, 20000);
    const modFreq = getClampedParam(node.modFreq, paramValues, 0.1, 20000);
    const modIndex = getClampedParam(node.modIndex, paramValues, 0, 50);

    carrierOsc.frequency.setValueAtTime(carrierFreq, context.currentTime);
    modOsc.frequency.setValueAtTime(modFreq, context.currentTime);

    // Frequency deviation Delta f = I * f_m
    const initialDeviation = modIndex * modFreq;
    modGain.gain.setValueAtTime(initialDeviation, context.currentTime);

    registerParamBinding(node.carrier.id, carrierOsc.frequency);
    registerParamBinding(node.modFreq.id, modOsc.frequency);
    registerParamBinding(node.modIndex.id, modGain.gain);

    registerCustomUpdater(node.modIndex.id, (val, t) => {
      const curModFreq = paramValues[node.modFreq.id] ?? node.modFreq.defaultVal;
      try {
        modGain.gain.cancelScheduledValues(t);
        modGain.gain.setValueAtTime(val * curModFreq, t);
      } catch (err) {
        console.error('[compileFM] Error updating FM modulation gain:', err);
      }
    });

    registerCustomUpdater(node.modFreq.id, (val, t) => {
      const curModIndex = paramValues[node.modIndex.id] ?? node.modIndex.defaultVal;
      try {
        modGain.gain.cancelScheduledValues(t);
        modGain.gain.setValueAtTime(curModIndex * val, t);
      } catch (err) {
        console.error('[compileFM] Error updating FM modulator frequency scaling:', err);
      }
    });

    modOsc.connect(modGain);
    modGain.connect(carrierOsc.frequency);

    wrapSource(carrierOsc);
    wrapSource(modOsc);

    return carrierOsc;
  }

  function compileSubtree(node: ASTAudioNode): globalThis.AudioNode {
    switch (node.kind) {
      case 'sine':
        return compileSine(node);
      case 'saw':
        return compileSaw(node);
      case 'square':
        return compileSquare(node);
      case 'noise':
        return compileNoise(node);
      case 'lowpass':
        return compileLowPass(node);
      case 'gain':
        return compileGain(node);
      case 'mix':
        return compileMix(node);
      case 'adsr':
        return compileADSR(node);
      case 'fm':
        return compileFM(node);
      default: {
        // Fallback for hole, macro, or unknown audio kinds
        const silentNode = context.createGain();
        allCreatedNodes.add(silentNode);
        silentNode.gain.setValueAtTime(0.0, context.currentTime);
        return silentNode;
      }
    }
  }

  // Compile AST starting at root
  const rawOutputNode = compileSubtree(ast);

  // Audition Gate for graphs without explicit ADSR envelopes
  const auditionGate = context.createGain();
  allCreatedNodes.add(auditionGate);
  auditionGate.gain.setValueAtTime(adsrEnvelopes.length > 0 ? 1.0 : 1.0, context.currentTime);
  rawOutputNode.connect(auditionGate);

  // 1. Visualizer Tap (Uncompressed, dry signal directly to 2048-pt FFT & Oscilloscope)
  const analyserNode = context.createAnalyser();
  allCreatedNodes.add(analyserNode);
  analyserNode.fftSize = 2048;
  analyserNode.minDecibels = -90;
  analyserNode.maxDecibels = -10;
  analyserNode.smoothingTimeConstant = 0.8;
  auditionGate.connect(analyserNode);

  // 2. Acoustic Limiter Tap (Dynamics Compressor + Master Gain to protect speaker & ear safety)
  const compressor = context.createDynamicsCompressor();
  allCreatedNodes.add(compressor);
  compressor.threshold.setValueAtTime(-12, context.currentTime);
  compressor.knee.setValueAtTime(30, context.currentTime);
  compressor.ratio.setValueAtTime(12, context.currentTime);
  compressor.attack.setValueAtTime(0.003, context.currentTime);
  compressor.release.setValueAtTime(0.25, context.currentTime);

  const masterGain = context.createGain();
  allCreatedNodes.add(masterGain);
  masterGain.gain.setValueAtTime(0.7, context.currentTime);

  auditionGate.connect(compressor);
  compressor.connect(masterGain);
  masterGain.connect(context.destination);

  // --- Envelope Triggers ---

  const triggerAttack = (t?: number): void => {
    const now = t ?? context.currentTime;

    if (adsrEnvelopes.length === 0) {
      // Default gate attack
      try {
        auditionGate.gain.cancelScheduledValues(now);
        auditionGate.gain.setValueAtTime(auditionGate.gain.value, now);
        auditionGate.gain.linearRampToValueAtTime(1.0, now + 0.01);
      } catch (err) {
        console.error('[triggerAttack] Error ramping audition gate:', err);
      }
      return;
    }

    for (const env of adsrEnvelopes) {
      const att = Math.max(0.001, paramValues[env.attackParam.id] ?? env.attackParam.defaultVal);
      const dec = Math.max(0.001, paramValues[env.decayParam.id] ?? env.decayParam.defaultVal);
      const sus = Math.max(0.0, Math.min(1.0, paramValues[env.sustainParam.id] ?? env.sustainParam.defaultVal));

      try {
        env.envGain.gain.cancelScheduledValues(now);
        env.envGain.gain.setValueAtTime(env.envGain.gain.value, now);
        env.envGain.gain.linearRampToValueAtTime(1.0, now + att);
        env.envGain.gain.linearRampToValueAtTime(sus, now + att + dec);
      } catch (err) {
        console.error('[triggerAttack] Error scheduling ADSR attack ramp:', err);
      }
    }
  };

  const triggerRelease = (t?: number): void => {
    const now = t ?? context.currentTime;

    if (adsrEnvelopes.length === 0) {
      // Default gate release
      try {
        auditionGate.gain.cancelScheduledValues(now);
        auditionGate.gain.setValueAtTime(auditionGate.gain.value, now);
        auditionGate.gain.linearRampToValueAtTime(0.0001, now + 0.05);
        auditionGate.gain.setValueAtTime(0.0, now + 0.051);
      } catch (err) {
        console.error('[triggerRelease] Error ramping down audition gate:', err);
      }
      return;
    }

    for (const env of adsrEnvelopes) {
      const rel = Math.max(0.001, paramValues[env.releaseParam.id] ?? env.releaseParam.defaultVal);
      try {
        env.envGain.gain.cancelScheduledValues(now);
        env.envGain.gain.setValueAtTime(env.envGain.gain.value, now);
        env.envGain.gain.linearRampToValueAtTime(0.0001, now + rel);
        env.envGain.gain.setValueAtTime(0.0, now + rel + 0.001);
      } catch (err) {
        console.error('[triggerRelease] Error scheduling ADSR release ramp:', err);
      }
    }
  };

  const start = (t?: number): void => {
    const now = t ?? context.currentTime;
    for (const source of sources) {
      if (!source.isStarted) {
        source.start(now);
      }
    }
    triggerAttack(now);
  };

  const stop = (t?: number): void => {
    const now = t ?? context.currentTime;
    triggerRelease(now);
    for (const source of sources) {
      if (source.isStarted) {
        source.stop(now);
      }
    }
  };

  const updateParam = (id: string, value: number, time?: number): void => {
    const now = time ?? context.currentTime;
    paramValues[id] = value;

    const boundParams = paramBindings.get(id);
    if (boundParams) {
      for (const p of boundParams) {
        try {
          p.cancelScheduledValues(now);
          p.setValueAtTime(value, now);
        } catch (err) {
          console.error('[updateParam] Failed to schedule value on AudioParam:', err);
          p.value = value;
        }
      }
    }

    const updaters = customParamUpdaters.get(id);
    if (updaters) {
      for (const updater of updaters) {
        try {
          updater(value, now);
        } catch (err) {
          console.error('[updateParam] Custom updater failed for parameter:', err);
        }
      }
    }
  };

  const dispose = (): void => {
    for (const source of sources) {
      if (source.isStarted) {
        try {
          source.stop();
        } catch (err) {
          console.error('[dispose] Error stopping audio source during cleanup:', err);
        }
      }
    }
    sources.length = 0;

    for (const node of allCreatedNodes) {
      try {
        node.disconnect();
      } catch (err) {
        console.error('[dispose] Error disconnecting node during cleanup:', err);
      }
    }
    allCreatedNodes.clear();

    try {
      analyserNode.disconnect();
    } catch (err) {
      console.error('[dispose] AnalyserNode disconnect failed:', err);
    }
    try {
      compressor.disconnect();
    } catch (err) {
      console.error('[dispose] Compressor disconnect failed:', err);
    }
    try {
      masterGain.disconnect();
    } catch (err) {
      console.error('[dispose] MasterGain disconnect failed:', err);
    }

    paramBindings.clear();
    customParamUpdaters.clear();
    adsrEnvelopes.length = 0;
  };

  return {
    audioContext: context,
    outputNode: rawOutputNode,
    analyserNode,
    compressor,
    masterGain,
    paramBindings,
    sources,
    triggerAttack,
    triggerRelease,
    start,
    stop,
    updateParam,
    dispose,
  };
}
