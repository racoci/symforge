/**
 * Mock Web Audio API implementation for unit and component testing in headless JSDOM environments.
 */

export class MockAudioParam {
  value: number;
  defaultValue: number;
  minValue: number;
  maxValue: number;
  scheduledEvents: Array<{ type: string; value: number; time: number }>;

  constructor(defaultValue = 1.0, min = -Infinity, max = Infinity) {
    this.value = defaultValue;
    this.defaultValue = defaultValue;
    this.minValue = min;
    this.maxValue = max;
    this.scheduledEvents = [];
  }

  setValueAtTime(val: number, time: number): this {
    this.value = val;
    this.scheduledEvents.push({ type: 'setValueAtTime', value: val, time });
    return this;
  }

  linearRampToValueAtTime(val: number, time: number): this {
    this.value = val;
    this.scheduledEvents.push({ type: 'linearRampToValueAtTime', value: val, time });
    return this;
  }

  exponentialRampToValueAtTime(val: number, time: number): this {
    this.value = val;
    this.scheduledEvents.push({ type: 'exponentialRampToValueAtTime', value: val, time });
    return this;
  }

  setTargetAtTime(target: number, time: number, _timeConstant: number): this {
    this.value = target;
    this.scheduledEvents.push({ type: 'setTargetAtTime', value: target, time });
    return this;
  }

  cancelScheduledValues(time: number): this {
    this.scheduledEvents = this.scheduledEvents.filter((e) => e.time < time);
    return this;
  }
}

export class MockAudioNode {
  context: MockAudioContext;
  connectedTo: any[] = [];
  numberOfInputs = 1;
  numberOfOutputs = 1;

  constructor(context: MockAudioContext) {
    this.context = context;
  }

  connect(destination: any): any {
    this.connectedTo.push(destination);
    return destination;
  }

  disconnect(destination?: any): void {
    if (!destination) {
      this.connectedTo = [];
    } else {
      this.connectedTo = this.connectedTo.filter((d) => d !== destination);
    }
  }
}

export class MockOscillatorNode extends MockAudioNode {
  type: OscillatorType = 'sine';
  frequency: MockAudioParam = new MockAudioParam(440, 0, 20000);
  detune: MockAudioParam = new MockAudioParam(0, -153600, 153600);
  isStarted = false;
  isStopped = false;

  start(_time?: number): void {
    this.isStarted = true;
  }

  stop(_time?: number): void {
    this.isStopped = true;
    this.isStarted = false;
  }
}

export class MockGainNode extends MockAudioNode {
  gain: MockAudioParam = new MockAudioParam(1.0, 0, 10);
}

export class MockBiquadFilterNode extends MockAudioNode {
  type: BiquadFilterType = 'lowpass';
  frequency: MockAudioParam = new MockAudioParam(350, 10, 20000);
  Q: MockAudioParam = new MockAudioParam(1.0, 0.0001, 1000);
  detune: MockAudioParam = new MockAudioParam(0, -153600, 153600);
  gain: MockAudioParam = new MockAudioParam(0, -40, 40);
}

export class MockAudioBufferSourceNode extends MockAudioNode {
  buffer: any = null;
  loop = false;
  isStarted = false;
  isStopped = false;

  start(_time?: number): void {
    this.isStarted = true;
  }

  stop(_time?: number): void {
    this.isStopped = true;
    this.isStarted = false;
  }
}

export class MockAnalyserNode extends MockAudioNode {
  fftSize = 2048;
  frequencyBinCount = 1024;
  minDecibels = -90;
  maxDecibels = -10;
  smoothingTimeConstant = 0.8;

  getByteTimeDomainData(array: Uint8Array): void {
    for (let i = 0; i < array.length; i++) {
      array[i] = 128;
    }
  }

  getByteFrequencyData(array: Uint8Array): void {
    for (let i = 0; i < array.length; i++) {
      array[i] = Math.max(0, Math.min(255, Math.floor(128 - (i % 64))));
    }
  }
}

export class MockDynamicsCompressorNode extends MockAudioNode {
  threshold: MockAudioParam = new MockAudioParam(-12, -100, 0);
  knee: MockAudioParam = new MockAudioParam(30, 0, 40);
  ratio: MockAudioParam = new MockAudioParam(12, 1, 20);
  attack: MockAudioParam = new MockAudioParam(0.003, 0, 1);
  release: MockAudioParam = new MockAudioParam(0.25, 0, 1);
}

export class MockDelayNode extends MockAudioNode {
  delayTime: MockAudioParam = new MockAudioParam(0, 0, 1);
}

export class MockAudioContext {
  currentTime = 0.0;
  sampleRate = 44100;
  state: AudioContextState = 'running';
  destination: MockAudioNode;

  constructor() {
    this.destination = new MockAudioNode(this);
  }

  createOscillator(): MockOscillatorNode {
    return new MockOscillatorNode(this);
  }

  createGain(): MockGainNode {
    return new MockGainNode(this);
  }

  createBiquadFilter(): MockBiquadFilterNode {
    return new MockBiquadFilterNode(this);
  }

  createBuffer(channels: number, length: number, sampleRate: number): any {
    const channelData = new Float32Array(length);
    return {
      numberOfChannels: channels,
      length,
      sampleRate,
      duration: length / sampleRate,
      getChannelData: (_channelIndex: number) => channelData,
    };
  }

  createBufferSource(): MockAudioBufferSourceNode {
    return new MockAudioBufferSourceNode(this);
  }

  createAnalyser(): MockAnalyserNode {
    return new MockAnalyserNode(this);
  }

  createDynamicsCompressor(): MockDynamicsCompressorNode {
    return new MockDynamicsCompressorNode(this);
  }

  createDelay(_maxDelayTime = 1.0): MockDelayNode {
    return new MockDelayNode(this);
  }

  async resume(): Promise<void> {
    this.state = 'running';
  }

  async suspend(): Promise<void> {
    this.state = 'suspended';
  }

  async close(): Promise<void> {
    this.state = 'closed';
  }
}
