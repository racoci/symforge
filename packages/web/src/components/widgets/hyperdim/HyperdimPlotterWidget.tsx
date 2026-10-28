import React, {
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
} from 'react';
import * as THREE from 'three';
import type { DomainWidgetProps } from '../types';
import type { ASTNode } from '../../../core/ast/types';
import type {
  HyperdimFunction,
  ColorMapType,
  HyperdimDisplayMode,
  GridEvaluationResult,
} from './types';
import {
  BENCHMARK_FUNCTIONS,
  createFunctionFromAST,
  evaluate2DSlice,
  sampleColorMap,
  sampleColorMap255,
} from './functionEvaluator';

/**
 * Checks whether WebGL rendering context is supported in the current environment.
 */
export function isWebGLAvailable(): boolean {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return false;
  }
  if (!('WebGLRenderingContext' in window) || !window.WebGLRenderingContext) {
    return false;
  }
  try {
    const canvas = document.createElement('canvas');
    const gl =
      canvas.getContext('webgl') ||
      canvas.getContext('experimental-webgl') ||
      canvas.getContext('webgl2');
    return Boolean(gl);
  } catch (err) {
    console.error('[HyperdimPlotter] WebGL availability check threw exception:', {
      error: err,
      timestamp: Date.now(),
    });
    return false;
  }
}

/**
 * Checks whether standard 2D Canvas rendering is supported.
 */
export function isCanvasAvailable(): boolean {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return false;
  }
  try {
    const canvas = document.createElement('canvas');
    if (!canvas.getContext) return false;
    const ctx = canvas.getContext('2d');
    if (!ctx || typeof ctx.createImageData !== 'function') return false;
    const imgData = ctx.createImageData(1, 1);
    return Boolean(imgData && imgData.data && imgData.data.length >= 4);
  } catch (err) {
    console.error('[HyperdimPlotter] Canvas 2D check threw exception:', {
      error: err,
      timestamp: Date.now(),
    });
    return false;
  }
}

interface OrthogonalScrubberOverlayProps {
  readonly fn: HyperdimFunction;
  readonly axisX: number;
  readonly axisY: number;
  readonly sliceCoords: Float32Array;
  readonly onCoordinateChange: (dimIndex: number, val: number) => void;
  readonly onResetCoordinate: (dimIndex: number) => void;
  readonly onResetAll: () => void;
  readonly onClose: () => void;
}

/**
 * In-viewport floating drawer overlay for scrubbing orthogonal dimensions.
 */
export const OrthogonalScrubberOverlay: React.FC<OrthogonalScrubberOverlayProps> = React.memo(
  ({
    fn,
    axisX,
    axisY,
    sliceCoords,
    onCoordinateChange,
    onResetCoordinate,
    onResetAll,
    onClose,
  }) => {
    // Unselected orthogonal dimensions: k != axisX && k != axisY
    const orthogonalIndices = useMemo(() => {
      const indices: number[] = [];
      for (let k = 0; k < fn.inputDim; k++) {
        if (k !== axisX && k !== axisY) {
          indices.push(k);
        }
      }
      return indices;
    }, [fn.inputDim, axisX, axisY]);

    return (
      <div className="hyperdim-param-overlay" data-testid="hyperdim-param-overlay">
        <div className="hyperdim-param-overlay-header">
          <span className="hyperdim-param-overlay-title">
            Orthogonal Slices ({orthogonalIndices.length})
          </span>
          <div className="hyperdim-param-overlay-actions">
            {orthogonalIndices.length > 0 && (
              <button
                type="button"
                className="hyperdim-btn"
                onClick={onResetAll}
                data-testid="reset-all-slices-btn"
                style={{ fontSize: '0.6875rem', padding: '2px 6px' }}
                title="Reset all orthogonal dimensions to default"
              >
                Reset Slices
              </button>
            )}
            <button
              type="button"
              className="hyperdim-btn"
              onClick={onClose}
              data-testid="close-drawer-btn"
              style={{ fontSize: '0.6875rem', padding: '2px 6px' }}
              title="Close orthogonal parameter panel"
            >
              ✕
            </button>
          </div>
        </div>

        {orthogonalIndices.length === 0 ? (
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', padding: '12px 0' }} data-testid="no-orthogonal-msg">
            2D function: all dimensions active on the slicing plane.
          </div>
        ) : (
          <div className="hyperdim-param-list" data-testid="orthogonal-sliders-list">
            {orthogonalIndices.map((k) => {
              const dim = fn.dimensions[k];
              const currentVal = sliceCoords[k] ?? dim.defaultValue;
              const step = (dim.max - dim.min) / 100.0 || 0.01;

              return (
                <div key={dim.id} className="hyperdim-param-item" data-testid={`orthogonal-item-${dim.id}`}>
                  <div className="hyperdim-param-item-header">
                    <span className="hyperdim-param-name">{dim.name}</span>
                    <span className="hyperdim-param-val" data-testid={`orthogonal-val-${dim.id}`}>
                      {currentVal.toFixed(2)}
                    </span>
                  </div>
                  <div className="hyperdim-param-slider-row">
                    <input
                      type="range"
                      min={dim.min}
                      max={dim.max}
                      step={step}
                      value={currentVal}
                      onChange={(e) => onCoordinateChange(k, parseFloat(e.target.value))}
                      data-testid={`orthogonal-slider-${dim.id}`}
                    />
                    <button
                      type="button"
                      className="hyperdim-btn"
                      onClick={() => onResetCoordinate(k)}
                      data-testid={`reset-orthogonal-${dim.id}`}
                      style={{ fontSize: '0.625rem', padding: '2px 4px' }}
                      title={`Reset to default (${dim.defaultValue})`}
                    >
                      ↺
                    </button>
                  </div>
                  <div className="hyperdim-param-range-bounds">
                    <span>min: {dim.min}</span>
                    <span>def: {dim.defaultValue}</span>
                    <span>max: {dim.max}</span>
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

/**
 * Generalized Hyperdimensional Function Plotter Widget.
 *
 * Visualizes arbitrary functions f: R^n -> R^m across 2D contour slice heatmaps
 * and 3D heightmap meshes with live orthogonal coordinate scrubbing.
 */
export const HyperdimPlotterWidget: React.FC<DomainWidgetProps<ASTNode | null>> = ({
  ast,
  declarationName = 'R^n -> R^m',
}) => {
  // Pre-configured and AST-derived functions
  const astDerivedFunction = useMemo(() => {
    return ast ? createFunctionFromAST(ast, declarationName) : null;
  }, [ast, declarationName]);

  const availableFunctions = useMemo(() => {
    return astDerivedFunction
      ? [...BENCHMARK_FUNCTIONS, astDerivedFunction]
      : BENCHMARK_FUNCTIONS;
  }, [astDerivedFunction]);

  // Viewport Control State
  const [selectedFunctionId, setSelectedFunctionId] = useState<string>(
    availableFunctions[0].id
  );
  const [axisX, setAxisX] = useState<number>(0);
  const [axisY, setAxisY] = useState<number>(1);
  const [displayMode, setDisplayMode] = useState<HyperdimDisplayMode>('2d');
  const [colormap, setColormap] = useState<ColorMapType>('viridis');
  const [resolution, setResolution] = useState<number>(100);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [showDrawer, setShowDrawer] = useState<boolean>(true);

  // Active function resolution
  const activeFunction: HyperdimFunction = useMemo(() => {
    const found = availableFunctions.find((f) => f.id === selectedFunctionId);
    return found || availableFunctions[0];
  }, [availableFunctions, selectedFunctionId]);

  // Coordinate buffer for slicing (size n)
  const [sliceCoords, setSliceCoords] = useState<Float32Array>(() => {
    const coords = new Float32Array(activeFunction.inputDim);
    for (let i = 0; i < activeFunction.inputDim; i++) {
      coords[i] = activeFunction.dimensions[i]?.defaultValue ?? 0.0;
    }
    return coords;
  });

  // Re-initialize sliceCoords whenever active function changes
  useEffect(() => {
    const coords = new Float32Array(activeFunction.inputDim);
    for (let i = 0; i < activeFunction.inputDim; i++) {
      coords[i] = activeFunction.dimensions[i]?.defaultValue ?? 0.0;
    }
    setSliceCoords(coords);

    // Keep axis indices valid
    setAxisX(0);
    setAxisY(Math.min(1, activeFunction.inputDim - 1));
  }, [activeFunction]);

  // Environment checks
  const [hasWebGL] = useState<boolean>(() => isWebGLAvailable());
  const [hasCanvas] = useState<boolean>(() => isCanvasAvailable());

  // Fast zero-allocation evaluation buffer reference
  const outBufferRef = useRef<Float32Array>(new Float32Array(150 * 150));

  // Evaluated grid state
  const evaluationResult: GridEvaluationResult = useMemo(() => {
    const safeAxisX = Math.min(axisX, activeFunction.inputDim - 1);
    const safeAxisY = axisX === axisY ? (safeAxisX === 0 ? 1 : 0) : Math.min(axisY, activeFunction.inputDim - 1);

    return evaluate2DSlice(
      activeFunction,
      safeAxisX,
      safeAxisY,
      sliceCoords,
      resolution,
      resolution,
      outBufferRef.current
    );
  }, [activeFunction, axisX, axisY, sliceCoords, resolution]);

  // 2D Canvas rendering ref
  const canvas2DRef = useRef<HTMLCanvasElement>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    val: number;
  } | null>(null);

  // 3D Three.js rendering refs
  const threeWrapperRef = useRef<HTMLDivElement>(null);
  const cameraDistanceRef = useRef<number>(9.0);
  const cameraThetaRef = useRef<number>(0.7);
  const cameraPhiRef = useRef<number>(0.6);
  const cameraTargetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));

  // Update a single orthogonal coordinate
  const handleCoordinateChange = useCallback((dimIndex: number, val: number) => {
    setSliceCoords((prev) => {
      const next = new Float32Array(prev);
      next[dimIndex] = val;
      return next;
    });
  }, []);

  // Reset a single orthogonal coordinate
  const handleResetCoordinate = useCallback((dimIndex: number) => {
    setSliceCoords((prev) => {
      const next = new Float32Array(prev);
      next[dimIndex] = activeFunction.dimensions[dimIndex]?.defaultValue ?? 0.0;
      return next;
    });
  }, [activeFunction]);

  // Reset all coordinates
  const handleResetAllCoordinates = useCallback(() => {
    const next = new Float32Array(activeFunction.inputDim);
    for (let i = 0; i < activeFunction.inputDim; i++) {
      next[i] = activeFunction.dimensions[i]?.defaultValue ?? 0.0;
    }
    setSliceCoords(next);
  }, [activeFunction]);

  // Reset camera view
  const handleResetView = useCallback(() => {
    cameraDistanceRef.current = 9.0;
    cameraThetaRef.current = 0.7;
    cameraPhiRef.current = 0.6;
    cameraTargetRef.current.set(0, 0, 0);
    handleResetAllCoordinates();
  }, [handleResetAllCoordinates]);

  // Handle axis selections
  const handleAxisXChange = (newX: number) => {
    if (newX === axisY) {
      setAxisY(axisX);
    }
    setAxisX(newX);
  };

  const handleAxisYChange = (newY: number) => {
    if (newY === axisX) {
      setAxisX(axisY);
    }
    setAxisY(newY);
  };

  // ==========================================
  // 2D Heatmap & Contour Rendering Effect
  // ==========================================
  useEffect(() => {
    if (!hasCanvas || displayMode === '3d') return;
    const canvas = canvas2DRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.clientWidth || 600;
    const height = canvas.clientHeight || 500;
    canvas.width = width;
    canvas.height = height;

    const { resolutionX: resX, resolutionY: resY, values, minVal, maxVal } = evaluationResult;
    const range = maxVal - minVal || 1.0;

    // 1. Offscreen canvas for fast colormap pixel transfer
    const offCanvas = document.createElement('canvas');
    offCanvas.width = resX;
    offCanvas.height = resY;
    const offCtx = offCanvas.getContext('2d');
    if (!offCtx) return;

    const imgData = offCtx.createImageData ? offCtx.createImageData(resX, resY) : null;
    if (!imgData || !imgData.data) return;
    const data = imgData.data;

    for (let j = 0; j < resY; j++) {
      // Invert Y so bottom is Cartesian minimum
      const rowInvert = resY - 1 - j;
      const srcOffset = rowInvert * resX;
      const dstOffset = j * resX * 4;

      for (let i = 0; i < resX; i++) {
        const val = values[srcOffset + i];
        const t = (val - minVal) / range;
        const [r, g, b] = sampleColorMap255(t, colormap);
        const idx = dstOffset + i * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    // 2. Draw scaled heatmap onto main canvas
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offCanvas, 0, 0, width, height);

    // 3. Draw Contour Isolines via Marching Squares
    const numContours = 8;
    ctx.lineWidth = 1.0;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.beginPath();

    for (let c = 1; c <= numContours; c++) {
      const threshold = minVal + (c / (numContours + 1)) * range;

      for (let j = 0; j < resY - 1; j++) {
        const y0 = height - (j / (resY - 1)) * height;
        const y1 = height - ((j + 1) / (resY - 1)) * height;

        for (let i = 0; i < resX - 1; i++) {
          const x0 = (i / (resX - 1)) * width;
          const x1 = ((i + 1) / (resX - 1)) * width;

          // Corner values
          const vTL = values[(j + 1) * resX + i];
          const vTR = values[(j + 1) * resX + (i + 1)];
          const vBR = values[j * resX + (i + 1)];
          const vBL = values[j * resX + i];

          let cellIndex = 0;
          if (vTL >= threshold) cellIndex |= 8;
          if (vTR >= threshold) cellIndex |= 4;
          if (vBR >= threshold) cellIndex |= 2;
          if (vBL >= threshold) cellIndex |= 1;

          if (cellIndex === 0 || cellIndex === 15) continue;

          // Edge crossing interpolations
          const topX = x0 + ((threshold - vTL) / (vTR - vTL || 1e-6)) * (x1 - x0);
          const bottomX = x0 + ((threshold - vBL) / (vBR - vBL || 1e-6)) * (x1 - x0);
          const leftY = y1 + ((threshold - vTL) / (vBL - vTL || 1e-6)) * (y0 - y1);
          const rightY = y1 + ((threshold - vTR) / (vBR - vTR || 1e-6)) * (y0 - y1);

          switch (cellIndex) {
            case 1:
            case 14:
              ctx.moveTo(x0, leftY);
              ctx.lineTo(bottomX, y0);
              break;
            case 2:
            case 13:
              ctx.moveTo(bottomX, y0);
              ctx.lineTo(x1, rightY);
              break;
            case 3:
            case 12:
              ctx.moveTo(x0, leftY);
              ctx.lineTo(x1, rightY);
              break;
            case 4:
            case 11:
              ctx.moveTo(topX, y1);
              ctx.lineTo(x1, rightY);
              break;
            case 5:
              ctx.moveTo(x0, leftY);
              ctx.lineTo(topX, y1);
              ctx.moveTo(bottomX, y0);
              ctx.lineTo(x1, rightY);
              break;
            case 6:
            case 9:
              ctx.moveTo(topX, y1);
              ctx.lineTo(bottomX, y0);
              break;
            case 7:
            case 8:
              ctx.moveTo(x0, leftY);
              ctx.lineTo(topX, y1);
              break;
            case 10:
              ctx.moveTo(topX, y1);
              ctx.lineTo(x1, rightY);
              ctx.moveTo(x0, leftY);
              ctx.lineTo(bottomX, y0);
              break;
          }
        }
      }
    }
    ctx.stroke();

    // 4. Draw Crosshair coordinates if hovering
    if (hoverInfo) {
      const dimX = activeFunction.dimensions[evaluationResult.axisX];
      const dimY = activeFunction.dimensions[evaluationResult.axisY];

      const u = (hoverInfo.x - dimX.min) / (dimX.max - dimX.min || 1.0);
      const v = (hoverInfo.y - dimY.min) / (dimY.max - dimY.min || 1.0);

      const px = u * width;
      const py = height - v * height;

      ctx.save();
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
      ctx.lineWidth = 1.0;
      ctx.setLineDash([4, 4]);

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(width, py);
      ctx.stroke();

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, height);
      ctx.stroke();

      // Intersection dot
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.restore();
    }
  }, [
    hasCanvas,
    displayMode,
    evaluationResult,
    colormap,
    hoverInfo,
    activeFunction,
  ]);

  // Mouse move handler for 2D crosshair coordinate readout
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvas2DRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const w = rect.width;
    const h = rect.height;

    const dimX = activeFunction.dimensions[evaluationResult.axisX];
    const dimY = activeFunction.dimensions[evaluationResult.axisY];

    const u = Math.max(0.0, Math.min(1.0, px / w));
    const v = Math.max(0.0, Math.min(1.0, 1.0 - py / h));

    const xVal = dimX.min + u * (dimX.max - dimX.min);
    const yVal = dimY.min + v * (dimY.max - dimY.min);

    // Reconstruct coordinate query vector
    const queryCoords = new Float32Array(activeFunction.inputDim);
    for (let k = 0; k < activeFunction.inputDim; k++) {
      queryCoords[k] = sliceCoords[k] ?? activeFunction.dimensions[k]?.defaultValue ?? 0.0;
    }
    queryCoords[evaluationResult.axisX] = xVal;
    queryCoords[evaluationResult.axisY] = yVal;

    const evalRes = activeFunction.evaluate(queryCoords);
    const fnVal = typeof evalRes === 'number' ? evalRes : evalRes[0] ?? 0.0;

    setHoverInfo({ x: xVal, y: yVal, val: fnVal });
  };

  const handleCanvasMouseLeave = () => {
    setHoverInfo(null);
  };

  // ==========================================
  // 3D Heightmap Three.js Rendering Effect
  // ==========================================
  useEffect(() => {
    if (!hasWebGL || displayMode === '2d') return;
    const container = threeWrapperRef.current;
    if (!container) return;

    let renderer: THREE.WebGLRenderer | null = null;
    let scene: THREE.Scene | null = null;
    let camera: THREE.PerspectiveCamera | null = null;
    let mesh: THREE.Mesh | null = null;
    let geometry: THREE.PlaneGeometry | null = null;
    let material: THREE.MeshStandardMaterial | null = null;
    let animId: number | null = null;

    try {
      const width = container.clientWidth || 600;
      const height = container.clientHeight || 500;

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);

      // Lighting
      const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambLight);

      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(6, 12, 8);
      scene.add(dirLight);

      // Terrain Plane Geometry
      const segX = resolution - 1;
      const segY = resolution - 1;
      geometry = new THREE.PlaneGeometry(6, 6, segX, segY);
      geometry.rotateX(-Math.PI / 2); // Horizontal orientation

      // Apply Elevation Displacements & Vertex Colors
      const posAttr = geometry.attributes.position;
      const vertexCount = posAttr.count;
      const colors = new Float32Array(vertexCount * 3);

      const { values, minVal, maxVal } = evaluationResult;
      const range = maxVal - minVal || 1.0;

      for (let j = 0; j < resolution; j++) {
        for (let i = 0; i < resolution; i++) {
          const gridIdx = j * resolution + i;
          const val = values[gridIdx];
          const t = (val - minVal) / range;
          const heightY = (t - 0.5) * 2.2;

          // Vertex index in PlaneGeometry
          const vIdx = j * resolution + i;
          posAttr.setY(vIdx, heightY);

          const [r, g, b] = sampleColorMap(t, colormap);
          colors[vIdx * 3] = r;
          colors[vIdx * 3 + 1] = g;
          colors[vIdx * 3 + 2] = b;
        }
      }

      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.computeVertexNormals();

      material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        wireframe,
        roughness: 0.45,
        metalness: 0.1,
        side: THREE.DoubleSide,
      });

      mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
      container.appendChild(renderer.domElement);

      // Orbit Control Events
      let isDragging = false;
      let dragButton = 0;
      let lastMouseX = 0;
      let lastMouseY = 0;

      const onMouseDown = (e: MouseEvent) => {
        isDragging = true;
        dragButton = e.button;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      };

      const onMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;

        const isPan = dragButton === 2 || e.shiftKey;
        if (isPan) {
          const dist = cameraDistanceRef.current;
          const theta = cameraThetaRef.current;
          const phi = cameraPhiRef.current;
          const pos = new THREE.Vector3(
            dist * Math.cos(phi) * Math.sin(theta),
            dist * Math.sin(phi),
            dist * Math.cos(phi) * Math.cos(theta)
          );
          const forward = new THREE.Vector3().subVectors(cameraTargetRef.current, pos).normalize();
          let right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
          if (right.lengthSq() < 0.0001) right = new THREE.Vector3(1, 0, 0);
          const up = new THREE.Vector3().crossVectors(right, forward).normalize();

          const panSpeed = dist * 0.002;
          cameraTargetRef.current.addScaledVector(right, -dx * panSpeed);
          cameraTargetRef.current.addScaledVector(up, dy * panSpeed);
        } else {
          cameraThetaRef.current += dx * 0.008;
          cameraPhiRef.current = Math.max(
            0.05,
            Math.min(Math.PI * 0.48, cameraPhiRef.current + dy * 0.008)
          );
        }
      };

      const onMouseUp = () => {
        isDragging = false;
      };

      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const factor = Math.exp(e.deltaY * 0.001);
        cameraDistanceRef.current = Math.max(2.0, Math.min(30.0, cameraDistanceRef.current * factor));
      };

      const onContextMenu = (e: MouseEvent) => e.preventDefault();

      const dom = renderer.domElement;
      dom.addEventListener('mousedown', onMouseDown);
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
      dom.addEventListener('wheel', onWheel, { passive: false });
      dom.addEventListener('contextmenu', onContextMenu);

      // Render Loop
      const renderFrame = () => {
        animId = requestAnimationFrame(renderFrame);
        if (!renderer || !scene || !camera) return;

        const dist = cameraDistanceRef.current;
        const theta = cameraThetaRef.current;
        const phi = cameraPhiRef.current;
        const target = cameraTargetRef.current;

        camera.position.set(
          target.x + dist * Math.cos(phi) * Math.sin(theta),
          target.y + dist * Math.sin(phi),
          target.z + dist * Math.cos(phi) * Math.cos(theta)
        );
        camera.lookAt(target);

        renderer.render(scene, camera);
      };
      renderFrame();

      return () => {
        if (animId !== null) cancelAnimationFrame(animId);
        dom.removeEventListener('mousedown', onMouseDown);
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
        dom.removeEventListener('wheel', onWheel);
        dom.removeEventListener('contextmenu', onContextMenu);

        if (mesh && scene) scene.remove(mesh);
        if (geometry) geometry.dispose();
        if (material) material.dispose();
        if (renderer) {
          renderer.dispose();
          if (renderer.domElement && renderer.domElement.parentNode) {
            renderer.domElement.parentNode.removeChild(renderer.domElement);
          }
        }
      };
    } catch (err) {
      console.error('[HyperdimPlotter] Three.js Heightmap initialization failed with context:', {
        error: err,
        resolution,
        functionId: activeFunction.id,
      });
    }
  }, [
    hasWebGL,
    displayMode,
    evaluationResult,
    colormap,
    wireframe,
    resolution,
    activeFunction.id,
  ]);

  const activeDimX = activeFunction.dimensions[evaluationResult.axisX];
  const activeDimY = activeFunction.dimensions[evaluationResult.axisY];

  return (
    <div className="hyperdim-container" data-testid="hyperdim-plotter-widget">
      {/* Interactive Viewport Control Toolbar */}
      <div className="hyperdim-toolbar" data-testid="hyperdim-toolbar">
        {/* Function Selector */}
        <div className="hyperdim-toolbar-group">
          <span className="hyperdim-label">Function:</span>
          <select
            className="hyperdim-select"
            value={selectedFunctionId}
            onChange={(e) => setSelectedFunctionId(e.target.value)}
            data-testid="function-select"
          >
            {availableFunctions.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name} ({f.inputDim}D)
              </option>
            ))}
          </select>
        </div>

        {/* Axis Pair Pickers */}
        <div className="hyperdim-toolbar-group">
          <span className="hyperdim-label">X-Axis:</span>
          <select
            className="hyperdim-select"
            value={axisX}
            onChange={(e) => handleAxisXChange(parseInt(e.target.value, 10))}
            data-testid="axis-x-select"
          >
            {activeFunction.dimensions.map((d, idx) => (
              <option key={d.id} value={idx}>
                {d.name}
              </option>
            ))}
          </select>

          <span className="hyperdim-label">Y-Axis:</span>
          <select
            className="hyperdim-select"
            value={axisY}
            onChange={(e) => handleAxisYChange(parseInt(e.target.value, 10))}
            data-testid="axis-y-select"
          >
            {activeFunction.dimensions.map((d, idx) => (
              <option key={d.id} value={idx}>
                {d.name}
              </option>
            ))}
          </select>
        </div>

        {/* Display Mode Switcher */}
        <div className="hyperdim-toolbar-group">
          <span className="hyperdim-label">Mode:</span>
          <button
            type="button"
            className={`hyperdim-btn ${displayMode === '2d' ? 'active' : ''}`}
            onClick={() => setDisplayMode('2d')}
            data-testid="mode-2d-btn"
          >
            2D Heatmap
          </button>
          <button
            type="button"
            className={`hyperdim-btn ${displayMode === '3d' ? 'active' : ''}`}
            onClick={() => setDisplayMode('3d')}
            data-testid="mode-3d-btn"
          >
            3D Mesh
          </button>
          <button
            type="button"
            className={`hyperdim-btn ${displayMode === 'split' ? 'active' : ''}`}
            onClick={() => setDisplayMode('split')}
            data-testid="mode-split-btn"
          >
            Split View
          </button>
        </div>

        {/* Colormap Dropdown */}
        <div className="hyperdim-toolbar-group">
          <span className="hyperdim-label">Colormap:</span>
          <select
            className="hyperdim-select"
            value={colormap}
            onChange={(e) => setColormap(e.target.value as ColorMapType)}
            data-testid="colormap-select"
          >
            <option value="viridis">Viridis</option>
            <option value="turbo">Turbo</option>
            <option value="magma">Magma</option>
            <option value="coolwarm">CoolWarm</option>
          </select>
        </div>

        {/* Resolution Buttons */}
        <div className="hyperdim-toolbar-group">
          <span className="hyperdim-label">Grid:</span>
          <button
            type="button"
            className={`hyperdim-btn ${resolution === 50 ? 'active' : ''}`}
            onClick={() => setResolution(50)}
            data-testid="res-50-btn"
          >
            50²
          </button>
          <button
            type="button"
            className={`hyperdim-btn ${resolution === 100 ? 'active' : ''}`}
            onClick={() => setResolution(100)}
            data-testid="res-100-btn"
          >
            100²
          </button>
          <button
            type="button"
            className={`hyperdim-btn ${resolution === 150 ? 'active' : ''}`}
            onClick={() => setResolution(150)}
            data-testid="res-150-btn"
          >
            150²
          </button>
        </div>

        {/* Utility Buttons: Wireframe, Drawer Toggle, Reset */}
        <div className="hyperdim-toolbar-group">
          {displayMode !== '2d' && (
            <button
              type="button"
              className={`hyperdim-btn ${wireframe ? 'active' : ''}`}
              onClick={() => setWireframe((prev) => !prev)}
              data-testid="toggle-wireframe-btn"
              title="Toggle Wireframe Shading"
            >
              Wireframe
            </button>
          )}
          <button
            type="button"
            className={`hyperdim-btn ${showDrawer ? 'active' : ''}`}
            onClick={() => setShowDrawer((prev) => !prev)}
            data-testid="toggle-drawer-btn"
            title="Toggle Orthogonal Slices Panel"
          >
            Slices ({activeFunction.inputDim - 2 > 0 ? activeFunction.inputDim - 2 : 0})
          </button>
          <button
            type="button"
            className="hyperdim-btn"
            onClick={handleResetView}
            data-testid="reset-view-btn"
            title="Reset View and Coordinates"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Floating Orthogonal Dimension Scrubber Panel */}
      {showDrawer && (
        <OrthogonalScrubberOverlay
          fn={activeFunction}
          axisX={evaluationResult.axisX}
          axisY={evaluationResult.axisY}
          sliceCoords={sliceCoords}
          onCoordinateChange={handleCoordinateChange}
          onResetCoordinate={handleResetCoordinate}
          onResetAll={handleResetAllCoordinates}
          onClose={() => setShowDrawer(false)}
        />
      )}

      {/* Main Visual Viewport Area */}
      {hasCanvas || hasWebGL ? (
        <div className="hyperdim-view-wrapper" data-testid="hyperdim-view-wrapper">
          {displayMode === 'split' ? (
            <div className="hyperdim-split-view" data-testid="hyperdim-split-view">
              <div className="hyperdim-split-pane" data-testid="split-pane-2d">
                <canvas
                  ref={canvas2DRef}
                  className="hyperdim-canvas-2d"
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                  data-testid="canvas-2d"
                />
              </div>
              <div className="hyperdim-split-pane" data-testid="split-pane-3d">
                <div
                  ref={threeWrapperRef}
                  className="hyperdim-canvas-3d-wrapper"
                  data-testid="canvas-3d-wrapper"
                />
              </div>
            </div>
          ) : displayMode === '2d' ? (
            <div className="hyperdim-view-single" data-testid="single-view-2d">
              <canvas
                ref={canvas2DRef}
                className="hyperdim-canvas-2d"
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
                data-testid="canvas-2d"
              />
            </div>
          ) : (
            <div className="hyperdim-view-single" data-testid="single-view-3d">
              <div
                ref={threeWrapperRef}
                className="hyperdim-canvas-3d-wrapper"
                data-testid="canvas-3d-wrapper"
              />
            </div>
          )}

          {/* Coordinate Readout Badge */}
          <div className="hyperdim-readout-badge" data-testid="hyperdim-readout-badge">
            <span>
              {activeDimX.name}: {hoverInfo ? hoverInfo.x.toFixed(3) : `${activeDimX.min}..${activeDimX.max}`}
            </span>
            <span>
              {activeDimY.name}: {hoverInfo ? hoverInfo.y.toFixed(3) : `${activeDimY.min}..${activeDimY.max}`}
            </span>
            <span>
              f(x): {hoverInfo ? hoverInfo.val.toFixed(4) : `[${evaluationResult.minVal.toFixed(2)}..${evaluationResult.maxVal.toFixed(2)}]`}
            </span>
          </div>

          {/* Colormap Legend */}
          <div className="hyperdim-legend-bar" data-testid="hyperdim-legend-bar">
            <span>{evaluationResult.minVal.toFixed(2)}</span>
            <div
              className="hyperdim-legend-gradient"
              style={{
                background:
                  colormap === 'viridis'
                    ? 'linear-gradient(to right, #440154, #21918c, #fde725)'
                    : colormap === 'turbo'
                    ? 'linear-gradient(to right, #30123b, #1ae4b6, #e1dc32, #7a0403)'
                    : colormap === 'magma'
                    ? 'linear-gradient(to right, #000004, #721f81, #f1605d, #fcfdbf)'
                    : 'linear-gradient(to right, #3a53a4, #ddd, #b2182b)',
              }}
            />
            <span>{evaluationResult.maxVal.toFixed(2)}</span>
          </div>
        </div>
      ) : (
        /* Headless Fallback Card for JSDOM and Non-Canvas Environments */
        <div className="widget-stage" data-testid="hyperdim-plotter-fallback">
          <div className="stage-card">
            <div className="stage-icon" aria-hidden="true">📈</div>
            <h2 className="stage-title">Generalized Hyperdim Plotter</h2>
            <p className="stage-description">
              Multi-dimensional function evaluator for <code>{activeFunction.name}</code> (f: R^{activeFunction.inputDim} → R).
            </p>
            <div className="domain-tags">
              <span className="domain-tag">2D Contour Slice</span>
              <span className="domain-tag">3D Heightmap</span>
              <span className="domain-tag">{resolution}x{resolution} Grid</span>
              <span className="domain-tag">{colormap.toUpperCase()}</span>
            </div>
            <div style={{ marginTop: '16px', width: '100%', textAlign: 'left' }}>
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Active Slicing Plane:
              </span>
              <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
                <span className="domain-tag" style={{ color: 'var(--accent-cyan)' }}>
                  Horizontal: {activeDimX.name} [{activeDimX.min}..{activeDimX.max}]
                </span>
                <span className="domain-tag" style={{ color: 'var(--accent-indigo)' }}>
                  Vertical: {activeDimY.name} [{activeDimY.min}..{activeDimY.max}]
                </span>
              </div>
            </div>
            {activeFunction.inputDim > 2 && (
              <div style={{ marginTop: '12px', width: '100%', textAlign: 'left' }}>
                <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Orthogonal Dimensions ({activeFunction.inputDim - 2}):
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                  {activeFunction.dimensions
                    .filter((_, idx) => idx !== evaluationResult.axisX && idx !== evaluationResult.axisY)
                    .map((dim) => (
                      <div
                        key={dim.id}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          fontSize: '0.75rem',
                          background: 'var(--bg-surface)',
                          padding: '4px 8px',
                          borderRadius: '4px',
                        }}
                      >
                        <span style={{ color: 'var(--text-primary)' }}>{dim.name}</span>
                        <span style={{ color: 'var(--accent-cyan)' }}>
                          val: {(sliceCoords[activeFunction.dimensions.findIndex((d) => d.id === dim.id)] ?? dim.defaultValue).toFixed(2)}
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
