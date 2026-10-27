import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import type { DomainWidgetProps } from '../types';
import type { ASTNode, SDFNode, UniversalSDFNode } from '../../../core/ast/types';
import { isSDFNode, collectFreeParams, createSphere, createFreeParam } from '../../../core/ast/models';
import { useFreeParamStore } from '../../../core/store/freeParamStore';
import {
  generateRaymarchingFragmentShader,
  SCREEN_QUAD_VERTEX_SHADER,
  createRaymarchingUniforms,
  getParamUniformName,
} from './glslCodegen';

/**
 * Checks whether WebGL rendering context is supported in the current runtime environment.
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
    console.error('[isWebGLAvailable] Environment WebGL feature check failed with context:', err);
    return false;
  }
}

/**
 * Interactive 3D WebGL Raymarching Widget for SDFNode.
 *
 * Transpiles the active SDFNode AST into a continuous distance-field fragment shader,
 * mounted on a screen-aligned quad in Three.js. Supports orbit rotation, scroll zooming,
 * right-click panning, real-time FreeParam uniform modulation, and cross-sectional slicing.
 */
export const SDFRaymarchWidget: React.FC<DomainWidgetProps<ASTNode>> = ({
  ast,
  declarationName = 'SDFNode',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

  // Slicing state: 0 = None, 1 = XY, 2 = YZ, 3 = ZX
  const [slicePlane, setSlicePlane] = useState<number>(0);
  const [sliceOffset, setSliceOffset] = useState<number>(0.0);
  const slicePlaneRef = useRef<number>(0);
  const sliceOffsetRef = useRef<number>(0.0);

  // Synchronize ref values for the 60 FPS animation loop
  useEffect(() => {
    slicePlaneRef.current = slicePlane;
  }, [slicePlane]);

  useEffect(() => {
    sliceOffsetRef.current = sliceOffset;
  }, [sliceOffset]);

  // WebGL availability state
  const [hasWebGL, setHasWebGL] = useState<boolean>(() => isWebGLAvailable());

  // Resolve concrete SDFNode from ast prop
  const resolvedNode: SDFNode = useMemo(() => {
    if (!ast) {
      return createSphere(createFreeParam('radius', 0.1, 5.0, 1.0));
    }
    if (ast.kind === 'universal_sdf') {
      return (ast as UniversalSDFNode).node;
    }
    if (isSDFNode(ast)) {
      return ast;
    }
    return createSphere(createFreeParam('radius', 0.1, 5.0, 1.0));
  }, [ast]);

  // Extract all continuous parameters
  const params = useMemo(() => collectFreeParams(resolvedNode), [resolvedNode]);

  // Virtual Orbit Camera state refs
  const cameraDistanceRef = useRef<number>(4.0);
  const cameraThetaRef = useRef<number>(0.8); // azimuth
  const cameraPhiRef = useRef<number>(0.4); // elevation
  const cameraTargetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));

  // Reset camera view callback
  const handleResetView = useCallback(() => {
    cameraDistanceRef.current = 4.0;
    cameraThetaRef.current = 0.8;
    cameraPhiRef.current = 0.4;
    cameraTargetRef.current.set(0, 0, 0);
  }, []);

  // WebGL Lifecycle Effect
  useEffect(() => {
    if (!hasWebGL) return;
    const canvasContainer = canvasWrapperRef.current;
    if (!canvasContainer) return;

    let renderer: THREE.WebGLRenderer | null = null;
    let scene: THREE.Scene | null = null;
    let camera: THREE.OrthographicCamera | null = null;
    let material: THREE.ShaderMaterial | null = null;
    let geometry: THREE.PlaneGeometry | null = null;
    let mesh: THREE.Mesh | null = null;
    let animId: number | null = null;
    const startTime = performance.now();

    try {
      const width = canvasContainer.clientWidth || 800;
      const height = canvasContainer.clientHeight || 600;

      scene = new THREE.Scene();
      camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      geometry = new THREE.PlaneGeometry(2, 2);

      const fragmentShader = generateRaymarchingFragmentShader(resolvedNode, params);
      const initialValues = useFreeParamStore.getState().values;
      const uniforms = createRaymarchingUniforms(resolvedNode, params, initialValues);
      uniforms.u_resolution.value.set(width, height);

      material = new THREE.ShaderMaterial({
        vertexShader: SCREEN_QUAD_VERTEX_SHADER,
        fragmentShader,
        uniforms,
        depthTest: false,
        depthWrite: false,
      });

      mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      // Clear any stale canvas
      while (canvasContainer.firstChild) {
        canvasContainer.removeChild(canvasContainer.firstChild);
      }
      canvasContainer.appendChild(renderer.domElement);

      // Mouse Orbit Controls
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
          // Pan camera target
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
          // Orbit camera angles
          cameraThetaRef.current += dx * 0.008;
          cameraPhiRef.current = Math.max(
            -Math.PI * 0.48,
            Math.min(Math.PI * 0.48, cameraPhiRef.current + dy * 0.008)
          );
        }
      };

      const onMouseUp = () => {
        isDragging = false;
      };

      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const zoomFactor = Math.exp(e.deltaY * 0.001);
        cameraDistanceRef.current = Math.max(0.5, Math.min(30.0, cameraDistanceRef.current * zoomFactor));
      };

      const onContextMenu = (e: MouseEvent) => {
        e.preventDefault();
      };

      const dom = renderer.domElement;
      dom.addEventListener('mousedown', onMouseDown);
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
      dom.addEventListener('wheel', onWheel, { passive: false });
      dom.addEventListener('contextmenu', onContextMenu);

      // Resize observer
      const onResize = () => {
        if (!canvasContainer || !renderer || !material) return;
        const w = canvasContainer.clientWidth || 800;
        const h = canvasContainer.clientHeight || 600;
        renderer.setSize(w, h);
        material.uniforms.u_resolution.value.set(w, h);
      };
      const resizeObserver = new ResizeObserver(onResize);
      resizeObserver.observe(canvasContainer);

      // Render Loop
      const renderFrame = () => {
        animId = requestAnimationFrame(renderFrame);

        if (!renderer || !scene || !camera || !material) return;

        const dist = cameraDistanceRef.current;
        const theta = cameraThetaRef.current;
        const phi = cameraPhiRef.current;
        const target = cameraTargetRef.current;

        const posX = target.x + dist * Math.cos(phi) * Math.sin(theta);
        const posY = target.y + dist * Math.sin(phi);
        const posZ = target.z + dist * Math.cos(phi) * Math.cos(theta);
        const cameraPos = new THREE.Vector3(posX, posY, posZ);

        const forward = new THREE.Vector3().subVectors(target, cameraPos).normalize();
        let right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
        if (right.lengthSq() < 0.0001) right = new THREE.Vector3(1, 0, 0);
        const up = new THREE.Vector3().crossVectors(right, forward).normalize();

        const camMatrix = new THREE.Matrix3().set(
          right.x, up.x, forward.x,
          right.y, up.y, forward.y,
          right.z, up.z, forward.z
        );

        const u = material.uniforms;
        u.u_time.value = (performance.now() - startTime) * 0.001;
        u.u_camera_pos.value.copy(cameraPos);
        u.u_camera_target.value.copy(target);
        u.u_camera_matrix.value.copy(camMatrix);

        u.u_slice_enabled.value = slicePlaneRef.current > 0 ? 1.0 : 0.0;
        u.u_slice_plane.value = slicePlaneRef.current;
        u.u_slice_offset.value = sliceOffsetRef.current;

        // Reactive parameter store integration
        const storeValues = useFreeParamStore.getState().values;
        for (const p of params) {
          const uName = getParamUniformName(p);
          if (u[uName]) {
            const val = storeValues[p.id] ?? p.defaultVal;
            u[uName].value = val;
          }
        }

        renderer.render(scene, camera);
      };

      renderFrame();

      // Clean up resources on unmount
      return () => {
        if (animId !== null) {
          cancelAnimationFrame(animId);
        }
        resizeObserver.disconnect();
        dom.removeEventListener('mousedown', onMouseDown);
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
        dom.removeEventListener('wheel', onWheel);
        dom.removeEventListener('contextmenu', onContextMenu);

        if (mesh && scene) {
          scene.remove(mesh);
        }
        if (geometry) {
          geometry.dispose();
        }
        if (material) {
          material.dispose();
        }
        if (renderer) {
          renderer.dispose();
          if (renderer.domElement && renderer.domElement.parentNode) {
            renderer.domElement.parentNode.removeChild(renderer.domElement);
          }
        }
      };
    } catch (err) {
      console.error('[SDFRaymarchWidget] WebGL initialization failed, switching to fallback.', err);
      setHasWebGL(false);
    }
  }, [hasWebGL, resolvedNode, params]);

  return (
    <div
      ref={containerRef}
      className="sdf-raymarch-container"
      data-testid="sdf-raymarch-widget"
    >
      {/* Slicing & Camera Toolbar */}
      <div className="sdf-toolbar" data-testid="sdf-slicing-toolbar">
        <div className="sdf-toolbar-group">
          <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Slicing:
          </span>
          <button
            type="button"
            className={`sdf-btn ${slicePlane === 0 ? 'active' : ''}`}
            onClick={() => setSlicePlane(0)}
            data-testid="slice-none-btn"
          >
            Slice: None
          </button>
          <button
            type="button"
            className={`sdf-btn ${slicePlane === 1 ? 'active' : ''}`}
            onClick={() => setSlicePlane(1)}
            data-testid="slice-xy-btn"
          >
            Slice: XY
          </button>
          <button
            type="button"
            className={`sdf-btn ${slicePlane === 2 ? 'active' : ''}`}
            onClick={() => setSlicePlane(2)}
            data-testid="slice-yz-btn"
          >
            Slice: YZ
          </button>
          <button
            type="button"
            className={`sdf-btn ${slicePlane === 3 ? 'active' : ''}`}
            onClick={() => setSlicePlane(3)}
            data-testid="slice-zx-btn"
          >
            Slice: ZX
          </button>
        </div>

        {slicePlane > 0 && (
          <div className="sdf-slider-container">
            <span>Offset: {sliceOffset.toFixed(2)}</span>
            <input
              type="range"
              className="sdf-slider"
              min="-5"
              max="5"
              step="0.05"
              value={sliceOffset}
              onChange={(e) => setSliceOffset(parseFloat(e.target.value))}
              data-testid="slice-offset-slider"
            />
          </div>
        )}

        <div className="sdf-toolbar-group">
          <button
            type="button"
            className="sdf-btn"
            onClick={handleResetView}
            data-testid="reset-view-btn"
          >
            Reset View
          </button>
        </div>
      </div>

      {/* Interactive WebGL Canvas Viewport */}
      {hasWebGL ? (
        <div
          ref={canvasWrapperRef}
          className="sdf-canvas-wrapper"
          data-testid="sdf-canvas-wrapper"
        />
      ) : (
        /* Fallback View for Headless / JSDOM / Unsupported Devices */
        <div className="widget-stage" data-testid="sdf-raymarch-fallback">
          <div className="stage-card">
            <div className="stage-icon" aria-hidden="true">🔮</div>
            <h2 className="stage-title">3D WebGL SDF Raymarching</h2>
            <p className="stage-description">
              Interactive 3D WebGL raymarching canvas with Phong illumination and real-time FreeParam uniform modulation for{' '}
              <code>{declarationName || 'SDFNode'}</code>.
            </p>
            <div className="domain-tags">
              <span className="domain-tag">Raymarching 60 FPS</span>
              <span className="domain-tag">Three.js Canvas</span>
              <span className="domain-tag">CSG Invariants</span>
            </div>
            {params.length > 0 && (
              <div style={{ marginTop: '16px', width: '100%', textAlign: 'left' }}>
                <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
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
                        [{p.minVal}..{p.maxVal}] (def: {p.defaultVal})
                      </span>
                    </div>
                  ))}
                  {params.length > 5 && (
                    <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                      +{params.length - 5} more in Inspector
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
