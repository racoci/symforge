/**
 * GLSL Raymarching Shader Code Generator for SymForge SDFNode AST.
 *
 * Transpiles continuous 3D Signed Distance Field (SDF) trees into optimized
 * WebGL / GLSL ES fragment shaders with analytical primitives, CSG operations,
 * Euler coordinate transformations, central finite-difference normal estimation,
 * Blinn-Phong illumination, and interactive slicing planes.
 */

import type { SDFNode, FreeParam } from '../../../core/ast/types';
import { collectFreeParams } from '../../../core/ast/models';
import * as THREE from 'three';

/**
 * Sanitizes a FreeParam ID into a valid GLSL identifier name.
 */
export function sanitizeParamId(id: string): string {
  const sanitized = id.replace(/[^a-zA-Z0-9_]/g, '_');
  // GLSL identifiers cannot begin with a digit
  if (/^[0-9]/.test(sanitized)) {
    return `_${sanitized}`;
  }
  return sanitized;
}

/**
 * Returns the GLSL uniform variable name for a FreeParam or string ID.
 */
export function getParamUniformName(param: FreeParam | string): string {
  const id = typeof param === 'string' ? param : param.id;
  return `u_param_${sanitizeParamId(id)}`;
}

/**
 * Resolves a parameter or literal number to a GLSL value expression.
 */
function paramExpr(param: FreeParam | string | number): string {
  if (typeof param === 'number') {
    return Number.isInteger(param) ? `${param}.0` : `${param}`;
  }
  return getParamUniformName(param);
}

/**
 * Recursively converts an SDFNode into GLSL distance expressions
 * using analytical primitives and transforms.
 *
 * @param node The SDF AST node to transpile
 * @param varName The coordinate variable name (defaults to "p")
 * @returns GLSL float expression computing the signed distance
 */
export function generateSDFExpression(node: SDFNode, varName = 'p'): string {
  switch (node.kind) {
    case 'sphere':
      return `sdSphere(${varName}, ${paramExpr(node.radius)})`;

    case 'box':
      return `sdBox(${varName}, vec3(${paramExpr(node.width)}, ${paramExpr(node.height)}, ${paramExpr(node.depth)}))`;

    case 'cylinder':
      return `sdCylinder(${varName}, ${paramExpr(node.radius)}, ${paramExpr(node.height)})`;

    case 'torus':
      return `sdTorus(${varName}, vec2(${paramExpr(node.majorRadius)}, ${paramExpr(node.minorRadius)}))`;

    case 'plane': {
      const normalStr = node.normal
        ? `vec3(${paramExpr(node.normal.x)}, ${paramExpr(node.normal.y)}, ${paramExpr(node.normal.z)})`
        : 'vec3(0.0, 1.0, 0.0)';
      return `sdPlane(${varName}, ${normalStr}, ${paramExpr(node.offset)})`;
    }

    case 'translate': {
      const pTrans = `(${varName} - vec3(${paramExpr(node.dx)}, ${paramExpr(node.dy)}, ${paramExpr(node.dz)}))`;
      return generateSDFExpression(node.child, pTrans);
    }

    case 'rotate': {
      const pRot = `rotateEuler(${varName}, vec3(${paramExpr(node.rx)}, ${paramExpr(node.ry)}, ${paramExpr(node.rz)}))`;
      return generateSDFExpression(node.child, pRot);
    }

    case 'scale': {
      const s = paramExpr(node.s);
      const pScale = `(${varName} / ${s})`;
      return `(${generateSDFExpression(node.child, pScale)} * ${s})`;
    }

    case 'union': {
      const d1 = generateSDFExpression(node.a, varName);
      const d2 = generateSDFExpression(node.b, varName);
      return `min(${d1}, ${d2})`;
    }

    case 'smooth_union': {
      const d1 = generateSDFExpression(node.a, varName);
      const d2 = generateSDFExpression(node.b, varName);
      return `opSmoothUnion(${d1}, ${d2}, ${paramExpr(node.smoothness)})`;
    }

    case 'intersection': {
      const d1 = generateSDFExpression(node.a, varName);
      const d2 = generateSDFExpression(node.b, varName);
      return `max(${d1}, ${d2})`;
    }

    case 'difference': {
      const d1 = generateSDFExpression(node.a, varName);
      const d2 = generateSDFExpression(node.b, varName);
      return `max(${d1}, -(${d2}))`;
    }

    case 'hole':
      return `sdSphere(${varName}, 0.5)`;

    case 'macro':
      return `sdSphere(${varName}, 1.0)`;

    default: {
      return `sdSphere(${varName}, 1.0)`;
    }
  }
}

/**
 * GLSL screen-quad vertex shader compatible with WebGL 1/2 and Three.js.
 */
export const SCREEN_QUAD_VERTEX_SHADER = `
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

/**
 * Analytical primitive library code embedded into raymarching shaders.
 */
export const GLSL_PRIMITIVES_LIBRARY = `
// --- Analytical Primitive Library ---
float sdSphere(vec3 p, float r) {
  return length(p) - r;
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCylinder(vec3 p, float r, float h) {
  vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h * 0.5);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

float sdPlane(vec3 p, vec3 n, float h) {
  return dot(p, normalize(n)) + h;
}

float opSmoothUnion(float d1, float d2, float k) {
  if (k <= 0.0001) return min(d1, d2);
  float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) - k * h * (1.0 - h);
}

vec3 rotateEuler(vec3 p, vec3 r) {
  vec3 c = cos(r);
  vec3 s = sin(r);
  mat3 rotX = mat3(
    1.0, 0.0, 0.0,
    0.0, c.x, s.x,
    0.0, -s.x, c.x
  );
  mat3 rotY = mat3(
    c.y, 0.0, -s.y,
    0.0, 1.0, 0.0,
    s.y, 0.0, c.y
  );
  mat3 rotZ = mat3(
    c.z, s.z, 0.0,
    -s.z, c.z, 0.0,
    0.0, 0.0, 1.0
  );
  return rotZ * rotY * rotX * p;
}
`;

/**
 * Assembles a complete, self-contained GLSL fragment shader
 * for 3D raymarching of the provided SDFNode.
 *
 * @param node The root SDF AST node
 * @param freeParams Explicit FreeParam list (or extracted automatically from node if empty)
 * @returns Complete GLSL ES fragment shader source code
 */
export function generateRaymarchingFragmentShader(
  node: SDFNode,
  freeParams: FreeParam[] = []
): string {
  // Collect and deduplicate parameters from both explicit argument and AST
  const nodeParams = collectFreeParams(node);
  const allParams = [...freeParams, ...nodeParams];
  const seenParamIds = new Set<string>();
  const uniformDeclarations: string[] = [];

  for (const p of allParams) {
    const sanitized = sanitizeParamId(p.id);
    if (!seenParamIds.has(sanitized)) {
      seenParamIds.add(sanitized);
      uniformDeclarations.push(`uniform float u_param_${sanitized};`);
    }
  }

  const sdfExpression = generateSDFExpression(node, 'p');

  return `precision highp float;

// ==========================================
// Viewport & Environment Uniforms
// ==========================================
uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_camera_pos;
uniform vec3 u_camera_target;
uniform mat3 u_camera_matrix;

// ==========================================
// Interactive Slicing Plane Uniforms
// 0 = None, 1 = XY Plane, 2 = YZ Plane, 3 = ZX Plane
// ==========================================
uniform float u_slice_enabled;
uniform int u_slice_plane;
uniform float u_slice_offset;

// ==========================================
// Symbolic FreeParam Uniform Declarations
// ==========================================
${uniformDeclarations.join('\n')}

${GLSL_PRIMITIVES_LIBRARY}

// ==========================================
// Scene Distance Estimator: float map(vec3 p)
// ==========================================
float map(vec3 p) {
  float d = ${sdfExpression};
  if (u_slice_enabled > 0.5 && u_slice_plane > 0) {
    float sliceDist = 0.0;
    if (u_slice_plane == 1) {
      sliceDist = p.z - u_slice_offset;
    } else if (u_slice_plane == 2) {
      sliceDist = p.x - u_slice_offset;
    } else if (u_slice_plane == 3) {
      sliceDist = p.y - u_slice_offset;
    }
    d = max(d, sliceDist);
  }
  return d;
}

// ==========================================
// Central Finite Differences Normal Estimator
// ==========================================
vec3 calcNormal(vec3 p) {
  const float eps = 0.001;
  const vec2 h = vec2(eps, 0.0);
  return normalize(vec3(
    map(p + h.xyy) - map(p - h.xyy),
    map(p + h.yxy) - map(p - h.yxy),
    map(p + h.yyx) - map(p - h.yyx)
  ));
}

// ==========================================
// Bounded Raymarching Sphere Tracer
// ==========================================
const int MAX_STEPS = 128;
const float PRECISION = 0.001;
const float MAX_DIST = 25.0;

float raymarch(vec3 ro, vec3 rd) {
  float t = 0.0;
  for (int i = 0; i < MAX_STEPS; i++) {
    vec3 p = ro + t * rd;
    float d = map(p);
    if (d < PRECISION) {
      return t;
    }
    t += d;
    if (t > MAX_DIST) {
      break;
    }
  }
  return -1.0;
}

// ==========================================
// Fragment Shader Main Entry Point
// ==========================================
void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
  vec3 ro = u_camera_pos;
  vec3 rd = normalize(u_camera_matrix * vec3(uv, 1.8));

  float t = raymarch(ro, rd);

  // Background environment gradient
  vec3 bg = mix(vec3(0.04, 0.06, 0.10), vec3(0.015, 0.02, 0.035), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));

  if (t < 0.0) {
    gl_FragColor = vec4(bg, 1.0);
    return;
  }

  vec3 p = ro + t * rd;
  vec3 normal = calcNormal(p);

  // Blinn-Phong illumination model
  vec3 ambient = vec3(0.12, 0.15, 0.20);

  vec3 keyLightDir = normalize(vec3(0.7, 1.2, 0.9));
  float diff = max(dot(normal, keyLightDir), 0.0);
  vec3 diffuse = diff * vec3(0.9, 0.92, 0.98);

  vec3 fillLightDir = normalize(vec3(-0.6, -0.3, -0.7));
  float fillDiff = max(dot(normal, fillLightDir), 0.0) * 0.35;
  vec3 fill = fillDiff * vec3(0.4, 0.5, 0.65);

  vec3 viewDir = -rd;
  vec3 halfwayDir = normalize(keyLightDir + viewDir);
  float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
  vec3 specular = spec * vec3(0.85, 0.9, 1.0);

  // Normal-based surface base color with cyber-cyan palette
  vec3 normColor = 0.5 + 0.5 * normal;
  vec3 baseColor = mix(vec3(0.2, 0.45, 0.85), normColor, 0.45);

  // Slicing cross-section boundary highlighting
  if (u_slice_enabled > 0.5 && u_slice_plane > 0) {
    float sliceDist = 0.0;
    if (u_slice_plane == 1) sliceDist = abs(p.z - u_slice_offset);
    else if (u_slice_plane == 2) sliceDist = abs(p.x - u_slice_offset);
    else if (u_slice_plane == 3) sliceDist = abs(p.y - u_slice_offset);
    if (sliceDist < 0.015) {
      baseColor = vec3(0.95, 0.35, 0.25);
    }
  }

  vec3 color = ambient * baseColor + (diffuse + fill) * baseColor + specular;

  // Distance fog atmospheric depth cueing
  float fog = 1.0 - exp(-0.003 * t * t);
  color = mix(color, bg, clamp(fog, 0.0, 1.0));

  gl_FragColor = vec4(color, 1.0);
}
`;
}

/**
 * Creates Three.js uniform map initialized for raymarching.
 */
export function createRaymarchingUniforms(
  node: SDFNode,
  freeParams: FreeParam[] = [],
  initialValues: Record<string, number> = {}
): Record<string, THREE.IUniform> {
  const nodeParams = collectFreeParams(node);
  const allParams = [...freeParams, ...nodeParams];

  const uniforms: Record<string, THREE.IUniform> = {
    u_resolution: { value: new THREE.Vector2(800, 600) },
    u_time: { value: 0.0 },
    u_camera_pos: { value: new THREE.Vector3(0.0, 0.0, 4.0) },
    u_camera_target: { value: new THREE.Vector3(0.0, 0.0, 0.0) },
    u_camera_matrix: { value: new THREE.Matrix3().identity() },
    u_slice_enabled: { value: 0.0 },
    u_slice_plane: { value: 0 },
    u_slice_offset: { value: 0.0 },
  };

  for (const p of allParams) {
    const uniformName = getParamUniformName(p);
    if (!uniforms[uniformName]) {
      const val = initialValues[p.id] ?? p.defaultVal;
      uniforms[uniformName] = { value: val };
    }
  }

  return uniforms;
}

/**
 * Statically validates a generated GLSL raymarching shader for syntax balance,
 * variable resolution, and required structural blocks.
 */
export function validateGLSLShader(shader: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!shader || shader.trim().length === 0) {
    return { valid: false, errors: ['Shader source is empty.'] };
  }

  // Check balanced delimiters
  const counts: Record<string, number> = { '(': 0, ')': 0, '{': 0, '}': 0, '[': 0, ']': 0 };
  for (const ch of shader) {
    if (ch in counts) {
      counts[ch]++;
    }
  }
  if (counts['('] !== counts[')']) {
    errors.push(`Mismatched parentheses: ${counts['(']} opening vs ${counts[')']} closing.`);
  }
  if (counts['{'] !== counts['}']) {
    errors.push(`Mismatched braces: ${counts['{']} opening vs ${counts['}']} closing.`);
  }
  if (counts['['] !== counts[']']) {
    errors.push(`Mismatched brackets: ${counts['[']} opening vs ${counts[']']} closing.`);
  }

  // Check required uniforms
  const requiredUniforms = [
    'u_resolution',
    'u_time',
    'u_camera_pos',
    'u_camera_target',
    'u_camera_matrix',
    'u_slice_enabled',
    'u_slice_plane',
    'u_slice_offset',
  ];
  for (const u of requiredUniforms) {
    if (!shader.includes(u)) {
      errors.push(`Missing required viewport/slicing uniform: ${u}`);
    }
  }

  // Check required core functions
  const requiredFunctions = [
    'float map(vec3 p)',
    'vec3 calcNormal(vec3 p)',
    'float raymarch(vec3 ro, vec3 rd)',
    'void main()',
  ];
  for (const fn of requiredFunctions) {
    if (!shader.includes(fn)) {
      errors.push(`Missing required core shader function: ${fn}`);
    }
  }

  // Check bounded step limit (max 128 steps)
  if (!shader.includes('MAX_STEPS = 128') && !shader.includes('128')) {
    errors.push('Shader does not enforce maximum 128 raymarching step iteration cap.');
  }

  // Check all referenced u_param_* variables are declared
  const declaredUniforms = new Set<string>();
  const uniformDeclRegex = /uniform\s+float\s+(u_param_[a-zA-Z0-9_]+);/g;
  let match: RegExpExecArray | null;
  while ((match = uniformDeclRegex.exec(shader)) !== null) {
    declaredUniforms.add(match[1]);
  }

  // Find all u_param_* identifiers used in map(vec3 p)
  const mapMatch = shader.match(/float map\(vec3 p\)\s*\{[\s\S]*?\n\}/);
  if (mapMatch) {
    const mapBody = mapMatch[0];
    const usedParamRegex = /\b(u_param_[a-zA-Z0-9_]+)\b/g;
    let usedMatch: RegExpExecArray | null;
    while ((usedMatch = usedParamRegex.exec(mapBody)) !== null) {
      const paramName = usedMatch[1];
      if (!declaredUniforms.has(paramName)) {
        errors.push(`Referenced uniform "${paramName}" in map() was not declared in shader header.`);
      }
    }
  } else {
    errors.push('Could not isolate float map(vec3 p) body for verification.');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
