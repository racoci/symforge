import { describe, it, expect } from 'vitest';
import {
  generateSDFExpression,
  generateRaymarchingFragmentShader,
  validateGLSLShader,
  sanitizeParamId,
  getParamUniformName,
  createRaymarchingUniforms,
} from './glslCodegen';
import {
  createSphere,
  createBox,
  createCylinder,
  createTorus,
  createPlane,
  createTranslate,
  createRotate,
  createScale,
  createUnion,
  createSmoothUnion,
  createIntersection,
  createDifference,
  createFreeParam,
} from '../../../core/ast/models';
import type { SDFNode } from '../../../core/ast/types';

describe('GLSL Raymarching Codegen', () => {
  describe('Parameter Sanitization & Uniform Naming', () => {
    it('sanitizes special characters into valid GLSL identifiers', () => {
      expect(sanitizeParamId('radius-x')).toBe('radius_x');
      expect(sanitizeParamId('param.123')).toBe('param_123');
      expect(sanitizeParamId('123abc')).toBe('_123abc');
      expect(sanitizeParamId('normal_id')).toBe('normal_id');
    });

    it('generates uniform names matching FreeParam records', () => {
      const p = createFreeParam('head_radius', 0.1, 10.0, 1.0);
      expect(getParamUniformName(p)).toBe('u_param_head_radius');
      expect(getParamUniformName('torus-thickness')).toBe('u_param_torus_thickness');
    });
  });

  describe('Analytical Primitives GLSL Codegen', () => {
    it('generates sdSphere for SphereSDFNode', () => {
      const sphere = createSphere(createFreeParam('radius', 0.1, 5.0, 1.0));
      const expr = generateSDFExpression(sphere, 'p');
      expect(expr).toBe('sdSphere(p, u_param_radius)');
    });

    it('generates sdBox for BoxSDFNode', () => {
      const box = createBox(
        createFreeParam('w', 0.1, 5.0, 1.0),
        createFreeParam('h', 0.1, 5.0, 2.0),
        createFreeParam('d', 0.1, 5.0, 0.5)
      );
      const expr = generateSDFExpression(box, 'p');
      expect(expr).toBe('sdBox(p, vec3(u_param_w, u_param_h, u_param_d))');
    });

    it('generates sdCylinder for CylinderSDFNode', () => {
      const cyl = createCylinder(
        createFreeParam('r', 0.1, 2.0, 0.5),
        createFreeParam('h', 0.1, 10.0, 3.0)
      );
      const expr = generateSDFExpression(cyl, 'p');
      expect(expr).toBe('sdCylinder(p, u_param_r, u_param_h)');
    });

    it('generates sdTorus for TorusSDFNode', () => {
      const torus = createTorus(
        createFreeParam('R', 0.5, 5.0, 2.0),
        createFreeParam('r', 0.05, 2.0, 0.4)
      );
      const expr = generateSDFExpression(torus, 'p');
      expect(expr).toBe('sdTorus(p, vec2(u_param_R, u_param_r))');
    });

    it('generates sdPlane for PlaneSDFNode', () => {
      const plane = createPlane(createFreeParam('h_offset', -5.0, 5.0, 0.0));
      const expr = generateSDFExpression(plane, 'p');
      expect(expr).toBe('sdPlane(p, vec3(0.0, 1.0, 0.0), u_param_h_offset)');
    });
  });

  describe('Spatial Transforms & CSG Operations GLSL Codegen', () => {
    it('generates translated coordinate space for TranslateSDFNode', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const trans = createTranslate(
        createFreeParam('dx', -5.0, 5.0, 1.0),
        createFreeParam('dy', -5.0, 5.0, 2.0),
        createFreeParam('dz', -5.0, 5.0, -1.0),
        sphere
      );
      const expr = generateSDFExpression(trans, 'p');
      expect(expr).toBe('sdSphere((p - vec3(u_param_dx, u_param_dy, u_param_dz)), u_param_r)');
    });

    it('generates Euler rotation transform for RotateSDFNode', () => {
      const box = createBox(
        createFreeParam('w', 0.1, 2.0, 1.0),
        createFreeParam('h', 0.1, 2.0, 1.0),
        createFreeParam('d', 0.1, 2.0, 1.0)
      );
      const rot = createRotate(
        createFreeParam('rx', 0.0, 6.28, 0.5),
        createFreeParam('ry', 0.0, 6.28, 1.2),
        createFreeParam('rz', 0.0, 6.28, 0.0),
        box
      );
      const expr = generateSDFExpression(rot, 'p');
      expect(expr).toBe('sdBox(rotateEuler(p, vec3(u_param_rx, u_param_ry, u_param_rz)), vec3(u_param_w, u_param_h, u_param_d))');
    });

    it('generates scale transform with distance compensation for ScaleSDFNode', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const scale = createScale(createFreeParam('s', 0.1, 5.0, 2.0), sphere);
      const expr = generateSDFExpression(scale, 'p');
      expect(expr).toBe('(sdSphere((p / u_param_s), u_param_r) * u_param_s)');
    });

    it('generates min(d1, d2) for UnionSDFNode', () => {
      const s1 = createSphere(createFreeParam('r1', 0.1, 5.0, 1.0));
      const s2 = createSphere(createFreeParam('r2', 0.1, 5.0, 0.8));
      const union = createUnion(createFreeParam('smooth', 0.0, 1.0, 0.0), s1, s2);
      const expr = generateSDFExpression(union, 'p');
      expect(expr).toBe('min(sdSphere(p, u_param_r1), sdSphere(p, u_param_r2))');
    });

    it('generates opSmoothUnion(d1, d2, k) for SmoothUnionSDFNode', () => {
      const s1 = createSphere(createFreeParam('r1', 0.1, 5.0, 1.0));
      const s2 = createSphere(createFreeParam('r2', 0.1, 5.0, 0.8));
      const sm = createSmoothUnion(createFreeParam('blend_k', 0.0, 1.0, 0.25), s1, s2);
      const expr = generateSDFExpression(sm, 'p');
      expect(expr).toBe('opSmoothUnion(sdSphere(p, u_param_r1), sdSphere(p, u_param_r2), u_param_blend_k)');
    });

    it('generates max(d1, d2) for IntersectionSDFNode', () => {
      const s1 = createSphere(createFreeParam('r1', 0.1, 5.0, 1.0));
      const s2 = createSphere(createFreeParam('r2', 0.1, 5.0, 0.8));
      const inter = createIntersection(s1, s2);
      const expr = generateSDFExpression(inter, 'p');
      expect(expr).toBe('max(sdSphere(p, u_param_r1), sdSphere(p, u_param_r2))');
    });

    it('generates max(d1, -(d2)) for DifferenceSDFNode', () => {
      const s1 = createSphere(createFreeParam('r1', 0.1, 5.0, 1.0));
      const s2 = createSphere(createFreeParam('r2', 0.1, 5.0, 0.8));
      const diff = createDifference(s1, s2);
      const expr = generateSDFExpression(diff, 'p');
      expect(expr).toBe('max(sdSphere(p, u_param_r1), -(sdSphere(p, u_param_r2)))');
    });
  });

  describe('Raymarching Fragment Shader Assembly & Invariants', () => {
    it('declares all required viewport uniforms and slicing controls', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const shader = generateRaymarchingFragmentShader(sphere);

      expect(shader).toContain('uniform vec2 u_resolution;');
      expect(shader).toContain('uniform float u_time;');
      expect(shader).toContain('uniform vec3 u_camera_pos;');
      expect(shader).toContain('uniform vec3 u_camera_target;');
      expect(shader).toContain('uniform mat3 u_camera_matrix;');
      expect(shader).toContain('uniform float u_slice_enabled;');
      expect(shader).toContain('uniform int u_slice_plane;');
      expect(shader).toContain('uniform float u_slice_offset;');
    });

    it('declares uniform float u_param_<id> for all AST FreeParam nodes', () => {
      const box = createBox(
        createFreeParam('width_x', 0.1, 5.0, 1.0),
        createFreeParam('height_y', 0.1, 5.0, 2.0),
        createFreeParam('depth_z', 0.1, 5.0, 0.5)
      );
      const shader = generateRaymarchingFragmentShader(box);

      expect(shader).toContain('uniform float u_param_width_x;');
      expect(shader).toContain('uniform float u_param_height_y;');
      expect(shader).toContain('uniform float u_param_depth_z;');
    });

    it('enforces maximum 128 raymarching step limit, precision, and max distance', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const shader = generateRaymarchingFragmentShader(sphere);

      expect(shader).toContain('const int MAX_STEPS = 128;');
      expect(shader).toContain('const float PRECISION = 0.001;');
      expect(shader).toContain('const float MAX_DIST = 25.0;');
      expect(shader).toContain('for (int i = 0; i < MAX_STEPS; i++)');
    });

    it('includes central finite differences normal estimator calcNormal(vec3 p)', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const shader = generateRaymarchingFragmentShader(sphere);

      expect(shader).toContain('vec3 calcNormal(vec3 p)');
      expect(shader).toContain('const float eps = 0.001;');
      expect(shader).toContain('const vec2 h = vec2(eps, 0.0);');
      expect(shader).toContain('map(p + h.xyy) - map(p - h.xyy)');
    });

    it('includes Blinn-Phong specular and ambient illumination model', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const shader = generateRaymarchingFragmentShader(sphere);

      expect(shader).toContain('ambient');
      expect(shader).toContain('diffuse');
      expect(shader).toContain('specular');
      expect(shader).toContain('halfwayDir');
      expect(shader).toContain('pow(max(dot(normal, halfwayDir), 0.0), 32.0)');
    });

    it('includes slicing plane cross-section rendering in map() and fragment shader', () => {
      const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
      const shader = generateRaymarchingFragmentShader(sphere);

      expect(shader).toContain('u_slice_enabled');
      expect(shader).toContain('u_slice_plane == 1');
      expect(shader).toContain('p.z - u_slice_offset');
      expect(shader).toContain('p.x - u_slice_offset');
      expect(shader).toContain('p.y - u_slice_offset');
    });

    it('creates Three.js uniform map with matching default values', () => {
      const sphere = createSphere(createFreeParam('rad', 0.1, 5.0, 1.25));
      const uniforms = createRaymarchingUniforms(sphere);

      expect(uniforms.u_resolution).toBeDefined();
      expect(uniforms.u_param_rad).toBeDefined();
      expect(uniforms.u_param_rad.value).toBe(1.25);
    });
  });

  describe('Pass-to-Complete Gate: 20 Complex CSG Geometric Trees', () => {
    // 20 distinct, architecturally challenging CSG geometric models
    const testCases: { name: string; tree: SDFNode }[] = [
      {
        name: '1. Hollow Sphere (CSG Difference of nested spheres)',
        tree: createDifference(
          createSphere(createFreeParam('outer_r', 0.5, 5.0, 2.0)),
          createSphere(createFreeParam('inner_r', 0.1, 4.0, 1.5))
        ),
      },
      {
        name: '2. Rounded Box Frame (Box with 3 orthogonal drilled cylinders)',
        tree: createDifference(
          createDifference(
            createDifference(
              createBox(
                createFreeParam('bx_w', 0.5, 4.0, 1.5),
                createFreeParam('bx_h', 0.5, 4.0, 1.5),
                createFreeParam('bx_d', 0.5, 4.0, 1.5)
              ),
              createCylinder(createFreeParam('drill_r1', 0.1, 2.0, 0.6), createFreeParam('drill_h1', 1.0, 10.0, 4.0))
            ),
            createRotate(
              createFreeParam('rot_x1', 0.0, 3.14, 1.5708),
              createFreeParam('rot_y1', 0.0, 3.14, 0.0),
              createFreeParam('rot_z1', 0.0, 3.14, 0.0),
              createCylinder(createFreeParam('drill_r2', 0.1, 2.0, 0.6), createFreeParam('drill_h2', 1.0, 10.0, 4.0))
            )
          ),
          createRotate(
            createFreeParam('rot_x2', 0.0, 3.14, 0.0),
            createFreeParam('rot_y2', 0.0, 3.14, 0.0),
            createFreeParam('rot_z2', 0.0, 3.14, 1.5708),
            createCylinder(createFreeParam('drill_r3', 0.1, 2.0, 0.6), createFreeParam('drill_h3', 1.0, 10.0, 4.0))
          )
        ),
      },
      {
        name: '3. CSG Mechanical Bolt with Torus Groove',
        tree: createUnion(
          createFreeParam('bolt_sm', 0.0, 1.0, 0.0),
          createCylinder(createFreeParam('shank_r', 0.1, 2.0, 0.5), createFreeParam('shank_h', 0.5, 10.0, 3.0)),
          createTranslate(
            createFreeParam('head_dx', -1.0, 1.0, 0.0),
            createFreeParam('head_dy', -5.0, 5.0, 1.6),
            createFreeParam('head_dz', -1.0, 1.0, 0.0),
            createDifference(
              createCylinder(createFreeParam('head_r', 0.5, 3.0, 1.0), createFreeParam('head_h', 0.1, 2.0, 0.5)),
              createTorus(createFreeParam('groove_R', 0.5, 2.0, 0.9), createFreeParam('groove_r', 0.01, 0.5, 0.1))
            )
          )
        ),
      },
      {
        name: '4. Organic Metaspheres (SmoothUnion of 4 tetrahedral spheres)',
        tree: createSmoothUnion(
          createFreeParam('tetra_k1', 0.0, 1.0, 0.3),
          createSmoothUnion(
            createFreeParam('tetra_k2', 0.0, 1.0, 0.3),
            createTranslate(createFreeParam('t1_x', -3, 3, 0), createFreeParam('t1_y', -3, 3, 1), createFreeParam('t1_z', -3, 3, 0), createSphere(createFreeParam('sr1', 0.1, 2, 0.7))),
            createTranslate(createFreeParam('t2_x', -3, 3, 0.94), createFreeParam('t2_y', -3, 3, -0.33), createFreeParam('t2_z', -3, 3, 0), createSphere(createFreeParam('sr2', 0.1, 2, 0.7)))
          ),
          createSmoothUnion(
            createFreeParam('tetra_k3', 0.0, 1.0, 0.3),
            createTranslate(createFreeParam('t3_x', -3, 3, -0.47), createFreeParam('t3_y', -3, 3, -0.33), createFreeParam('t3_z', -3, 3, 0.81), createSphere(createFreeParam('sr3', 0.1, 2, 0.7))),
            createTranslate(createFreeParam('t4_x', -3, 3, -0.47), createFreeParam('t4_y', -3, 3, -0.33), createFreeParam('t4_z', -3, 3, -0.81), createSphere(createFreeParam('sr4', 0.1, 2, 0.7)))
          )
        ),
      },
      {
        name: '5. Classical Fluted Column (Cylinder with subtracted flutes and base torus)',
        tree: createUnion(
          createFreeParam('col_u', 0, 1, 0),
          createDifference(
            createCylinder(createFreeParam('col_r', 0.5, 3.0, 1.0), createFreeParam('col_h', 1.0, 10.0, 4.0)),
            createTranslate(
              createFreeParam('flute_dx', -2, 2, 1.0),
              createFreeParam('flute_dy', -2, 2, 0.0),
              createFreeParam('flute_dz', -2, 2, 0.0),
              createCylinder(createFreeParam('flute_r', 0.05, 0.5, 0.15), createFreeParam('flute_h', 1, 10, 4.2))
            )
          ),
          createTranslate(
            createFreeParam('base_dx', -1, 1, 0),
            createFreeParam('base_dy', -5, 5, -2.1),
            createFreeParam('base_dz', -1, 1, 0),
            createTorus(createFreeParam('base_R', 0.5, 3.0, 1.2), createFreeParam('base_r', 0.05, 1.0, 0.2))
          )
        ),
      },
      {
        name: '6. Nested Gimbal Rings (3 orthogonal rotated tori)',
        tree: createUnion(
          createFreeParam('gimb_u1', 0, 1, 0),
          createTorus(createFreeParam('gimb_R1', 1.0, 5.0, 2.5), createFreeParam('gimb_r1', 0.05, 0.5, 0.12)),
          createUnion(
            createFreeParam('gimb_u2', 0, 1, 0),
            createRotate(
              createFreeParam('gimb_rx1', 0, 3.14, 1.5708),
              createFreeParam('gimb_ry1', 0, 3.14, 0),
              createFreeParam('gimb_rz1', 0, 3.14, 0),
              createTorus(createFreeParam('gimb_R2', 0.8, 4.0, 2.0), createFreeParam('gimb_r2', 0.05, 0.5, 0.12))
            ),
            createRotate(
              createFreeParam('gimb_rx2', 0, 3.14, 0),
              createFreeParam('gimb_ry2', 0, 3.14, 1.5708),
              createFreeParam('gimb_rz2', 0, 3.14, 0),
              createTorus(createFreeParam('gimb_R3', 0.6, 3.0, 1.5), createFreeParam('gimb_r3', 0.05, 0.5, 0.12))
            )
          )
        ),
      },
      {
        name: '7. Engine Piston Cavity (Cylinder minus pin bores and ring grooves)',
        tree: createDifference(
          createCylinder(createFreeParam('piston_r', 0.5, 4.0, 1.5), createFreeParam('piston_h', 0.5, 6.0, 2.0)),
          createUnion(
            createFreeParam('pist_u', 0, 1, 0),
            createRotate(
              createFreeParam('bore_rx', 0, 3.14, 1.5708),
              createFreeParam('bore_ry', 0, 3.14, 0),
              createFreeParam('bore_rz', 0, 3.14, 0),
              createCylinder(createFreeParam('wristpin_r', 0.1, 1.0, 0.4), createFreeParam('wristpin_h', 0.5, 5.0, 3.5))
            ),
            createTranslate(
              createFreeParam('bowl_x', -1, 1, 0),
              createFreeParam('bowl_y', -3, 3, 0.9),
              createFreeParam('bowl_z', -1, 1, 0),
              createSphere(createFreeParam('bowl_r', 0.2, 2.0, 0.9))
            )
          )
        ),
      },
      {
        name: '8. Sliced Mobius Ribbon Loop (Rotated translated tori with smooth blend)',
        tree: createSmoothUnion(
          createFreeParam('mob_k', 0.01, 1.0, 0.35),
          createTorus(createFreeParam('m_R1', 0.5, 3.0, 1.8), createFreeParam('m_r1', 0.05, 0.5, 0.15)),
          createRotate(
            createFreeParam('m_rx', 0, 3.14, 0.5),
            createFreeParam('m_ry', 0, 3.14, 0.8),
            createFreeParam('m_rz', 0, 3.14, 0.2),
            createTorus(createFreeParam('m_R2', 0.5, 3.0, 1.6), createFreeParam('m_r2', 0.05, 0.5, 0.15))
          )
        ),
      },
      {
        name: '9. Stepped Architectural Arch (Wall box minus translated curved cylinder)',
        tree: createDifference(
          createBox(createFreeParam('arch_w', 1, 5, 3), createFreeParam('arch_h', 1, 5, 4), createFreeParam('arch_d', 0.1, 2, 0.6)),
          createTranslate(
            createFreeParam('arch_cx', -2, 2, 0),
            createFreeParam('arch_cy', -3, 3, -1.0),
            createFreeParam('arch_cz', -1, 1, 0),
            createRotate(
              createFreeParam('arch_rx', 0, 3.14, 1.5708),
              createFreeParam('arch_ry', 0, 3.14, 0),
              createFreeParam('arch_rz', 0, 3.14, 0),
              createCylinder(createFreeParam('arch_r', 0.5, 3.0, 1.2), createFreeParam('arch_dep', 0.5, 4.0, 1.5))
            )
          )
        ),
      },
      {
        name: '10. Gaming Die (Box minus spherical face pips)',
        tree: createDifference(
          createBox(createFreeParam('die_w', 0.5, 3, 1), createFreeParam('die_h', 0.5, 3, 1), createFreeParam('die_d', 0.5, 3, 1)),
          createUnion(
            createFreeParam('die_u', 0, 1, 0),
            createTranslate(createFreeParam('pip1_x', -2, 2, 0), createFreeParam('pip1_y', -2, 2, 1.0), createFreeParam('pip1_z', -2, 2, 0), createSphere(createFreeParam('pip1_r', 0.05, 0.5, 0.2))),
            createTranslate(createFreeParam('pip2_x', -2, 2, 0), createFreeParam('pip2_y', -2, 2, -1.0), createFreeParam('pip2_z', -2, 2, 0), createSphere(createFreeParam('pip2_r', 0.05, 0.5, 0.2)))
          )
        ),
      },
      {
        name: '11. Mechanical Sprocket (Cylinder hub union teeth boxes minus axle cylinder)',
        tree: createDifference(
          createUnion(
            createFreeParam('sprock_u', 0, 1, 0),
            createCylinder(createFreeParam('sp_hub_r', 0.5, 3, 1.5), createFreeParam('sp_hub_h', 0.1, 2, 0.4)),
            createBox(createFreeParam('tooth_w', 0.1, 2, 2.2), createFreeParam('tooth_h', 0.1, 1, 0.35), createFreeParam('tooth_d', 0.1, 1, 0.35))
          ),
          createCylinder(createFreeParam('axle_r', 0.1, 1.5, 0.4), createFreeParam('axle_h', 0.5, 4.0, 1.0))
        ),
      },
      {
        name: '12. Molecular Orb Cluster (Central sphere smooth-blended with 6 axial spheres)',
        tree: createSmoothUnion(
          createFreeParam('mol_k1', 0.01, 1, 0.25),
          createSphere(createFreeParam('mol_core_r', 0.5, 3, 1.2)),
          createSmoothUnion(
            createFreeParam('mol_k2', 0.01, 1, 0.25),
            createTranslate(createFreeParam('mx_p', -3, 3, 1.5), createFreeParam('my_0', -1, 1, 0), createFreeParam('mz_0', -1, 1, 0), createSphere(createFreeParam('sat_r1', 0.1, 1, 0.45))),
            createTranslate(createFreeParam('mx_n', -3, 3, -1.5), createFreeParam('my_02', -1, 1, 0), createFreeParam('mz_02', -1, 1, 0), createSphere(createFreeParam('sat_r2', 0.1, 1, 0.45)))
          )
        ),
      },
      {
        name: '13. Torus Knot Intersect (Two rotated intersecting tori)',
        tree: createIntersection(
          createTorus(createFreeParam('tk_R1', 0.5, 4, 1.8), createFreeParam('tk_r1', 0.1, 1, 0.35)),
          createRotate(
            createFreeParam('tk_rx', 0, 3.14, 1.2),
            createFreeParam('tk_ry', 0, 3.14, 0.6),
            createFreeParam('tk_rz', 0, 3.14, 0.0),
            createTorus(createFreeParam('tk_R2', 0.5, 4, 1.8), createFreeParam('tk_r2', 0.1, 1, 0.35))
          )
        ),
      },
      {
        name: '14. Hollow Flanged Pipe (Cylinder minus inner cylinder union flange)',
        tree: createUnion(
          createFreeParam('flange_u', 0, 1, 0),
          createDifference(
            createCylinder(createFreeParam('pipe_out_r', 0.5, 3, 1.0), createFreeParam('pipe_h', 1, 8, 3.5)),
            createCylinder(createFreeParam('pipe_in_r', 0.1, 2.5, 0.8), createFreeParam('pipe_in_h', 1, 9, 3.8))
          ),
          createTranslate(
            createFreeParam('flg_x', -1, 1, 0),
            createFreeParam('flg_y', -5, 5, 1.7),
            createFreeParam('flg_z', -1, 1, 0),
            createTorus(createFreeParam('flg_R', 0.5, 3, 1.1), createFreeParam('flg_r', 0.05, 1, 0.2))
          )
        ),
      },
      {
        name: '15. Star Polyhedron (Orthogonal intersecting rotated boxes)',
        tree: createIntersection(
          createBox(createFreeParam('star_w1', 0.5, 3, 1.2), createFreeParam('star_h1', 0.5, 3, 1.2), createFreeParam('star_d1', 0.5, 3, 1.2)),
          createRotate(
            createFreeParam('star_rx', 0, 3.14, 0.7854),
            createFreeParam('star_ry', 0, 3.14, 0.7854),
            createFreeParam('star_rz', 0, 3.14, 0.0),
            createBox(createFreeParam('star_w2', 0.5, 3, 1.2), createFreeParam('star_h2', 0.5, 3, 1.2), createFreeParam('star_d2', 0.5, 3, 1.2))
          )
        ),
      },
      {
        name: '16. Chamfered Box with Spherical Inset (Box intersect sphere minus torus)',
        tree: createDifference(
          createIntersection(
            createBox(createFreeParam('chamf_w', 0.5, 3, 1.4), createFreeParam('chamf_h', 0.5, 3, 1.4), createFreeParam('chamf_d', 0.5, 3, 1.4)),
            createSphere(createFreeParam('chamf_sr', 0.5, 4, 1.8))
          ),
          createTorus(createFreeParam('chamf_tR', 0.5, 3, 1.3), createFreeParam('chamf_tr', 0.05, 1, 0.15))
        ),
      },
      {
        name: '17. Hydraulic Valve Manifold (Block with orthogonal crossed fluid passages)',
        tree: createDifference(
          createBox(createFreeParam('valve_w', 1, 5, 2.0), createFreeParam('valve_h', 1, 5, 2.0), createFreeParam('valve_d', 1, 5, 2.0)),
          createUnion(
            createFreeParam('pass_u', 0, 1, 0),
            createCylinder(createFreeParam('pass_r1', 0.1, 1.5, 0.5), createFreeParam('pass_h1', 1, 8, 3.0)),
            createRotate(
              createFreeParam('pass_rx', 0, 3.14, 1.5708),
              createFreeParam('pass_ry', 0, 3.14, 0),
              createFreeParam('pass_rz', 0, 3.14, 0),
              createCylinder(createFreeParam('pass_r2', 0.1, 1.5, 0.5), createFreeParam('pass_h2', 1, 8, 3.0))
            )
          )
        ),
      },
      {
        name: '18. Scaled Self-Similar Spheres (Nested scale with translate)',
        tree: createUnion(
          createFreeParam('ss_u', 0, 1, 0),
          createSphere(createFreeParam('ss_r0', 0.5, 4, 1.5)),
          createScale(
            createFreeParam('ss_s1', 0.1, 2, 0.5),
            createTranslate(
              createFreeParam('ss_dx', -5, 5, 2.5),
              createFreeParam('ss_dy', -5, 5, 1.0),
              createFreeParam('ss_dz', -5, 5, 0.0),
              createSphere(createFreeParam('ss_r1', 0.5, 4, 1.5))
            )
          )
        ),
      },
      {
        name: '19. Aerodynamic Fin / Hydrofoil (Rotated scaled cylinder intersecting plane)',
        tree: createIntersection(
          createRotate(
            createFreeParam('fin_rx', 0, 3.14, 0.2),
            createFreeParam('fin_ry', 0, 3.14, 0.0),
            createFreeParam('fin_rz', 0, 3.14, 1.1),
            createScale(
              createFreeParam('fin_s', 0.5, 4, 1.5),
              createCylinder(createFreeParam('foil_r', 0.1, 2, 0.7), createFreeParam('foil_h', 0.5, 5, 3.0))
            )
          ),
          createPlane(createFreeParam('ground_plane', -5, 5, 0.5))
        ),
      },
      {
        name: '20. Mechanical Ball Bearing Assembly (Rings + multiple spherical balls)',
        tree: createUnion(
          createFreeParam('brg_u1', 0, 1, 0),
          createDifference(
            createCylinder(createFreeParam('race_out_r', 1, 5, 2.5), createFreeParam('race_h', 0.2, 3, 0.8)),
            createCylinder(createFreeParam('race_mid_r', 0.5, 4, 1.9), createFreeParam('race_mid_h', 0.3, 4, 1.0))
          ),
          createUnion(
            createFreeParam('brg_u2', 0, 1, 0),
            createTranslate(createFreeParam('b1_x', -3, 3, 2.2), createFreeParam('b1_y', -1, 1, 0), createFreeParam('b1_z', -3, 3, 0), createSphere(createFreeParam('ball_r1', 0.1, 1, 0.3))),
            createTranslate(createFreeParam('b2_x', -3, 3, -2.2), createFreeParam('b2_y', -1, 1, 0), createFreeParam('b2_z', -3, 3, 0), createSphere(createFreeParam('ball_r2', 0.1, 1, 0.3)))
          )
        ),
      },
    ];

    it('passes validation for all 20 complex CSG geometric trees with 100% valid shader syntax', () => {
      expect(testCases.length).toBe(20);

      for (const tc of testCases) {
        const shader = generateRaymarchingFragmentShader(tc.tree);
        const validation = validateGLSLShader(shader);

        if (!validation.valid) {
          console.error(`Shader validation failed for "${tc.name}":`, validation.errors);
        }

        expect(validation.valid, `Shader validation failed for ${tc.name}: ${validation.errors.join('; ')}`).toBe(true);
        expect(validation.errors).toHaveLength(0);
        expect(shader).toContain('void main()');
        expect(shader).toContain('float map(vec3 p)');
        expect(shader).toContain('MAX_STEPS = 128');
      }
    });
  });
});
