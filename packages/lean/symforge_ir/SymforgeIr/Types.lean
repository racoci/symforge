import Lean.Data.Json

namespace SymforgeIr

open Lean

/-- Free parameter to be optimized via gradient descent or black-box search -/
structure FreeParam where
  id : String
  minVal : Float
  maxVal : Float
  defaultVal : Float
  deriving Repr, BEq, Inhabited, ToJson, FromJson

/-- Inductive grammar for 1D Procedural Audio & DSP synthesis -/
inductive AudioNode : Type where
  | sine (freq : FreeParam) (phase : FreeParam) : AudioNode
  | saw (freq : FreeParam) : AudioNode
  | square (freq : FreeParam) (duty : FreeParam) : AudioNode
  | noise (seed : Int) : AudioNode
  | adsr (attack : FreeParam) (decay : FreeParam) (sustain : FreeParam) (release : FreeParam) (gate : AudioNode) : AudioNode
  | lowpass (cutoff : FreeParam) (resonance : FreeParam) (input : AudioNode) : AudioNode
  | gain (gain : FreeParam) (input : AudioNode) : AudioNode
  | mix (ratio : FreeParam) (input1 input2 : AudioNode) : AudioNode
  | fm (carrier : FreeParam) (modFreq : FreeParam) (modIndex : FreeParam) : AudioNode
  deriving Repr, BEq, Inhabited, ToJson, FromJson

/-- Inductive grammar for 3D Implicit Surface (SDF) and CSG Geometry -/
inductive SDFNode : Type where
  | sphere (radius : FreeParam) : SDFNode
  | box (width : FreeParam) (height : FreeParam) (depth : FreeParam) : SDFNode
  | cylinder (radius : FreeParam) (height : FreeParam) : SDFNode
  | torus (majorRadius : FreeParam) (minorRadius : FreeParam) : SDFNode
  | translate (dx : FreeParam) (dy : FreeParam) (dz : FreeParam) (child : SDFNode) : SDFNode
  | rotate (rx : FreeParam) (ry : FreeParam) (rz : FreeParam) (child : SDFNode) : SDFNode
  | scale (s : FreeParam) (child : SDFNode) : SDFNode
  | union (smoothness : FreeParam) (a b : SDFNode) : SDFNode
  | intersection (a b : SDFNode) : SDFNode
  | difference (a b : SDFNode) : SDFNode
  deriving Repr, BEq, Inhabited, ToJson, FromJson

/-- Inductive grammar for Hardware Instructions and Superoptimization -/
inductive HardwareNode : Type where
  | reg (name : String) : HardwareNode
  | imm (val : Int) : HardwareNode
  | add (dst : String) (src1 src2 : HardwareNode) : HardwareNode
  | sub (dst : String) (src1 src2 : HardwareNode) : HardwareNode
  | mul (dst : String) (src1 src2 : HardwareNode) : HardwareNode
  | shl (dst : String) (src : HardwareNode) (shift : Int) : HardwareNode
  | shr (dst : String) (src : HardwareNode) (shift : Int) : HardwareNode
  | seq (first second : HardwareNode) : HardwareNode
  deriving Repr, BEq, Inhabited, ToJson, FromJson

/-- Unified Domain-Agnostic Node Container -/
inductive UniversalNode : Type where
  | audio (node : AudioNode) : UniversalNode
  | sdf (node : SDFNode) : UniversalNode
  | hardware (node : HardwareNode) : UniversalNode
  | macroNode (name : String) (children : List UniversalNode) : UniversalNode
  deriving Repr, BEq, Inhabited, ToJson, FromJson

/-- Compute the node size (number of AST vertices) of an AudioNode -/
def AudioNode.size : AudioNode → Nat
  | .sine _ _ => 1
  | .saw _ => 1
  | .square _ _ => 1
  | .noise _ => 1
  | .adsr _ _ _ _ g => 1 + g.size
  | .lowpass _ _ i => 1 + i.size
  | .gain _ i => 1 + i.size
  | .mix _ a b => 1 + a.size + b.size
  | .fm _ _ _ => 1

/-- Compute the tree depth of an AudioNode -/
def AudioNode.depth : AudioNode → Nat
  | .sine _ _ => 1
  | .saw _ => 1
  | .square _ _ => 1
  | .noise _ => 1
  | .adsr _ _ _ _ g => 1 + g.depth
  | .lowpass _ _ i => 1 + i.depth
  | .gain _ i => 1 + i.depth
  | .mix _ a b => 1 + Nat.max a.depth b.depth
  | .fm _ _ _ => 1

/-- Extract all continuous FreeParams from an AudioNode -/
def AudioNode.collectParams : AudioNode → List FreeParam
  | .sine f p => [f, p]
  | .saw f => [f]
  | .square f d => [f, d]
  | .noise _ => []
  | .adsr a d s r g => [a, d, s, r] ++ g.collectParams
  | .lowpass c res i => [c, res] ++ i.collectParams
  | .gain g i => [g] ++ i.collectParams
  | .mix r a b => [r] ++ a.collectParams ++ b.collectParams
  | .fm c mf mi => [c, mf, mi]

/-- Compute the node size of an SDFNode -/
def SDFNode.size : SDFNode → Nat
  | .sphere _ => 1
  | .box _ _ _ => 1
  | .cylinder _ _ => 1
  | .torus _ _ => 1
  | .translate _ _ _ c => 1 + c.size
  | .rotate _ _ _ c => 1 + c.size
  | .scale _ c => 1 + c.size
  | .union _ a b => 1 + a.size + b.size
  | .intersection a b => 1 + a.size + b.size
  | .difference a b => 1 + a.size + b.size

/-- Compute the tree depth of an SDFNode -/
def SDFNode.depth : SDFNode → Nat
  | .sphere _ => 1
  | .box _ _ _ => 1
  | .cylinder _ _ => 1
  | .torus _ _ => 1
  | .translate _ _ _ c => 1 + c.depth
  | .rotate _ _ _ c => 1 + c.depth
  | .scale _ c => 1 + c.depth
  | .union _ a b => 1 + Nat.max a.depth b.depth
  | .intersection a b => 1 + Nat.max a.depth b.depth
  | .difference a b => 1 + Nat.max a.depth b.depth

/-- Extract all continuous FreeParams from an SDFNode -/
def SDFNode.collectParams : SDFNode → List FreeParam
  | .sphere r => [r]
  | .box w h d => [w, h, d]
  | .cylinder r h => [r, h]
  | .torus r1 r2 => [r1, r2]
  | .translate dx dy dz c => [dx, dy, dz] ++ c.collectParams
  | .rotate rx ry rz c => [rx, ry, rz] ++ c.collectParams
  | .scale s c => [s] ++ c.collectParams
  | .union sm a b => [sm] ++ a.collectParams ++ b.collectParams
  | .intersection a b => a.collectParams ++ b.collectParams
  | .difference a b => a.collectParams ++ b.collectParams

end SymforgeIr
