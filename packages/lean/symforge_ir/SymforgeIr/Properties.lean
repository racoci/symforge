import SymforgeIr.Types

namespace SymforgeIr

theorem AudioNode.size_pos (node : AudioNode) : node.size ≥ 1 := by
  cases node <;> simp [AudioNode.size] <;> omega

theorem AudioNode.depth_pos (node : AudioNode) : node.depth ≥ 1 := by
  cases node <;> simp [AudioNode.depth] <;> omega

theorem SDFNode.size_pos (node : SDFNode) : node.size ≥ 1 := by
  cases node <;> simp [SDFNode.size] <;> omega

theorem SDFNode.depth_pos (node : SDFNode) : node.depth ≥ 1 := by
  cases node <;> simp [SDFNode.depth] <;> omega

-- Example valid procedural audio node: A filtered sawtooth oscillator
def exampleAudioPatch : AudioNode :=
  let freq := ⟨"f0", 20.0, 20000.0, 440.0⟩
  let cutoff := ⟨"fc", 100.0, 10000.0, 1200.0⟩
  let res := ⟨"q", 0.1, 10.0, 0.707⟩
  .lowpass cutoff res (.saw freq)

-- Example valid procedural 3D SDF node: A smooth union of a sphere and box
def exampleSDFPatch : SDFNode :=
  let r := ⟨"radius", 0.01, 5.0, 1.0⟩
  let w := ⟨"w", 0.01, 5.0, 0.8⟩
  let h := ⟨"h", 0.01, 5.0, 0.8⟩
  let d := ⟨"d", 0.01, 5.0, 0.8⟩
  let sm := ⟨"k_smooth", 0.0, 1.0, 0.2⟩
  .union sm (.sphere r) (.box w h d)

end SymforgeIr
