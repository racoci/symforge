import SymforgeIr.Types

namespace SymforgeIr

/-- Emit Python/PyTorch differentiable expression string from an AudioNode -/
def AudioNode.toPyTorchExpr (node : AudioNode) : String :=
  match node with
  | .sine f p => s!"torch.sin(2.0 * 3.141592653589793 * theta['{f.id}'] * t + theta['{p.id}'])"
  | .saw f => s!"(2.0 * (theta['{f.id}'] * t - torch.floor(theta['{f.id}'] * t + 0.5)))"
  | .square f _ => s!"torch.tanh(10.0 * torch.sin(2.0 * 3.141592653589793 * theta['{f.id}'] * t))"
  | .noise _ => s!"torch.randn_like(t)"
  | .adsr a d s r g => s!"(adsr_envelope(t, theta['{a.id}'], theta['{d.id}'], theta['{s.id}'], theta['{r.id}']) * ({g.toPyTorchExpr}))"
  | .lowpass c res i => s!"differentiable_lowpass({i.toPyTorchExpr}, theta['{c.id}'], theta['{res.id}'])"
  | .gain g i => s!"(theta['{g.id}'] * ({i.toPyTorchExpr}))"
  | .mix r a b => s!"torch.lerp({a.toPyTorchExpr}, {b.toPyTorchExpr}, theta['{r.id}'])"
  | .fm c mf mi => s!"torch.sin(2.0 * 3.141592653589793 * theta['{c.id}'] * t + theta['{mi.id}'] * torch.sin(2.0 * 3.141592653589793 * theta['{mf.id}'] * t))"

/-- Emit Python/PyTorch differentiable expression string from an SDFNode -/
def SDFNode.toPyTorchExpr (node : SDFNode) : String :=
  match node with
  | .sphere r => s!"(torch.norm(p, dim=-1) - theta['{r.id}'])"
  | .box w h d => s!"sdf_box(p, theta['{w.id}'], theta['{h.id}'], theta['{d.id}'])"
  | .cylinder r h => s!"sdf_cylinder(p, theta['{r.id}'], theta['{h.id}'])"
  | .torus r1 r2 => s!"sdf_torus(p, theta['{r1.id}'], theta['{r2.id}'])"
  | .translate dx dy dz c => s!"sdf_translate(p, theta['{dx.id}'], theta['{dy.id}'], theta['{dz.id}'], {c.toPyTorchExpr})"
  | .rotate rx ry rz c => s!"sdf_rotate(p, theta['{rx.id}'], theta['{ry.id}'], theta['{rz.id}'], {c.toPyTorchExpr})"
  | .scale s c => s!"(theta['{s.id}'] * ({c.toPyTorchExpr}))"
  | .union sm a b => s!"smooth_min({a.toPyTorchExpr}, {b.toPyTorchExpr}, theta['{sm.id}'])"
  | .intersection a b => s!"torch.maximum({a.toPyTorchExpr}, {b.toPyTorchExpr})"
  | .difference a b => s!"torch.maximum({a.toPyTorchExpr}, -({b.toPyTorchExpr}))"

end SymforgeIr
