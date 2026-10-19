import math
import pytest
import torch

from symforge.core.ast import (
    FreeParam,
    SineNode,
    SawNode,
    LowPassNode,
    FMNode,
    SphereNode,
    BoxNode,
    UnionNode,
)
from symforge.core.transpiler import (
    DifferentiableAudioModule,
    DifferentiableSDFModule,
    transpile_ast_to_module,
)
from symforge.core.bridge import LeanBridge


def test_differentiable_audio_module_forward_and_grad():
    f0 = FreeParam("f0", 20.0, 1000.0, 440.0)
    phase = FreeParam("p0", 0.0, 6.28, 0.0)
    sine = SineNode(freq=f0, phase=phase)

    module = DifferentiableAudioModule(sine)
    assert isinstance(module.theta, torch.nn.Parameter)
    assert module.theta.shape[0] == 2

    # Forward pass over 16000 samples
    t = torch.linspace(0.0, 1.0, 1000, requires_grad=False)
    y = module(t)
    assert y.shape == (1000,)

    # Analytical reference: sin(2 * pi * 440 * t)
    expected = torch.sin(2.0 * math.pi * 440.0 * t)
    assert torch.allclose(y, expected, atol=1e-5)

    # Autograd backward pass
    loss = (y - 0.5).pow(2).mean()
    loss.backward()
    assert module.theta.grad is not None
    assert torch.norm(module.theta.grad) > 0.0


def test_differentiable_audio_fm_synthesis():
    c = FreeParam("fc", 100.0, 1000.0, 220.0)
    mf = FreeParam("fm", 10.0, 500.0, 110.0)
    mi = FreeParam("mi", 0.1, 10.0, 2.0)
    fm_node = FMNode(carrier=c, mod_freq=mf, mod_index=mi)

    module = DifferentiableAudioModule(fm_node)
    t = torch.linspace(0.0, 0.1, 500)
    y = module(t)

    loss = y.sum()
    loss.backward()
    assert module.theta.grad is not None
    # All 3 parameters receive gradient
    assert (module.theta.grad.abs() > 0.0).all()


def test_differentiable_sdf_module_forward_and_grad():
    r = FreeParam("r", 0.1, 5.0, 1.0)
    sphere = SphereNode(radius=r)

    module = DifferentiableSDFModule(sphere)
    assert isinstance(module.theta, torch.nn.Parameter)
    assert module.theta.shape[0] == 1

    # Points on sphere surface: dist should be 0.0
    p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    d = module(p)
    assert torch.allclose(d, torch.zeros(3), atol=1e-5)

    # Points outside sphere: (2, 0, 0) -> dist = 1.0
    p_out = torch.tensor([[2.0, 0.0, 0.0]])
    assert torch.allclose(module(p_out), torch.tensor([1.0]), atol=1e-5)

    # Autograd test
    loss = module(p_out).pow(2)
    loss.backward()
    assert module.theta.grad is not None
    assert module.theta.grad.item() != 0.0


def test_differentiable_sdf_smooth_union():
    r = FreeParam("r", 0.1, 5.0, 1.0)
    sphere = SphereNode(radius=r)
    w = FreeParam("w", 0.1, 5.0, 0.8)
    h = FreeParam("h", 0.1, 5.0, 0.8)
    d = FreeParam("d", 0.1, 5.0, 0.8)
    box = BoxNode(width=w, height=h, depth=d)
    sm = FreeParam("k", 0.01, 1.0, 0.2)
    union_node = UnionNode(smoothness=sm, a=sphere, b=box)

    module = DifferentiableSDFModule(union_node)
    assert module.theta.shape[0] == 5

    p = torch.randn(64, 3)
    out = module(p)
    assert out.shape == (64,)

    loss = out.pow(2).mean()
    loss.backward()
    assert module.theta.grad is not None
    assert torch.norm(module.theta.grad) > 0.0


def test_transpile_via_lean_bridge():
    bridge = LeanBridge()
    f0 = FreeParam("f0", 20.0, 1000.0, 440.0)
    saw = SawNode(freq=f0)

    # Verify Lean transpilation command
    proc = bridge._execute_cmd(
        ["transpile-audio"],
        input_data='{"saw":{"freq":{"defaultVal":440,"id":"f0","maxVal":1000,"minVal":20}}}',
    )
    assert "torch.floor" in proc.stdout
