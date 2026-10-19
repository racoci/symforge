import pytest
from symforge.core.ast import AudioNode, SDFNode
from symforge.core.pcfg import PCFGTaskGenerator, GrammarRule


def test_pcfg_generator_initialization():
    generator = PCFGTaskGenerator(domain="audio", seed=42)
    assert generator.domain == "audio"
    assert len(generator.rules) > 0


def test_pcfg_sampling_determinism():
    gen1 = PCFGTaskGenerator(domain="audio", seed=123)
    gen2 = PCFGTaskGenerator(domain="audio", seed=123)

    ast1 = gen1.sample(max_depth=4)
    ast2 = gen2.sample(max_depth=4)

    assert ast1 == ast2


def test_pcfg_depth_bounding():
    generator = PCFGTaskGenerator(domain="audio", seed=42)
    for _ in range(50):
        ast = generator.sample(max_depth=3)
        assert ast.tree_depth <= 3


def test_pcfg_audio_domain_validity():
    generator = PCFGTaskGenerator(domain="audio", seed=777)
    for _ in range(20):
        ast = generator.sample(max_depth=4)
        assert isinstance(ast, AudioNode)
        assert ast.size >= 1
        assert len(ast.collect_params()) >= 1


def test_pcfg_sdf_domain_validity():
    generator = PCFGTaskGenerator(domain="sdf", seed=888)
    for _ in range(20):
        ast = generator.sample(max_depth=4)
        assert isinstance(ast, SDFNode)
        assert ast.size >= 1
        assert len(ast.collect_params()) >= 1


def test_pcfg_500_synthetic_derivations_pass_gate():
    # Pass-to-complete gate: Generates 500 valid synthetic derivations without infinite recursive loops or type violations
    generator = PCFGTaskGenerator(domain="audio", seed=2026)
    count = 0
    for _ in range(500):
        ast = generator.sample(max_depth=4)
        assert isinstance(ast, AudioNode)
        assert 1 <= ast.tree_depth <= 4
        count += 1
    assert count == 500
