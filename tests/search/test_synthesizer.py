import pytest
import torch

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    SDFNode,
    AudioHoleNode,
    SDFHoleNode,
    SineNode,
    SawNode,
    SphereNode,
    BoxNode,
    TranslateNode,
    FreeParam,
    has_holes,
)
from symforge.eval.cost import HardwareCostModel
from symforge.eval.loss import compute_infonce_loss
from symforge.search.synthesizer import (
    SearchConfig,
    TypedGrammarExpander,
    TypeGuidedBeamSynthesizer,
    AStarSynthesizer,
    ReferenceASTScorer,
    CustomLossScorer,
)


def test_typed_hole_expansion_type_safety():
    expander = TypedGrammarExpander()

    # Invariant: AudioHole expansion produces ONLY AudioNode instances
    audio_hole = AudioHoleNode("test_audio")
    audio_expansions = expander.get_expansions(audio_hole)
    assert len(audio_expansions) > 0
    for node in audio_expansions:
        assert isinstance(node, AudioNode), f"Expected AudioNode, got {type(node)}"
        assert not isinstance(node, SDFNode), f"Audio hole expanded to SDFNode: {type(node)}"

    # Invariant: SDFHole expansion produces ONLY SDFNode instances
    sdf_hole = SDFHoleNode("test_sdf")
    sdf_expansions = expander.get_expansions(sdf_hole)
    assert len(sdf_expansions) > 0
    for node in sdf_expansions:
        assert isinstance(node, SDFNode), f"Expected SDFNode, got {type(node)}"
        assert not isinstance(node, AudioNode), f"SDF hole expanded to AudioNode: {type(node)}"


def test_beam_search_prunes_cost_budget():
    # Set tight cost budget that allows simple primitive (e.g. SawNode cost = 3.5)
    # but strictly prunes composite operations (e.g. LowPassNode cost >= 12.0)
    cost_model = HardwareCostModel()
    cfg = SearchConfig(max_cost_budget=4.0, beam_width=5, max_expansions=50)

    synthesizer = TypeGuidedBeamSynthesizer(cost_model=cost_model)
    target = SawNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0))
    scorer = ReferenceASTScorer(target=target, domain="audio")

    result = synthesizer.synthesize(
        initial_hole=AudioHoleNode("root"),
        scorer=scorer,
        config=cfg,
    )

    # Invariant: Solution found within budget
    assert result.success is True
    assert result.best_cost <= cfg.max_cost_budget
    assert isinstance(result.best_program, SawNode)


def test_beam_synthesizer_solves_baseline_audio():
    target = SawNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0))
    scorer = ReferenceASTScorer(target=target, domain="audio")

    synthesizer = TypeGuidedBeamSynthesizer()
    cfg = SearchConfig(beam_width=10, max_expansions=200, tolerance=1e-4)

    result = synthesizer.synthesize(
        initial_hole=AudioHoleNode("root"),
        scorer=scorer,
        config=cfg,
    )

    # Gate: Synthesizer solves baseline target problem within 200 expansion steps
    assert result.success is True
    assert result.total_expansions < 200
    assert result.best_program is not None
    assert not has_holes(result.best_program)
    assert isinstance(result.best_program, SawNode)
    assert result.best_score < 1e-4


def test_beam_synthesizer_solves_baseline_sdf():
    target = SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0))
    scorer = ReferenceASTScorer(target=target, domain="sdf")

    synthesizer = TypeGuidedBeamSynthesizer()
    cfg = SearchConfig(beam_width=10, max_expansions=200, tolerance=1e-4)

    result = synthesizer.synthesize(
        initial_hole=SDFHoleNode("root"),
        scorer=scorer,
        config=cfg,
    )

    # Gate: Synthesizer solves baseline SDF target problem within 200 expansion steps
    assert result.success is True
    assert result.total_expansions < 200
    assert result.best_program is not None
    assert not has_holes(result.best_program)
    assert isinstance(result.best_program, SphereNode)
    assert result.best_score < 1e-4


def test_astar_synthesizer_solves_composite_sdf():
    sphere = SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0))
    dx = FreeParam("dx", -5.0, 5.0, 1.0)
    dy = FreeParam("dy", -5.0, 5.0, 0.0)
    dz = FreeParam("dz", -5.0, 5.0, 0.0)
    target = TranslateNode(dx=dx, dy=dy, dz=dz, child=sphere)

    scorer = ReferenceASTScorer(target=target, domain="sdf")
    synthesizer = AStarSynthesizer()
    cfg = SearchConfig(max_expansions=200, tolerance=1e-4)

    result = synthesizer.synthesize(
        initial_hole=SDFHoleNode("root"),
        scorer=scorer,
        config=cfg,
    )

    # Gate: Discovers composite translated shape within 200 expansion steps
    assert result.success is True
    assert result.total_expansions < 200
    assert result.best_program is not None
    assert isinstance(result.best_program, TranslateNode)
    assert isinstance(result.best_program.child, SphereNode)


def test_astar_synthesizer_solves_composite_audio():
    from symforge.core.ast import LowPassNode

    f0 = FreeParam("freq", 20.0, 5000.0, 440.0)
    saw = SawNode(freq=f0)
    fc = FreeParam("cutoff", 100.0, 10000.0, 1000.0)
    res = FreeParam("res", 0.1, 10.0, 0.707)
    target = LowPassNode(cutoff=fc, resonance=res, input_node=saw)

    scorer = ReferenceASTScorer(target=target, domain="audio")
    synthesizer = AStarSynthesizer()
    cfg = SearchConfig(max_expansions=200, tolerance=1e-4)

    result = synthesizer.synthesize(
        initial_hole=AudioHoleNode("root"),
        scorer=scorer,
        config=cfg,
    )

    assert result.success is True
    assert result.total_expansions < 200
    assert result.best_program is not None
    assert isinstance(result.best_program, LowPassNode)
    assert isinstance(result.best_program.input_node, SawNode)


def test_synthesizer_with_contrastive_loss_scorer():
    # Evaluate synthesizer using custom InfoNCE contrastive perceptual loss
    target = SawNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0))
    ref_scorer = ReferenceASTScorer(target=target, domain="audio")
    pos_latent = ref_scorer.target_output.unsqueeze(0)
    # Orthogonal distractors
    neg_latent = torch.randn(3, pos_latent.shape[1])

    def contrastive_eval(prog: ASTNode) -> float:
        if has_holes(prog):
            return 2.0
        mse = ref_scorer.score(prog)
        if mse < 1e-4:
            return 0.0
        # Compute InfoNCE
        from symforge.core.transpiler import transpile_ast_to_module
        mod = transpile_ast_to_module(prog)
        with torch.no_grad():
            out = mod(ref_scorer.eval_inputs).unsqueeze(0)
        return compute_infonce_loss(out, pos_latent, neg_latent, temperature=0.1).item()

    synthesizer = TypeGuidedBeamSynthesizer()
    scorer = CustomLossScorer(contrastive_eval)
    cfg = SearchConfig(beam_width=10, max_expansions=100, tolerance=1e-3)

    result = synthesizer.synthesize(
        initial_hole=AudioHoleNode("root"),
        scorer=scorer,
        config=cfg,
    )

    assert result.success is True
    assert isinstance(result.best_program, SawNode)
