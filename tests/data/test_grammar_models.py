"""Automated unit and invariant tests for PCFG and PHOG probabilistic grammar models."""

from __future__ import annotations
import math
import pytest

from symforge.core.ast import (
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FMNode,
    FreeParam,
    MixNode,
    RotateNode,
    SineNode,
    SphereNode,
    TranslateNode,
    UnionNode,
)
from symforge.core.pcfg import PCFGTaskGenerator
from symforge.data.grammar import PCFGModel, PHOGModel
from symforge.data.harvester import ProductionTuple, ProductionTupleHarvester


# ============================================================================
# Test Fixtures & Synthetic Correlated Datasets
# ============================================================================


def make_correlated_cad_dataset() -> list[ProductionTuple]:
    """Generate a CAD dataset with strong structural/contextual correlations.

    Rule pattern:
    - DifferenceNode always has Box as left child (a), and Cylinder as right child (b) [subtractive drill].
    - UnionNode always has Sphere as left child (a), and Box as right child (b).
    Context strongly dictates the child production, creating higher mutual information for PHOG.
    """
    harvester = ProductionTupleHarvester()
    nodes = []

    for i in range(25):
        # Difference with Translate(Box) and Cylinder
        w = FreeParam(f"w_{i}", 1.0, 10.0, 5.0)
        h = FreeParam(f"h_{i}", 1.0, 10.0, 5.0)
        d = FreeParam(f"d_{i}", 1.0, 10.0, 5.0)
        box = BoxNode(width=w, height=h, depth=d)

        r = FreeParam(f"r_{i}", 0.5, 5.0, 2.0)
        cyl_h = FreeParam(f"ch_{i}", 1.0, 10.0, 6.0)
        cylinder = CylinderNode(radius=r, height=cyl_h)

        diff = DifferenceNode(a=box, b=cylinder)
        nodes.append(diff)

        # Union with Sphere and Box
        sr = FreeParam(f"sr_{i}", 0.5, 5.0, 3.0)
        sphere = SphereNode(radius=sr)
        union = UnionNode(smoothness=FreeParam(f"s_{i}", 0.0, 1.0, 0.0), a=sphere, b=box)
        nodes.append(union)

    return harvester.harvest_batch(nodes)


# ============================================================================
# PCFGModel Invariants
# ============================================================================


def test_pcfg_model_fit_and_distribution_sums_to_one():
    """Verify PCFGModel probabilities sum to 1.0 +/- 1e-5 across all nonterminals."""
    dataset = make_correlated_cad_dataset()
    pcfg = PCFGModel(alpha_prior=1.0)
    pcfg.fit(dataset)

    assert "SDF" in pcfg.known_nonterminals
    probs = pcfg.predict_probabilities("SDF")
    assert len(probs) > 0
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(p > 0.0 for p in probs.values())


def test_pcfg_model_laplace_dirichlet_smoothing():
    """Verify alpha_prior smoothing prevents zero probabilities and shifts marginals toward uniform."""
    t1 = ProductionTuple(
        nonterminal="SDF",
        production="SphereNode",
        production_id="sdf_sphere",
        depth_from_root=0,
        depth_to_leaves=0,
        parent_context=["Root"],
        grandparent_context=None,
        left_sibling_context=None,
        ancestor_chain=["Root"],
        expected_type="SDF",
        num_free_params=1,
        free_params=[],
    )
    t2 = ProductionTuple(
        nonterminal="SDF",
        production="BoxNode",
        production_id="sdf_box",
        depth_from_root=0,
        depth_to_leaves=0,
        parent_context=["Root"],
        grandparent_context=None,
        left_sibling_context=None,
        ancestor_chain=["Root"],
        expected_type="SDF",
        num_free_params=1,
        free_params=[],
    )
    # Fit on corpus where Sphere is observed 10x and Box is observed 1x
    corpus = [t1] * 10 + [t2] * 1

    # With low smoothing (0.1), sphere probability remains close to empirical 10/11 ~ 0.909
    pcfg_smooth01 = PCFGModel(alpha_prior=0.1)
    pcfg_smooth01.fit(corpus)
    probs01 = pcfg_smooth01.predict_probabilities("SDF")

    # With high smoothing (10.0), sphere probability is pulled towards uniform 0.5
    pcfg_smooth10 = PCFGModel(alpha_prior=10.0)
    pcfg_smooth10.fit(corpus)
    probs10 = pcfg_smooth10.predict_probabilities("SDF")

    assert probs01["sdf_sphere"] > probs10["sdf_sphere"]
    assert probs01["sdf_box"] < probs10["sdf_box"]
    assert sum(probs01.values()) == pytest.approx(1.0, abs=1e-5)
    assert sum(probs10.values()) == pytest.approx(1.0, abs=1e-5)


def test_pcfg_model_log_likelihood_and_perplexity():
    """Verify log_likelihood and perplexity calculations."""
    dataset = make_correlated_cad_dataset()
    pcfg = PCFGModel(alpha_prior=0.5)
    pcfg.fit(dataset)

    ll = pcfg.log_likelihood(dataset)
    ppl = pcfg.perplexity(dataset)

    assert math.isfinite(ll)
    assert ll < 0.0
    assert math.isfinite(ppl)
    assert ppl >= 1.0

    # Theoretical relation: ppl = 2^(- (1/N) * ll)
    n = len(dataset)
    expected_ppl = 2.0 ** (-ll / n)
    assert ppl == pytest.approx(expected_ppl, rel=1e-4)


# ============================================================================
# PHOGModel Invariants
# ============================================================================


def test_phog_model_fit_and_distribution_sums_to_one():
    """Verify PHOGModel probabilities sum to 1.0 +/- 1e-5 across conditioned contexts."""
    dataset = make_correlated_cad_dataset()
    phog = PHOGModel(lambdas=(0.6, 0.25, 0.15), alpha_prior=1.0)
    phog.fit(dataset)

    # Test context 1: Child of Difference
    ctx_diff = {"parent": "Difference", "grandparent": "Root", "left_sibling": None}
    probs_diff = phog.predict_probabilities("SDF", context=ctx_diff)
    assert sum(probs_diff.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(p > 0.0 for p in probs_diff.values())

    # Test context 2: Child of Difference with Box left sibling
    ctx_diff_box = {"parent": "Difference", "grandparent": "Root", "left_sibling": "Box"}
    probs_diff_box = phog.predict_probabilities("SDF", context=ctx_diff_box)
    assert sum(probs_diff_box.values()) == pytest.approx(1.0, abs=1e-5)

    # Under Difference with Box sibling, Cylinder should have high conditional probability
    assert probs_diff_box["sdf_cylinder"] > probs_diff_box["sdf_sphere"]


def test_phog_jelinek_mercer_weight_validation():
    """Verify lambda weights sum to 1.0 or raise ValueError."""
    with pytest.raises(ValueError):
        PHOGModel(lambdas=(0.5, 0.5, 0.5))  # sum = 1.5 != 1.0


# ============================================================================
# Mathematical Invariant: PHOG vs PCFG Comparison
# ============================================================================


def test_phog_achieves_lower_perplexity_than_pcfg_on_correlated_corpus():
    """Verify mathematical invariant: On structured datasets with contextual correlations,

    PHOGModel achieves strictly higher log-likelihood and strictly lower perplexity than PCFGModel.
    """
    dataset = make_correlated_cad_dataset()
    # 80/20 train/eval split
    split_idx = int(len(dataset) * 0.8)
    train_tuples = dataset[:split_idx]
    eval_tuples = dataset[split_idx:]

    pcfg = PCFGModel(alpha_prior=0.1)
    pcfg.fit(train_tuples)

    phog = PHOGModel(lambdas=(0.7, 0.2, 0.1), alpha_prior=0.1)
    phog.fit(train_tuples)

    pcfg_ll = pcfg.log_likelihood(eval_tuples)
    phog_ll = phog.log_likelihood(eval_tuples)

    pcfg_ppl = pcfg.perplexity(eval_tuples)
    phog_ppl = phog.perplexity(eval_tuples)

    # PHOG has higher log-likelihood (closer to 0) and lower perplexity
    assert phog_ll > pcfg_ll, f"PHOG LL {phog_ll} should exceed PCFG LL {pcfg_ll}"
    assert phog_ppl < pcfg_ppl, f"PHOG PPL {phog_ppl} should be lower than PCFG PPL {pcfg_ppl}"


# ============================================================================
# Smoothing & Edge Cases
# ============================================================================


def test_smoothing_unseen_contexts_and_productions():
    """Verify unseen context and productions do not cause zero-division or -inf log likelihood."""
    dataset = make_correlated_cad_dataset()
    pcfg = PCFGModel(alpha_prior=1.0)
    pcfg.fit(dataset)

    phog = PHOGModel(lambdas=(0.5, 0.3, 0.2), alpha_prior=1.0)
    phog.fit(dataset)

    # Completely unseen context
    alien_context = {
        "parent": "AlienParentNode",
        "grandparent": "CosmicGrandparent",
        "left_sibling": "UnknownSibling",
    }
    phog_probs = phog.predict_probabilities("SDF", context=alien_context)
    assert sum(phog_probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(p > 0.0 for p in phog_probs.values())

    # Tuple with unseen production
    unseen_tuple = ProductionTuple(
        nonterminal="SDF",
        production="HyperSphereNode(dim: FreeParam)",
        production_id="sdf_hypersphere",
        depth_from_root=3,
        depth_to_leaves=0,
        parent_context=["AlienParentNode"],
        grandparent_context="CosmicGrandparent",
        left_sibling_context="UnknownSibling",
        ancestor_chain=["Root", "AlienParentNode"],
        expected_type="SDF",
        num_free_params=1,
        free_params=[],
    )

    # Evaluating unseen tuple must be finite (smoothed)
    pcfg_ll = pcfg.log_likelihood([unseen_tuple])
    phog_ll = phog.log_likelihood([unseen_tuple])

    assert math.isfinite(pcfg_ll)
    assert math.isfinite(phog_ll)
    assert not math.isnan(pcfg_ll)
    assert not math.isnan(phog_ll)

    pcfg_ppl = pcfg.perplexity([unseen_tuple])
    phog_ppl = phog.perplexity([unseen_tuple])
    assert math.isfinite(pcfg_ppl)
    assert math.isfinite(phog_ppl)


def test_empty_evaluation_tuples():
    """Verify edge cases for empty evaluation lists."""
    pcfg = PCFGModel()
    phog = PHOGModel()

    assert pcfg.log_likelihood([]) == 0.0
    assert phog.log_likelihood([]) == 0.0
    assert pcfg.perplexity([]) == 1.0
    assert phog.perplexity([]) == 1.0


def test_invalid_alpha_prior_and_unseen_nonterminal():
    """Verify validation on alpha_prior and handling of unseen nonterminals."""
    with pytest.raises(ValueError):
        PCFGModel(alpha_prior=0.0)
    with pytest.raises(ValueError):
        PCFGModel(alpha_prior=-1.5)

    pcfg = PCFGModel()
    assert pcfg.predict_probabilities("UnknownNT") == {}
    phog = PHOGModel()
    assert phog.predict_probabilities("UnknownNT") == {}


def test_audio_dexed_correlated_corpus_comparison():
    """Verify PHOG achieves lower perplexity than PCFG on audio synthesis topologies."""
    from symforge.core.ast import ADSRNode, FMNode, MixNode, SineNode

    harvester = ProductionTupleHarvester()
    nodes = []

    for i in range(20):
        # Top-level Mix with 2 voices
        # Voice 1: ADSR wrapping FM
        f_c1 = FreeParam(f"fc1_{i}", 100, 1000, 440)
        f_m1 = FreeParam(f"fm1_{i}", 10, 500, 220)
        idx1 = FreeParam(f"idx1_{i}", 0, 10, 2.0)
        fm1 = FMNode(carrier=f_c1, mod_freq=f_m1, mod_index=idx1)
        adsr1 = ADSRNode(
            attack=FreeParam(f"a1_{i}", 0.01, 1.0, 0.05),
            decay=FreeParam(f"d1_{i}", 0.01, 1.0, 0.2),
            sustain=FreeParam(f"s1_{i}", 0.0, 1.0, 0.7),
            release=FreeParam(f"r1_{i}", 0.01, 1.0, 0.3),
            gate=fm1,
        )

        # Voice 2: ADSR wrapping pure Sine
        f_s2 = FreeParam(f"fs2_{i}", 50, 500, 110)
        p_s2 = FreeParam(f"ps2_{i}", 0, 6.28, 0.0)
        sine2 = SineNode(freq=f_s2, phase=p_s2)
        adsr2 = ADSRNode(
            attack=FreeParam(f"a2_{i}", 0.01, 1.0, 0.1),
            decay=FreeParam(f"d2_{i}", 0.01, 1.0, 0.4),
            sustain=FreeParam(f"s2_{i}", 0.0, 1.0, 0.5),
            release=FreeParam(f"r2_{i}", 0.01, 1.0, 0.5),
            gate=sine2,
        )

        mix = MixNode(ratio=FreeParam(f"mix_{i}", 0, 1, 0.5), input1=adsr1, input2=adsr2)
        nodes.append(mix)

    dataset = harvester.harvest_batch(nodes)
    split_idx = int(len(dataset) * 0.75)
    train_tuples = dataset[:split_idx]
    eval_tuples = dataset[split_idx:]

    pcfg = PCFGModel(alpha_prior=0.1)
    pcfg.fit(train_tuples)

    phog = PHOGModel(lambdas=(0.6, 0.25, 0.15), alpha_prior=0.1)
    phog.fit(train_tuples)

    pcfg_ppl = pcfg.perplexity(eval_tuples)
    phog_ppl = phog.perplexity(eval_tuples)

    assert phog_ppl < pcfg_ppl, f"PHOG PPL {phog_ppl} should be strictly lower than PCFG PPL {pcfg_ppl}"

