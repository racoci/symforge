"""Probabilistic grammar models for symbolic program synthesis (PCFG & PHOG)."""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from symforge.data.harvester import ProductionTuple

ContextKey = Tuple[str, Optional[str], Optional[str]]


def extract_context(
    context: Union[Dict[str, Any], ProductionTuple, Tuple[str, Optional[str], Optional[str]], None]
) -> ContextKey:
    """Normalize context input into a canonical (parent, grandparent, left_sibling) tuple."""
    if context is None:
        return ("Root", None, None)

    if isinstance(context, tuple) and len(context) == 3:
        return (context[0] or "Root", context[1], context[2])

    if isinstance(context, ProductionTuple):
        parent = context.parent_context[0] if context.parent_context else "Root"
        return (parent, context.grandparent_context, context.left_sibling_context)

    parent = context.get("parent") or context.get("parent_context") or "Root"
    if isinstance(parent, list):
        parent = parent[0] if parent else "Root"
    grandparent = context.get("grandparent") or context.get("grandparent_context")
    sibling = context.get("left_sibling") or context.get("left_sibling_context")
    return (str(parent), str(grandparent) if grandparent else None, str(sibling) if sibling else None)


class PCFGModel:
    """Probabilistic Context-Free Grammar with Laplace/Dirichlet smoothing."""

    def __init__(self, alpha_prior: float = 1.0) -> None:
        if alpha_prior <= 0.0:
            raise ValueError(f"alpha_prior must be > 0.0, got {alpha_prior}")
        self.alpha_prior = float(alpha_prior)
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.totals: Dict[str, int] = defaultdict(int)
        self.productions: Dict[str, Set[str]] = defaultdict(set)

    @property
    def known_nonterminals(self) -> Set[str]:
        """Return the set of known nonterminals."""
        return set(self.productions.keys())

    def fit(self, tuples: List[ProductionTuple]) -> None:
        """Train marginal production counts on a corpus of ProductionTuples."""
        for t in tuples:
            nt = t.nonterminal
            pid = t.production_id
            self.counts[nt][pid] += 1
            self.totals[nt] += 1
            self.productions[nt].add(pid)

    def predict_probabilities(
        self, nonterminal: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Return normalized marginal probability distribution over productions for nonterminal."""
        prods = self.productions.get(nonterminal)
        if not prods:
            return {}

        total_count = self.totals[nonterminal]
        num_rules = len(prods)
        denom = total_count + num_rules * self.alpha_prior

        raw_probs = {
            pid: (self.counts[nonterminal][pid] + self.alpha_prior) / denom
            for pid in prods
        }
        sum_p = sum(raw_probs.values())
        return {pid: p / sum_p for pid, p in raw_probs.items()}

    def get_probability(self, nonterminal: str, production_id: str) -> float:
        """Return smoothed probability for a single production, supporting unseen rules."""
        prods = self.productions.get(nonterminal)
        if not prods:
            return 1e-12

        total_count = self.totals[nonterminal]
        num_rules = len(prods)
        # Smoothing denominator includes one virtual slot for out-of-vocabulary productions
        denom = total_count + (num_rules + 1) * self.alpha_prior

        count = self.counts[nonterminal].get(production_id, 0)
        return (count + self.alpha_prior) / denom

    def log_likelihood(self, tuples: List[ProductionTuple]) -> float:
        """Compute total log2-likelihood over evaluation tuples."""
        if not tuples:
            return 0.0
        total_ll = 0.0
        for t in tuples:
            prob = self.get_probability(t.nonterminal, t.production_id)
            total_ll += math.log2(max(prob, 1e-12))
        return total_ll

    def perplexity(self, tuples: List[ProductionTuple]) -> float:
        """Compute per-token perplexity: 2^(-(1/N) * sum log2 P(r_i))."""
        if not tuples:
            return 1.0
        ll = self.log_likelihood(tuples)
        return 2.0 ** (-ll / len(tuples))


class PHOGModel:
    """Probabilistic Higher-Order Grammar with hierarchical Jelinek-Mercer smoothing."""

    def __init__(
        self,
        lambdas: Tuple[float, float, float] = (0.6, 0.25, 0.15),
        alpha_prior: float = 1.0,
    ) -> None:
        if not math.isclose(sum(lambdas), 1.0, rel_tol=1e-5):
            raise ValueError(f"Lambdas must sum to 1.0, got sum={sum(lambdas)}")
        self.lambdas = lambdas
        self.pcfg = PCFGModel(alpha_prior=alpha_prior)
        self.context_counts: Dict[str, Dict[ContextKey, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.context_totals: Dict[str, Dict[ContextKey, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.parent_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.parent_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, tuples: List[ProductionTuple]) -> None:
        """Train PHOG higher-order context and backoff marginals."""
        self.pcfg.fit(tuples)
        for t in tuples:
            nt = t.nonterminal
            pid = t.production_id
            ctx = extract_context(t)
            parent = ctx[0]

            self.context_counts[nt][ctx][pid] += 1
            self.context_totals[nt][ctx] += 1

            self.parent_counts[nt][parent][pid] += 1
            self.parent_totals[nt][parent] += 1

    def _calculate_weights(
        self, nt: str, ctx: ContextKey, parent: str
    ) -> Tuple[float, float, float]:
        l1, l2, l3 = self.lambdas
        w1 = l1 if self.context_totals[nt].get(ctx, 0) > 0 else 0.0
        w2 = l2 if self.parent_totals[nt].get(parent, 0) > 0 else 0.0
        w3 = l3
        total_w = w1 + w2 + w3
        if total_w <= 0.0:
            return (0.0, 0.0, 1.0)
        return (w1 / total_w, w2 / total_w, w3 / total_w)

    def _conditional_mle(
        self, counts: Dict[str, int], total: int, pid: str
    ) -> float:
        if total <= 0:
            return 0.0
        return counts.get(pid, 0) / total

    def get_probability(
        self,
        nonterminal: str,
        production_id: str,
        context: Union[Dict[str, Any], ProductionTuple, ContextKey, None] = None,
    ) -> float:
        """Return Jelinek-Mercer smoothed probability conditioned on structural context."""
        ctx = extract_context(context)
        parent = ctx[0]
        w1, w2, w3 = self._calculate_weights(nonterminal, ctx, parent)

        p1 = self._conditional_mle(
            self.context_counts[nonterminal][ctx],
            self.context_totals[nonterminal].get(ctx, 0),
            production_id,
        )
        p2 = self._conditional_mle(
            self.parent_counts[nonterminal][parent],
            self.parent_totals[nonterminal].get(parent, 0),
            production_id,
        )
        p3 = self.pcfg.get_probability(nonterminal, production_id)

        return w1 * p1 + w2 * p2 + w3 * p3

    def predict_probabilities(
        self,
        nonterminal: str,
        context: Union[Dict[str, Any], ProductionTuple, ContextKey, None] = None,
    ) -> Dict[str, float]:
        """Return normalized probability distribution over all known productions for nonterminal."""
        prods = self.pcfg.productions.get(nonterminal)
        if not prods:
            return {}

        raw_probs = {
            pid: self.get_probability(nonterminal, pid, context)
            for pid in prods
        }
        sum_p = sum(raw_probs.values())
        if sum_p <= 0.0:
            return {pid: 1.0 / len(prods) for pid in prods}
        return {pid: p / sum_p for pid, p in raw_probs.items()}

    def log_likelihood(self, tuples: List[ProductionTuple]) -> float:
        """Compute total log2-likelihood over evaluation tuples."""
        if not tuples:
            return 0.0
        total_ll = 0.0
        for t in tuples:
            prob = self.get_probability(t.nonterminal, t.production_id, t)
            total_ll += math.log2(max(prob, 1e-12))
        return total_ll

    def perplexity(self, tuples: List[ProductionTuple]) -> float:
        """Compute per-token perplexity: 2^(-(1/N) * sum log2 P(r_i | c_i))."""
        if not tuples:
            return 1.0
        ll = self.log_likelihood(tuples)
        return 2.0 ** (-ll / len(tuples))
