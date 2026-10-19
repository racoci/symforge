import pytest
import torch
from symforge.eval.loss import (
    ContrastivePerceptualLoss,
    compute_infonce_loss,
    CompositeSynthesisObjective,
)


def test_infonce_loss_exact_match():
    # When query perfectly matches positive target and is orthogonal to distractors
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # 2 orthogonal negative distractors
    negs = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    loss = compute_infonce_loss(q, pos, negs, temperature=0.05)
    # With tau=0.05, sim(q, pos)=1.0 -> exp(20) ~ 4.8e8, negs exp(0) = 1.
    # Loss should be virtually zero (< 1e-4)
    assert loss.item() < 1e-4


def test_infonce_loss_monotonicity():
    # As similarity with positive target increases, loss must strictly decrease
    q = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    negs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    target_far = torch.tensor([[0.2, 0.8, 0.0]])
    target_mid = torch.tensor([[0.7, 0.3, 0.0]])
    target_close = torch.tensor([[0.99, 0.01, 0.0]])

    loss_far = compute_infonce_loss(q, target_far, negs, temperature=0.1)
    loss_mid = compute_infonce_loss(q, target_mid, negs, temperature=0.1)
    loss_close = compute_infonce_loss(q, target_close, negs, temperature=0.1)

    assert loss_far.item() > loss_mid.item() > loss_close.item()


def test_contrastive_perceptual_loss_gradient():
    criterion = ContrastivePerceptualLoss(temperature=0.1)

    q = torch.randn(1, 16, requires_grad=True)
    pos = torch.randn(1, 16)
    negs = torch.randn(4, 16)

    loss = criterion(q, pos, negs)
    loss.backward()

    assert q.grad is not None
    assert torch.norm(q.grad) > 0.0


def test_infonce_batch_dimension():
    # Batch size B = 2, Dim = 8, Negatives K = 3
    q = torch.randn(2, 8)
    pos = torch.randn(2, 8)
    negs = torch.randn(2, 3, 8)

    loss = compute_infonce_loss(q, pos, negs, temperature=0.1)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_infonce_invalid_temperature():
    q = torch.randn(1, 4)
    pos = torch.randn(1, 4)
    negs = torch.randn(2, 4)

    with pytest.raises(ValueError, match="strictly positive"):
        compute_infonce_loss(q, pos, negs, temperature=0.0)


def test_composite_synthesis_objective():
    obj = CompositeSynthesisObjective(
        temperature=0.1,
        lambda_contrast=1.0,
        lambda_rec=0.5,
        lambda_cost=0.02,
    )

    q = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    negs = torch.tensor([[0.0, 1.0, 0.0]])

    rec_loss = torch.tensor(0.5)
    hw_cost = 15.0

    total_loss = obj(q, pos, negs, reconstruction_loss=rec_loss, hardware_cost=hw_cost)
    total_loss.backward()

    assert q.grad is not None
    # total_loss includes lambda_rec * 0.5 (= 0.25) and lambda_cost * 15.0 (= 0.3)
    assert total_loss.item() > 0.55
