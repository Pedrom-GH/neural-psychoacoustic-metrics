import torch

from psychonet.model import MetricNet


def test_forward_shape_and_time_invariance_of_length():
    model = MetricNet(n_mels=128).eval()
    with torch.no_grad():
        assert model(torch.randn(4, 1, 128, 18)).shape == (4, 2)
        assert model(torch.randn(2, 1, 128, 40)).shape == (2, 2)


def test_overfits_single_batch():
    torch.manual_seed(0)
    model = MetricNet(n_mels=64, channels=(8, 16), hidden=32, dropout=0.0)
    x, y = torch.randn(16, 1, 64, 10), torch.randn(16, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
