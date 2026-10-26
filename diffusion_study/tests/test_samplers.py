"""Check the integrators against an analytic velocity field (data ~ N(m, s^2) in 1-D, noise ~ N(0, 1)).

For x_t = (1-t) x0 + t x1: Var(x_t) = (1-t)^2 + t^2 s^2, Cov(x_t, x1) = t s^2, so
E[x1 | x_t] = m + t s^2 / Var(x_t) * (x_t - t m), and the velocity is v = (E[x1 | x_t] - x_t) / (1 - t).
Every sampler must reproduce mean m and std s of the data from noise (large step counts -> tight tolerance).
"""
import torch

from flowgen.samplers import integrate

M, S = 2.0, 0.5


def vel(x, t, y):
    t = float(t)
    var = (1 - t) ** 2 + (t * S) ** 2
    ex1 = M + t * S * S / var * (x - t * M)
    return (ex1 - x) / (1 - t) if t < 1 - 1e-6 else ex1 - x


def run(method, steps, eta=0.0, n=200_000):
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(n, 1, generator=g)
    return integrate(vel, x0, None, steps, method, eta, g)


def test_ode_samplers_recover_data_distribution():
    for method, steps in [("euler", 400), ("heun", 100), ("midpoint", 100), ("rk4", 50)]:
        x = run(method, steps)
        assert abs(x.mean().item() - M) < 0.02, (method, x.mean().item())
        assert abs(x.std().item() - S) < 0.02, (method, x.std().item())


def test_sde_sampler_recovers_data_distribution():
    x = run("sde", 800, eta=1.0)
    assert abs(x.mean().item() - M) < 0.03 and abs(x.std().item() - S) < 0.03, (x.mean().item(), x.std().item())


def test_sde_with_zero_noise_equals_euler():
    a, b = run("sde", 50, eta=0.0, n=1000), run("euler", 50, n=1000)
    assert torch.allclose(a, b, atol=1e-5)
