"""ODE / SDE samplers for a flow-matching velocity model, with classifier-free guidance.

Convention: x_t = (1 - t) * x0 + t * x1, x0 ~ N(0, I) noise, x1 data, target velocity v = x1 - x0, t in [0, 1].
Sampling integrates dx/dt = v(x, t, y) from t = 0 (noise) to t = 1 (data).

Guidance: v = v_uncond + s * (v_cond - v_uncond); s = 1 is the plain conditional model, s = 0 unconditional.
NFE counts velocity-function evaluations; each guided evaluation costs TWO network forward passes (cond + uncond,
batched together), so `network_passes` is reported separately and is what wall-clock follows.
"""

from __future__ import annotations

import torch


class Counter:
    def __init__(self):
        self.nfe = 0
        self.passes = 0


def guided_velocity(net, scale: float, null_label: int, counter: Counter | None = None):
    """-> f(x, t, y) with CFG. `t` is a python float or tensor in [0, 1]."""
    def f(x, t, y):
        tt = torch.full((x.shape[0],), float(t), device=x.device) if not torch.is_tensor(t) else t
        if counter is not None:
            counter.nfe += 1
        if scale == 1.0:
            if counter is not None:
                counter.passes += 1
            return net(x, tt, y)
        if counter is not None:
            counter.passes += 2
        out = net(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, null_label)]))
        vc, vu = out.chunk(2)
        return vu + scale * (vc - vu)
    return f


@torch.no_grad()
def integrate(f, x, y, steps: int, method: str = "euler", eta: float = 0.0, generator=None):
    """Integrate from t=0 to t=1 on a uniform grid with `steps` steps.

    euler    : 1 NFE / step.     heun / midpoint: 2 NFE / step.     rk4: 4 NFE / step.
    sde      : Euler-Maruyama on the SDE with the same marginals,  dx = [v + g^2/2 * score] dt + g dW,
               g(t) = eta * (1 - t),  score(x, t) = (t * v - x) / (1 - t)   (derived for the linear path above).
               eta = 0 reduces to `euler`.  The last step is taken noise-free, as g(1) = 0.
    """
    ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
    for i in range(steps):
        t, t1 = ts[i], ts[i + 1]
        h = t1 - t
        if method == "euler":
            x = x + h * f(x, t, y)
        elif method == "heun":
            k1 = f(x, t, y)
            k2 = f(x + h * k1, t1, y)
            x = x + h * 0.5 * (k1 + k2)
        elif method == "midpoint":
            k1 = f(x, t, y)
            x = x + h * f(x + 0.5 * h * k1, t + 0.5 * h, y)
        elif method == "rk4":
            k1 = f(x, t, y)
            k2 = f(x + 0.5 * h * k1, t + 0.5 * h, y)
            k3 = f(x + 0.5 * h * k2, t + 0.5 * h, y)
            k4 = f(x + h * k3, t1, y)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == "sde":
            v = f(x, t, y)
            g = eta * (1.0 - t)
            score = (t * v - x) / (1.0 - t)
            x = x + h * (v + 0.5 * g * g * score)
            if i < steps - 1:
                x = x + g * (h ** 0.5) * torch.randn(x.shape, device=x.device, generator=generator)
        else:
            raise ValueError(method)
    return x


NFE_PER_STEP = {"euler": 1, "heun": 2, "midpoint": 2, "rk4": 4, "sde": 1}


@torch.no_grad()
def sample(net, y, method: str, nfe: int, scale: float = 1.0, eta: float = 0.0, seed: int = 0,
           null_label: int = 10, img: int = 32, channels: int = 3):
    """Generate len(y) images. `nfe` is the velocity-evaluation budget; steps = nfe // NFE_PER_STEP[method]."""
    steps = max(1, nfe // NFE_PER_STEP[method])
    gen = torch.Generator(device=y.device).manual_seed(seed)
    x0 = torch.randn(len(y), channels, img, img, device=y.device, generator=gen)
    c = Counter()
    out = integrate(guided_velocity(net, scale, null_label, c), x0, y, steps, method, eta, gen)
    return out, c
