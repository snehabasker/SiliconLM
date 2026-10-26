# diffusion_study: flow matching on CIFAR-10 from scratch

**Status: work in progress. Code only, no training and no results yet.** Nothing here has been trained or evaluated,
so there are no FID or accuracy numbers in this folder.

## What exists

- `flowgen/model.py`: class-conditional U-Net (16.36M parameters), with a null class for classifier-free guidance.
- `flowgen/samplers.py`: Euler, Heun, midpoint, RK4 and an SDE sampler, with classifier-free guidance
  (`v = v_uncond + s (v_cond - v_uncond)`). Flow-matching convention `x_t = (1-t) x0 + t x1`, target velocity `x1 - x0`.
- `flowgen/data.py`, `flowgen/metrics.py`: CIFAR-10 loading; FID (pytorch-fid Inception features) and a small ResNet
  classifier for class agreement.
- `train.py`, `train_classifier.py`: flow-matching training with EMA and label dropout; classifier training.
- `eval_samplers.py`, `eval_cfg.py`: sampler x step-count sweep and guidance-scale sweep.
- `distill.py`, `eval_students.py`: progressive distillation of the guided sampler into few-step students, and their
  evaluation next to the undistilled base model at the same number of network evaluations.
- `tests/test_samplers.py`: checks that every ODE solver and the SDE sampler recover a known 1-D Gaussian
  distribution analytically. These 3 tests pass; the only run so far.

## Planned, not done

Train the base model (60k steps by default), run the sampler and guidance sweeps, distil to 8, 4, 2 and 1 steps,
evaluate. FID will use 10k generated samples against the CIFAR-10 training set, so it will **not** be comparable to
published numbers; it is only for comparing the variants here against each other.

## Not novel

Flow matching, classifier-free guidance and progressive distillation are all existing methods. This is a
from-scratch re-implementation for learning and comparison, not new research.
