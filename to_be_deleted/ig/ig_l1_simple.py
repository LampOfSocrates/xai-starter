"""
Lesson 1: Integrated Gradients from scratch
============================================
What you'll learn
-----------------
- What the Integrated Gradients (IG) attribution method actually computes.
- How to implement it in ~10 lines of PyTorch (a Riemann sum of gradients).
- How to CHECK your implementation using IG's "completeness" axiom.

The one-sentence idea
---------------------
"Which inputs made the model output what it did?"  IG answers this by asking:
as I slide each input from a boring BASELINE up to its real value, how much
does the output move? Inputs whose changes move the output a lot get a large
attribution; inputs that don't matter get ~0.

The formula
-----------
For a model f, an input x, and a baseline x' (the "neutral" reference):

    IG_i(x) = (x_i - x'_i) * integral_{a=0..1} d f(x' + a*(x - x')) / d x_i  da

In words: walk along the straight line from x' to x. At many points on that
path, measure the gradient of the output w.r.t. input i. Average those
gradients, then scale by how far input i had to travel (x_i - x'_i).

We can't do a real integral, so we approximate it with a sum over M evenly
spaced steps (a Riemann sum). More steps = more accurate.

Why not just use the gradient at x?
-----------------------------------
A plain gradient df/dx tells you the LOCAL slope at x. But models saturate:
near the answer the slope can flatten to ~0 even for important inputs, so the
gradient says "this input doesn't matter" — which is wrong. IG fixes this by
integrating gradients along the whole path from baseline to input, capturing
the contribution even through saturated regions.

The completeness axiom (our correctness check)
----------------------------------------------
IG is built so that the attributions ADD UP to the change in output:

    sum_i IG_i(x)  ==  f(x) - f(x')

If your numbers don't satisfy this (to within Riemann-sum error), your
implementation is wrong. We verify it explicitly below.

This whole file is pure PyTorch + numpy, runs in under a second on CPU.
No protein model yet — we use a tiny known function so you can check the math
by hand. Later lessons apply the exact same recipe to a real pLM.
"""

import torch

# ---------------------------------------------------------------------------
# Configuration. Edit these to experiment.
# ---------------------------------------------------------------------------

# Number of steps in the Riemann sum approximating the integral.
# Higher = more accurate but more forward/backward passes. 32-256 is typical.
N_STEPS = 64


# ---------------------------------------------------------------------------
# A tiny "model". We use a known nonlinear function of 3 inputs so we can
# reason about the answer instead of trusting a black box.
#
#     f(x) = x0 * x1  +  sin(x2)
#
# Intuition for what IG SHOULD say at x = [2, 3, 0], baseline x' = [0, 0, 0]:
#   - x0 and x1 multiply together, so both should get sizeable attribution.
#   - x2 enters as sin(x2); since x2 = 0 it travels nowhere, so ~0 attribution.
# ---------------------------------------------------------------------------
def f(x):
    """x has shape (batch, 3). Returns shape (batch,)."""
    return x[:, 0] * x[:, 1] + torch.sin(x[:, 2])


def integrated_gradients(model, x, baseline, n_steps=N_STEPS):
    """Compute Integrated Gradients of `model` output w.r.t. input `x`.

    Args:
        model:    a function taking (batch, D) -> (batch,)
        x:        the input we want to explain, shape (D,)
        baseline: the neutral reference, shape (D,)  (often all zeros)
        n_steps:  number of points along the path (Riemann-sum resolution)

    Returns:
        attributions: shape (D,) — one number per input feature.
    """
    # 1. Build the straight-line path from baseline -> x.
    #    alphas are the interpolation fractions a in [0, 1].
    #    We use the midpoints of each step (the "midpoint rule"), which is a
    #    more accurate Riemann sum than using the left or right edges.
    alphas = (torch.arange(n_steps, dtype=torch.float32) + 0.5) / n_steps  # (M,)

    # path[k] = baseline + alpha_k * (x - baseline), shape (M, D).
    delta = x - baseline                       # (D,) how far each input travels
    path = baseline.unsqueeze(0) + alphas.unsqueeze(1) * delta.unsqueeze(0)
    path.requires_grad_(True)                  # we need gradients w.r.t. these

    # 2. Forward pass on ALL path points at once, then one backward pass.
    #    Summing the outputs lets us get d(out_k)/d(path_k) for every k in a
    #    single .backward() call (the cross terms are zero because out_k only
    #    depends on path_k).
    outputs = model(path)                      # (M,)
    outputs.sum().backward()
    grads = path.grad                          # (M, D) gradient at each point

    # 3. Average the gradients over the path (this approximates the integral),
    #    then scale by how far each input travelled. This is the IG formula.
    avg_grads = grads.mean(dim=0)              # (D,)
    attributions = delta * avg_grads           # (D,)
    return attributions.detach()


def main():
    # The point we want to explain and the neutral baseline.
    x = torch.tensor([2.0, 3.0, 0.0])
    baseline = torch.tensor([0.0, 0.0, 0.0])

    # Evaluate the model at both ends of the path (needed for the check).
    fx = f(x.unsqueeze(0)).item()
    fb = f(baseline.unsqueeze(0)).item()

    print(f"Input    x  = {x.tolist()}")
    print(f"Baseline x' = {baseline.tolist()}")
    print(f"f(x)  = {fx:.4f}")
    print(f"f(x') = {fb:.4f}")
    print(f"Output change to explain, f(x) - f(x') = {fx - fb:.4f}\n")

    # Compute the attributions.
    attr = integrated_gradients(f, x, baseline, n_steps=N_STEPS)
    print(f"Integrated Gradients ({N_STEPS} steps):")
    for i, a in enumerate(attr.tolist()):
        print(f"  input {i}: {a:+.4f}")

    # ---- The completeness check ----
    # sum of attributions should equal f(x) - f(x').
    total = attr.sum().item()
    target = fx - fb
    err = abs(total - target)
    print(f"\nCompleteness check (IG axiom):")
    print(f"  sum(attributions) = {total:.4f}")
    print(f"  f(x) - f(x')      = {target:.4f}")
    print(f"  difference        = {err:.6f}  (should be ~0)")
    print("  PASS" if err < 1e-2 else "  FAIL — increase N_STEPS")

    # ---- Sanity vs. the plain gradient ----
    # Show why IG differs from "just take the gradient at x".
    xg = x.clone().unsqueeze(0).requires_grad_(True)
    f(xg).backward()
    print(f"\nFor comparison, the PLAIN gradient at x: {xg.grad.squeeze().tolist()}")
    print("  Note: plain grad for input 2 is cos(0)=1, suggesting it matters,")
    print("  but IG correctly gives it ~0 because x2 never left the baseline.")

    print(
        """
Things to experiment with:
- Set N_STEPS = 2 and watch the completeness error grow; then try 256.
- Change x to [2, 3, 1.5708] (x2 = pi/2). Now x2 travels through sin's curve
  and SHOULD get a real attribution. Re-check completeness.
- Change the baseline to [1, 1, 0] and see attributions change — IG always
  explains the output RELATIVE to the baseline you choose. Baseline choice
  matters a lot in practice.
- Replace f() with your own function and confirm completeness still holds.
- Swap the midpoint alphas for left-edge ones:
      alphas = torch.arange(n_steps, dtype=torch.float32) / n_steps
  and observe the approximation gets slightly worse for the same N_STEPS.
"""
    )


if __name__ == "__main__":
    main()
