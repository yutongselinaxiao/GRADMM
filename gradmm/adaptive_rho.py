"""Adaptive ADMM penalty (rho/sigma) updates for synthetic text generation.

Ported from PISA/Data Heterogenerity/experiment_sisa_practise_admm.py — the same
math, adapted for GRADMM's single-batch (no federated) ADMM loop.

Three update rules are provided, increasing in sophistication:

  1. heuristic_update_sigma         : He, Yang & Wang (2000) ratio-based update.
                                       Only needs primal/dual residuals.
                                       No log-space, no gradient cache.

  2. online_convex_bal_update_u     : OGD on u = log(sigma) with loss
                                       0.5 * (u - log(primal/dual))^2.
                                       Pure scalar OGD, no Lipschitz floor.

  3. online_convex_bal_lipschitz    : Same as (2) + hard projection
                                       sigma >= alpha * L_hat. Needs L_hat
                                       (Barzilai-Borwein style), which in turn
                                       needs gradient cache across iterations.

For each rule, the inputs are (primal_res, dual_base[, L_hat]) and the output
is the new sigma (or new u = log(sigma)).

A small AdaptiveRhoState class bundles the cross-iteration state
(EMA buffers, z_prev for dual residual, grad_prev/x_prev for L_hat).
"""

import math
import torch

# ---------------------------------------------------------------------------
# Atomic update rules (verbatim port from PISA, only docstring trimmed)
# ---------------------------------------------------------------------------


def heuristic_update_sigma(sigma_old, primal_res, dual_res, mu=10.0, tau=2.0,
                           k=0, k_max=50):
    """He, Yang & Wang (2000) strategy S3: doubling/halving on residual ratio.

    Converges in finite # of updates (tau_k = 1 for k <= k_max, 0 after).
    """
    if k > k_max:
        return sigma_old
    if primal_res > mu * dual_res:
        return sigma_old * tau
    elif dual_res > mu * primal_res:
        return sigma_old / tau
    return sigma_old


def online_convex_bal_update_u(u, primal_res, dual_base,
                               eta_u=0.1,
                               u_min=math.log(1e-6),
                               u_max=math.log(1e4),
                               eps=1e-12,
                               G_clip=10.0):
    """OGD on u = log(sigma), loss 0.5 * (u - target)^2, target = log(primal/dual_base).

    Returns (u_new, loss_val, target, grad_u).
    """
    primal_clip = torch.clamp(primal_res.detach(), min=eps)
    dual_clip = torch.clamp(dual_base.detach(), min=eps)
    target = torch.log(primal_clip) - torch.log(dual_clip)
    grad_u = u - target
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)
    with torch.no_grad():
        u_new = u - eta_u * grad_u
        u_new = torch.clamp(u_new, min=u_min, max=u_max)
        loss_val = 0.5 * (u - target).pow(2)
    return u_new.detach(), loss_val.detach(), target.detach(), grad_u.detach()


def online_convex_bal_lipschitz_update_u(u, primal_res, dual_base, L_hat,
                                         eta_u=0.05,
                                         G_clip=10.0,
                                         u_min=math.log(1e-6),
                                         u_max=math.log(1e4),
                                         eps=1e-12,
                                         lipschitz_floor_alpha=1.0):
    """OGD on u with hard Lipschitz projection: sigma >= alpha * L_hat.

    Uses the SAME gradient form as online_convex_bal_update_u (grad_u = u - target,
    i.e. loss = 0.5*(u-target)^2). This is a deliberate divergence from the PISA
    port (which used grad_u = 2*diff, effectively 2x the OGD step size). With this
    fix, the ONLY difference between this function and online_convex_bal_update_u
    is the hard floor `sigma >= alpha * L_hat`, so apples-to-apples comparisons
    at the same eta_u are valid.

    Returns (u_new, res_loss, target, log_L, floor_active, grad_u).
    """
    primal_clip = torch.clamp(primal_res.detach(), min=eps)
    dual_clip = torch.clamp(dual_base.detach(), min=eps)
    L_clip = torch.clamp(L_hat.detach(), min=eps)
    target = torch.log(primal_clip) - torch.log(dual_clip)
    log_L = torch.log(L_clip)
    log_floor = log_L + math.log(max(lipschitz_floor_alpha, eps))
    diff = u - target
    res_loss = 0.5 * diff.pow(2)
    grad_u = diff  # was 2.0 * diff in PISA; aligned with online_convex_bal_update_u
    grad_u = torch.clamp(grad_u, -G_clip, G_clip)
    with torch.no_grad():
        u_raw = u - eta_u * grad_u
        floor_active = (u_raw < log_floor).to(log_floor.dtype)
        u_new = torch.maximum(u_raw, log_floor)
        u_new = torch.clamp(u_new, min=u_min, max=u_max)
    return (u_new.detach(), res_loss.detach(), target.detach(),
            log_L.detach(), floor_active.detach(), grad_u.detach())


# ---------------------------------------------------------------------------
# Stateful wrapper that handles caching, EMA smoothing, and call sequencing.
# This is the only object generate.py needs to interact with.
# ---------------------------------------------------------------------------


class AdaptiveRhoState:
    """Holds cross-iteration state for adaptive rho.

    Usage in generate.py:

        state = AdaptiveRhoState(initial_rho=args.admm_rho, mode=args.sigma_mode,
                                 eta_u=..., G_clip=..., ema_beta=..., ...)
        for it in range(args.n_steps):
            state.before_z_update(z_embeds)   # cache z^k
            # ... existing z-update, x-update, lambda-update ...
            new_rho = state.after_lambda_update(it, x_embeds, z_embeds,
                                                grad=x_embeds.grad)
            args.admm_rho = new_rho   # use updated rho in next iter

    Modes: 'fixed' | 'heuristic' | 'online_convex_bal' | 'online_convex_bal_lipschitz'
    """

    def __init__(self, initial_rho, mode='fixed', device='cuda',
                 # OGD hyperparams
                 eta_u=0.05, G_clip=10.0, u_min=math.log(1e-6), u_max=math.log(1e4),
                 # EMA smoothing
                 ema_beta=0.9,
                 # Heuristic hyperparams
                 heuristic_mu=10.0, heuristic_tau=2.0, heuristic_k_max=50,
                 # Lipschitz hyperparams
                 lipschitz_floor_alpha=1.0, lipschitz_min_dz=1e-6,
                 lipschitz_max=1e4, lipschitz_ema_beta=0.9,
                 # Cadence
                 update_freq=1):
        self.mode = mode
        self.device = device
        self.rho = float(initial_rho)
        self.u = torch.tensor(math.log(self.rho), device=device)
        # OGD params
        self.eta_u = eta_u
        self.G_clip = G_clip
        self.u_min = u_min
        self.u_max = u_max
        # EMA buffers
        self.ema_beta = ema_beta
        self.primal_ema = None
        self.dual_ema = None
        # Heuristic
        self.heuristic_mu = heuristic_mu
        self.heuristic_tau = heuristic_tau
        self.heuristic_k_max = heuristic_k_max
        # Lipschitz
        self.lipschitz_floor_alpha = lipschitz_floor_alpha
        self.lipschitz_min_dz = lipschitz_min_dz
        self.lipschitz_max = lipschitz_max
        self.lipschitz_ema_beta = lipschitz_ema_beta
        self.L_hat_ema = torch.tensor(self.rho, device=device)
        # Cross-iter caches
        self.z_prev = None      # for dual residual
        self.x_prev = None      # for BB Lipschitz
        self.grad_prev = None   # for BB Lipschitz
        # Cadence
        self.update_freq = update_freq
        # Last-call diagnostics (for logging)
        self.last = {}

    def before_z_update(self, z_embeds):
        """Call BEFORE z gets overwritten (i.e., at the top of each outer iter)."""
        self.z_prev = z_embeds.detach().clone() if z_embeds is not None else None

    def after_lambda_update(self, iteration, x_embeds, z_embeds, grad=None):
        """Call at end of outer iter, AFTER lambda update.

        iteration : current outer iter index
        x_embeds  : current x (after x-update)
        z_embeds  : current z (after z-update)
        grad      : optional, current grad of reconstruction loss w.r.t. x
                    (needed only for lipschitz mode; pass x_embeds.grad)

        Returns the new rho (float).
        """
        if self.mode == 'fixed':
            return self.rho
        if (iteration + 1) % self.update_freq != 0:
            # Skip update this iter but still cache for the next call
            self._cache(x_embeds, grad)
            return self.rho

        with torch.no_grad():
            # 1. Primal residual: ||x - z||
            primal_res = (x_embeds - z_embeds).detach().norm()

            # 2. Dual base: ||z^{k+1} - z^k||
            if self.z_prev is not None:
                dual_base = (z_embeds - self.z_prev).detach().norm()
            else:
                dual_base = torch.tensor(1e-6, device=self.device)

            # 3. EMA smooth
            if self.primal_ema is None:
                self.primal_ema = primal_res
                self.dual_ema = dual_base
            else:
                b = self.ema_beta
                self.primal_ema = b * self.primal_ema + (1 - b) * primal_res
                self.dual_ema = b * self.dual_ema + (1 - b) * dual_base

            # 4. Dispatch to chosen rule
            if self.mode == 'heuristic':
                new_rho = heuristic_update_sigma(
                    self.rho, self.primal_ema, self.dual_ema,
                    mu=self.heuristic_mu, tau=self.heuristic_tau,
                    k=iteration, k_max=self.heuristic_k_max,
                )
                self.rho = float(new_rho)
                self.u = torch.tensor(math.log(max(self.rho, 1e-12)), device=self.device)
                self.last = {'primal': float(primal_res), 'dual_base': float(dual_base),
                             'rho': self.rho}

            elif self.mode == 'online_convex_bal':
                u_new, loss_val, target, grad_u = online_convex_bal_update_u(
                    self.u, self.primal_ema, self.dual_ema,
                    eta_u=self.eta_u, u_min=self.u_min, u_max=self.u_max,
                    G_clip=self.G_clip,
                )
                self.u = u_new
                self.rho = float(torch.exp(self.u).item())
                self.last = {'primal': float(primal_res), 'dual_base': float(dual_base),
                             'target_u': float(target), 'grad_u': float(grad_u),
                             'res_loss': float(loss_val),
                             'u': float(self.u), 'rho': self.rho}

            elif self.mode == 'online_convex_bal_lipschitz':
                # Compute L_hat via BB ratio over x and grad
                if (self.x_prev is not None and self.grad_prev is not None
                        and grad is not None):
                    dx = (x_embeds - self.x_prev).detach().norm()
                    if dx.item() >= self.lipschitz_min_dz:
                        dg = (grad.detach() - self.grad_prev).norm()
                        L_hat_raw = (dg / dx).clamp(max=self.lipschitz_max)
                        self.L_hat_ema = (self.lipschitz_ema_beta * self.L_hat_ema
                                          + (1 - self.lipschitz_ema_beta) * L_hat_raw)
                u_new, res_loss, target, log_L, floor_active, grad_u = (
                    online_convex_bal_lipschitz_update_u(
                        self.u, self.primal_ema, self.dual_ema, self.L_hat_ema,
                        eta_u=self.eta_u, G_clip=self.G_clip,
                        u_min=self.u_min, u_max=self.u_max,
                        lipschitz_floor_alpha=self.lipschitz_floor_alpha,
                    )
                )
                self.u = u_new
                self.rho = float(torch.exp(self.u).item())
                self.last = {'primal': float(primal_res), 'dual_base': float(dual_base),
                             'target_u': float(target), 'grad_u': float(grad_u),
                             'L_hat': float(self.L_hat_ema), 'log_L': float(log_L),
                             'res_loss': float(res_loss),
                             'floor_active': float(floor_active),
                             'u': float(self.u), 'rho': self.rho}
            else:
                raise ValueError(f"Unknown sigma_mode: {self.mode}")

        self._cache(x_embeds, grad)
        return self.rho

    def _cache(self, x_embeds, grad):
        self.x_prev = x_embeds.detach().clone() if x_embeds is not None else None
        if grad is not None:
            self.grad_prev = grad.detach().clone()
