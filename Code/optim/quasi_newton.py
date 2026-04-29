"""
Self-Scaled Quasi-Newton Optimizers for Physics-Informed Neural Networks
=========================================================================
Implementation based on:
    Urbán, Stefanou & Pons (2025), "Unveiling the optimization process of
    physics informed neural networks: How accurate and competitive can PINNs be?"
    Journal of Computational Physics 523, 113656.

The paper proposes two modifications of the BFGS quasi-Newton algorithm —
SSBFGS and SSBroyden — under the general self-scaled Broyden family (eq. 10):

    H_{k+1} = (1/τ_k) [ H_k - (H_k y_k ⊗ H_k y_k)/(y_k · H_k y_k)
                        + φ_k v_k ⊗ v_k ] + (s_k ⊗ s_k)/(y_k · s_k)

where τ_k (scaling) and φ_k (updating) differ per algorithm.
Standard BFGS corresponds to τ_k = 1, φ_k = 1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot(a: Tensor, b: Tensor) -> Tensor:
    """Efficient dot product of two flat parameter vectors."""
    return torch.dot(a.view(-1), b.view(-1))


def _gather_flat_grad(params) -> Tensor:
    """Concatenate all parameter gradients into a single 1‑D tensor (float64)."""
    views = []
    for p in params:
        if p.grad is None:
            views.append(p.data.new_zeros(p.numel(), dtype=torch.float64))
        else:
            views.append(p.grad.detach().to(dtype=torch.float64).view(-1))
    return torch.cat(views)


def _gather_flat_param(params) -> Tensor:
    """Concatenate all parameters into a single 1‑D tensor (float64)."""
    views = [p.data.detach().to(dtype=torch.float64).view(-1) for p in params]
    return torch.cat(views)


def _set_params(params, flat: Tensor) -> None:
    """Write a flat vector (float64) back into parameters (preserving original dtype)."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p).to(p.dtype))
        offset += numel


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SelfScaledQuasiNewton(Optimizer, ABC):
    """
    Abstract base class for the self-scaled Broyden family of quasi-Newton
    line-search optimizers (Urbán et al. 2025, Sections 3.1-3.2).

    The general inverse-Hessian update rule is (eq. 10 of the paper):

        H_{k+1} = (1/tau_k) [ H_k
                              - (H_k y_k ⊗ H_k y_k) / (y_k . H_k y_k)
                              + phi_k  v_k ⊗ v_k ]
                  + (s_k ⊗ s_k) / (y_k . s_k)

    with auxiliary vector (eq. 9):

        v_k = sqrt(y_k . H_k y_k) * [ s_k / (y_k.s_k)
                                       - H_k y_k / (y_k . H_k y_k) ]

    Subclasses implement :meth:`_compute_tau_phi`, which returns the pair
    ``(tau_k, phi_k)`` specific to each algorithm variant.

    Public API matches torch.optim.LBFGS exactly.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 1,
        max_eval: Optional[int] = None,
        history_size: Optional[int] = None,      # kept for API compatibility
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
        line_search_fn: Optional[str] = "strong_wolfe",
        c1: float = 1e-4,                        # Armijo condition constant
        c2: float = 0.9,                         # Curvature condition constant
        beta: float = 0.5,                       # Backtracking factor
    ) -> None:
        if max_eval is None:
            max_eval = max_iter * 5

        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
            c1=c1,
            c2=c2,
            beta=beta,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "SelfScaledQuasiNewton does not support per-parameter options "
                "(multiple parameter groups). Pass all parameters together."
            )

    @abstractmethod
    def _compute_tau_phi(
        self,
        sk: Tensor,
        yk: Tensor,
        Hk: Tensor,
        alpha_k: float,
        grad_k: Tensor,
        n: int,
    ) -> tuple[float, float]:
        """
        Compute the scaling parameter tau_k and updating parameter phi_k.

        Parameters
        ----------
        sk : Tensor (n,)
            Parameter displacement s_k = theta_{k+1} - theta_k (eq. 7).
        yk : Tensor (n,)
            Gradient displacement y_k = grad_J(theta_{k+1}) - grad_J(theta_k) (eq. 8).
        Hk : Tensor (n, n)
            Current inverse-Hessian approximation.
        alpha_k : float
            Step length chosen by the line search.
        grad_k : Tensor (n,)
            Gradient grad_J(theta_k) before the step.
        n : int
            Total number of trainable parameters.

        Returns
        -------
        tau : float    scaling tau_k (1 recovers standard BFGS)
        phi : float    updating phi_k (1 recovers standard BFGS)
        """

    def _update_hessian(
        self,
        Hk: Tensor,
        sk: Tensor,
        yk: Tensor,
        tau: float,
        phi: float,
    ) -> Tensor:
        """
        Self-scaled Broyden inverse-Hessian update (eq. 10 of the paper).
        Returns Hk unchanged if curvature condition fails.
        """
        Hy = Hk @ yk
        yHy = _dot(yk, Hy).item()
        ys = _dot(yk, sk).item()

        if ys < 1e-12 or yHy < 1e-12 or abs(tau) < 1e-12:
            return Hk

        # Auxiliary vector v_k (eq. 9)
        vk = math.sqrt(yHy) * (sk / ys - Hy / yHy)

        # Update (eq. 10)
        H_new = (1.0 / tau) * (
            Hk - torch.outer(Hy, Hy) / yHy + phi * torch.outer(vk, vk)
        ) + torch.outer(sk, sk) / ys
        return H_new

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """Perform one (or more, if max_iter > 1) quasi-Newton step(s)."""
        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tol_grad = group["tolerance_grad"]
        tol_change = group["tolerance_change"]
        c1 = group["c1"]
        c2 = group["c2"]
        beta = group["beta"]
        ls_fn = group["line_search_fn"]

        params = [p for p in group["params"] if p.requires_grad]
        n = sum(p.numel() for p in params)

        # Initialise state (Hessian in float64 for numerical stability)
        state = self.state[params[0]]
        if not state:
            device = params[0].device
            state["H"] = torch.eye(n, dtype=torch.float64, device=device)
            state["n_iter"] = 0

        loss = None
        for _ in range(max_iter):
            # ---- 1. Evaluate loss & gradient at current point ----
            with torch.enable_grad():
                loss = closure()
            g_k = _gather_flat_grad(params)
            if g_k.abs().max().item() < tol_grad:
                break

            H = state["H"]
            d_k = - (H @ g_k)

            # Ensure descent direction (reset Hessian if needed)
            if _dot(d_k, g_k).item() >= 0:
                H = torch.eye(n, dtype=torch.float64, device=g_k.device)
                state["H"] = H
                d_k = -g_k

            x_k = _gather_flat_param(params)

            # ---- 2. Line search ----
            if ls_fn == "strong_wolfe":
                alpha, loss, g_kp1 = self._strong_wolfe(
                    params, x_k, loss, g_k, d_k, closure,
                    lr=lr, max_eval=max_eval, c1=c1, c2=c2,
                )
            else:
                # Fixed step or backtracking fallback
                alpha, loss, g_kp1 = self._backtracking(
                    params, x_k, loss, g_k, d_k, closure,
                    lr=lr, max_eval=max_eval, c1=c1, beta=beta,
                )

            # ---- 3. Displacements (eqs. 7-8) ----
            sk = alpha * d_k
            yk = g_kp1 - g_k

            # ---- 4. Hessian update (eq. 10) ----
            if _dot(yk, sk).item() > 1e-12:
                tau, phi = self._compute_tau_phi(sk, yk, H, alpha, g_k, n)
                state["H"] = self._update_hessian(H, sk, yk, tau, phi)

            # Check for convergence
            if sk.abs().max().item() < tol_change:
                break

            state["n_iter"] += 1

        return loss

    # ------------------------------------------------------------------
    # Strong Wolfe line search (using PyTorch's internal implementation)
    # ------------------------------------------------------------------

    def _strong_wolfe(
        self,
        params,
        x_k: Tensor,
        f_k: Tensor,
        g_k: Tensor,
        d_k: Tensor,
        closure: Callable,
        lr: float,
        max_eval: int,
        c1: float,
        c2: float,
    ) -> tuple[float, Tensor, Tensor]:
        """Strong Wolfe line search via torch.optim.lbfgs._strong_wolfe."""
        try:
            from torch.optim.lbfgs import _strong_wolfe  # type: ignore

            gtd = _dot(g_k, d_k).item()

            def obj_func(x, t, d):
                # Evaluate at x + t*d, returns (f, g)
                _set_params(params, x + t * d)
                with torch.enable_grad():
                    f = closure()
                g = _gather_flat_grad(params)
                _set_params(params, x)        # restore
                return f, g

            # The line search modifies parameters in place.
            alpha, _, _, f_new, _, g_new = _strong_wolfe(
                obj_func, x_k, lr, d_k, f_k, g_k, gtd,
                max_ls=max_eval, c1=c1, c2=c2,
            )
            # Ensure parameters are exactly at the accepted point
            _set_params(params, x_k + alpha * d_k)
            # Final closure run to get the correct loss/grad (safety)
            with torch.enable_grad():
                f_confirm = closure()
            g_confirm = _gather_flat_grad(params)
            return alpha, f_confirm, g_confirm

        except Exception:
            # Fallback to simple backtracking
            return self._backtracking(
                params, x_k, f_k, g_k, d_k, closure,
                lr=lr, max_eval=max_eval, c1=c1, beta=0.5,
            )

    # ------------------------------------------------------------------
    # Backtracking Armijo fallback
    # ------------------------------------------------------------------

    def _backtracking(
        self,
        params,
        x_k: Tensor,
        f_k: Tensor,
        g_k: Tensor,
        d_k: Tensor,
        closure: Callable,
        lr: float,
        max_eval: int,
        c1: float,
        beta: float,
    ) -> tuple[float, Tensor, Tensor]:
        """Backtracking line search (Armijo condition only)."""
        alpha = lr
        gtd = _dot(g_k, d_k).item()
        f0 = f_k.item() if isinstance(f_k, Tensor) else float(f_k)

        for _ in range(max_eval):
            _set_params(params, x_k + alpha * d_k)
            with torch.enable_grad():
                f_new = closure()
            if f_new.item() <= f0 + c1 * alpha * gtd:
                g_new = _gather_flat_grad(params)
                return alpha, f_new, g_new
            alpha *= beta

        # If we exit the loop, take the last tried step
        g_new = _gather_flat_grad(params)
        return alpha, f_new, g_new


# ---------------------------------------------------------------------------
# Concrete subclass: SSBFGS
# ---------------------------------------------------------------------------

class SSBFGS(SelfScaledQuasiNewton):
    """
    Self-Scaled BFGS (Urbán et al. 2025, Section 3.2; Al-Baali 1998).

    Uses:
        τ_k = min{ 1,  (y_k . s_k) / (s_k . H_k^{-1} s_k) }    (eq. 11)

    Computed efficiently without inverting H_k via eq. B.1
    (H_k^{-1} s_k = -α_k * grad_J(θ_k)):

        τ_k = min{ 1,  -(y_k . s_k) / (α_k  s_k . grad_J(θ_k)) }  (eq. B.2)
        φ_k = 1                                                   (eq. 12)

    When τ_k = 1 this reduces exactly to standard BFGS.
    """

    def _compute_tau_phi(self, sk, yk, Hk, alpha_k, grad_k, n):
        ys = _dot(yk, sk).item()
        sg = _dot(sk, grad_k).item()
        denom = -alpha_k * sg
        if abs(denom) > 1e-30:
            tau = min(1.0, ys / denom)
        else:
            tau = 1.0
        return tau, 1.0


# ---------------------------------------------------------------------------
# Concrete subclass: SSBroyden
# ---------------------------------------------------------------------------

class SSBroyden(SelfScaledQuasiNewton):
    """
    Self-Scaled Broyden (Urbán et al. 2025, Section 3.2; Al-Baali & Khalfan 2005).

    Extends SSBFGS by also varying φ_k. Auxiliary scalars (eqs. 15-23):

        b_k = -α_k (s_k . grad_J(θ_k)) / (y_k . s_k)   (eq. 15)
        h_k = (y_k . H_k y_k) / (y_k . s_k)            (eq. 16)
        a_k = h_k b_k - 1                              (eq. 17)
        c_k = sqrt(a_k / (a_k + 1))                    (eq. 18)
        ρ_k^- = min(1,  h_k (1 - c_k))                 (eq. 19)
        θ_k^- = (ρ_k^- - 1) / a_k                      (eq. 20)
        θ_k^+ = 1 / ρ_k^-                              (eq. 21)
        θ_k = max(θ_k^-, min(θ_k^+, (1-b_k)/b_k))      (eq. 22)
        σ_k = 1 + a_k θ_k                              (eq. 23)

    Updating parameter (eq. 14):
        φ_k = (1 - θ_k) / (1 + a_k θ_k)

    Scaling parameter (eq. 13):
        if θ_k > 0:  τ_k = τ_k^(1) * min(σ_k^{-1/(n-1)}, 1/θ_k)
        else:        τ_k = min(τ_k^(1) * σ_k^{-1/(n-1)}, σ_k)
    """

    def _compute_tau_phi(self, sk, yk, Hk, alpha_k, grad_k, n):
        # --- basic scalars ---
        ys = _dot(yk, sk).item()
        sg = _dot(sk, grad_k).item()
        Hy = Hk @ yk
        yHy = _dot(yk, Hy).item()

        # b_k (eq. 15), h_k (eq. 16), a_k (eq. 17)
        b_k = (-alpha_k * sg) / ys if abs(ys) > 1e-30 else 1.0
        h_k = yHy / ys if abs(ys) > 1e-30 else 1.0
        a_k = h_k * b_k - 1.0

        # c_k (eq. 18) – protect sqrt argument
        ratio = a_k / (a_k + 1.0) if abs(a_k + 1.0) > 1e-30 else 0.0
        c_k = math.sqrt(max(ratio, 0.0))

        # ρ_k^- (eq. 19), θ_k^- (eq. 20), θ_k^+ (eq. 21)
        rho_minus = min(1.0, h_k * (1.0 - c_k))
        theta_minus = (rho_minus - 1.0) / a_k if abs(a_k) > 1e-30 else 0.0
        theta_plus = 1.0 / rho_minus if abs(rho_minus) > 1e-30 else float("inf")

        # θ_k (eq. 22)
        mid = (1.0 - b_k) / b_k if abs(b_k) > 1e-30 else 0.0
        theta_k = max(theta_minus, min(theta_plus, mid))

        # σ_k (eq. 23)
        sigma_k = 1.0 + a_k * theta_k

        # τ_k^(1) from SSBFGS (eq. B.2)
        tau1 = min(1.0, -ys / (alpha_k * sg)) if abs(alpha_k * sg) > 1e-30 else 1.0

        # σ_k^{-1/(n-1)} with protection for n=1 (unlikely in PINNs)
        exp_val = -1.0 / max(n - 1, 1)
        sigma_power = abs(sigma_k) ** exp_val if abs(sigma_k) > 1e-30 else 1.0

        # τ_k (eq. 13)
        if theta_k > 0.0:
            inv_theta = 1.0 / theta_k if abs(theta_k) > 1e-30 else float("inf")
            tau = tau1 * min(sigma_power, inv_theta)
        else:
            tau = min(tau1 * sigma_power, sigma_k)

        # φ_k (eq. 14)
        denom_phi = 1.0 + a_k * theta_k
        phi = 1.0 - theta_k / denom_phi if abs(denom_phi) > 1e-30 else 1.0

        return tau, phi