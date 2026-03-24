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

Public API mirrors torch.optim.LBFGS so existing training loops require no
changes other than swapping the optimizer class.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dot(a: Tensor, b: Tensor) -> Tensor:
    """Flat dot product of two parameter-space vectors."""
    return torch.dot(a.view(-1), b.view(-1))


def _matvec(H: Tensor, v: Tensor) -> Tensor:
    """Matrix-vector product H @ v, H is (n, n), v is (n,)."""
    return H @ v


def _gather_flat_grad(params) -> Tensor:
    """Concatenate all parameter gradients into a single 1-D tensor."""
    views = []
    for p in params:
        if p.grad is None:
            views.append(p.data.new_zeros(p.numel()))
        elif p.grad.is_sparse:
            views.append(p.grad.to_dense().view(-1))
        else:
            views.append(p.grad.view(-1))
    return torch.cat(views)


def _set_params(params, flat: Tensor) -> None:
    """Write a flat vector back into the parameter list in-place."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p))
        offset += numel


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SelfScaledQuasiNewton(Optimizer, ABC):
    """
    Abstract base class for the self-scaled Broyden family of quasi-Newton
    line-search optimizers (Urbán et al. 2025, Sections 3.1-3.2).

    The general inverse-Hessian update rule is (eq. 10 of the paper):

        H_{k+1} = (1/tau_k) [ H_k
                              - (H_k y_k x H_k y_k) / (y_k . H_k y_k)
                              + phi_k  v_k x v_k ]
                  + (s_k x s_k) / (y_k . s_k)

    with auxiliary vector (eq. 9):

        v_k = sqrt(y_k . H_k y_k) * [ s_k / (y_k.s_k)
                                       - H_k y_k / (y_k . H_k y_k) ]

    Subclasses implement :meth:`_compute_tau_phi`, which returns the pair
    ``(tau_k, phi_k)`` specific to each algorithm variant.

    Public API
    ----------
    Matches ``torch.optim.LBFGS`` exactly so existing closures work unchanged::

        optimizer = SSBFGS(model.parameters(), lr=1.0,
                           max_iter=1, max_eval=20,
                           history_size=100,
                           tolerance_grad=1e-11,
                           tolerance_change=1e-11,
                           line_search_fn="strong_wolfe")

        for epoch in range(n_epochs):
            def closure():
                optimizer.zero_grad()
                loss = compute_loss(model)
                loss.backward()
                return loss
            optimizer.step(closure)

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Initial line-search step size.  Default 1.0 (standard for quasi-Newton).
    max_iter : int
        Max quasi-Newton steps per ``step()`` call.  Set to 1 to match the
        epoch-loop style used in the paper.
    max_eval : int or None
        Max function evaluations inside the line search.
        Defaults to ``max_iter * 5`` if not given.
    history_size : int or None
        Accepted for API compatibility with LBFGS; not used (full H is stored).
    tolerance_grad : float
        Stop if the infinity-norm of the gradient is below this value.
    tolerance_change : float
        Stop if the infinity-norm of the parameter update is below this value.
    line_search_fn : str or None
        ``"strong_wolfe"`` (default) or ``None`` (fixed step lr, for debugging).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 1,
        max_eval: Optional[int] = None,
        history_size: Optional[int] = None,
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
        line_search_fn: Optional[str] = "strong_wolfe",
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
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "SelfScaledQuasiNewton does not support per-parameter options "
                "(multiple parameter groups). Pass all parameters together."
            )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must override this
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_tau_phi(
        self,
        *,
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
            Parameter displacement  s_k = theta_{k+1} - theta_k  (eq. 7).
        yk : Tensor (n,)
            Gradient displacement  y_k = grad_J(theta_{k+1}) - grad_J(theta_k)
            (eq. 8).
        Hk : Tensor (n, n)
            Current inverse-Hessian approximation.
        alpha_k : float
            Step length chosen by the line search.
        grad_k : Tensor (n,)
            Gradient grad_J(theta_k) **before** the step (needed for eq. B.2).
        n : int
            Total number of trainable parameters.

        Returns
        -------
        tau : float    scaling  tau_k  (= 1 recovers standard BFGS)
        phi : float    updating phi_k  (= 1 recovers standard BFGS)
        """

    # ------------------------------------------------------------------
    # Shared inverse-Hessian update  (eq. 10)
    # ------------------------------------------------------------------

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

            H_{k+1} = (1/tau) [ H - Hy x Hy/(y.Hy) + phi v x v ] + s x s/(y.s)

        Returns Hk unchanged if the curvature condition y.s > 0 is not met,
        preserving positive definiteness.
        """
        Hy  = _matvec(Hk, yk)       # H_k y_k
        yHy = _dot(yk, Hy)          # y_k . H_k y_k
        ys  = _dot(yk, sk)          # y_k . s_k

        if ys.item() < 1e-20 or yHy.item() < 1e-20 or abs(tau) < 1e-20:
            return Hk

        # v_k  (eq. 9)
        vk = math.sqrt(yHy.item()) * (sk / ys - Hy / yHy)

        H_new = (1.0 / tau) * (
            Hk
            - torch.outer(Hy, Hy) / yHy
            + phi * torch.outer(vk, vk)
        ) + torch.outer(sk, sk) / ys

        return H_new

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]) -> Tensor:  # type: ignore[override]
        """
        Perform one (or more, if max_iter > 1) quasi-Newton step(s).

        Parameters
        ----------
        closure : callable
            Must zero gradients, run the forward pass, call loss.backward(),
            and return the loss -- identical to the torch.optim.LBFGS convention.

        Returns
        -------
        loss : Tensor   Loss at the new parameter values.
        """
        assert closure is not None, (
            "SelfScaledQuasiNewton.step() requires a closure."
        )

        group  = self.param_groups[0]
        lr     = group["lr"]
        ls_fn  = group["line_search_fn"]
        max_it = group["max_iter"]
        maxev  = group["max_eval"]

        params = [p for p in group["params"] if p.requires_grad]
        n      = sum(p.numel() for p in params)

        # Initialise state on first call
        state = self.state[params[0]]
        if len(state) == 0:
            state["H"]      = torch.eye(n, dtype=params[0].dtype,
                                        device=params[0].device)
            state["n_iter"] = 0

        loss = None
        for _ in range(max_it):

            # ---- 1. Evaluate loss + gradient at theta_k --------------------
            with torch.enable_grad():
                loss = closure()

            g_k = _gather_flat_grad(params).clone()

            if g_k.abs().max().item() < group["tolerance_grad"]:
                break

            H   = state["H"]
            d_k = -_matvec(H, g_k)         # search direction  (eq. 6)

            # If H is no longer a descent direction (numerical drift), reset it
            if _dot(d_k, g_k).item() >= 0:
                state["H"] = torch.eye(n, dtype=params[0].dtype,
                                       device=params[0].device)
                H   = state["H"]
                d_k = -g_k

            x_k = torch.cat([p.data.view(-1) for p in params]).clone()

            # ---- 2. Line search --------------------------------------------
            if ls_fn == "strong_wolfe":
                alpha, loss, g_kp1 = self._line_search_wolfe(
                    params=params,
                    x_k=x_k,
                    f_k=loss,
                    g_k=g_k,
                    d_k=d_k,
                    closure=closure,
                    lr=lr,
                    max_eval=maxev,
                )
            else:
                alpha = lr
                _set_params(params, x_k + alpha * d_k)
                with torch.enable_grad():
                    loss = closure()
                g_kp1 = _gather_flat_grad(params).clone()

            # ---- 3. Displacements  (eqs. 7-8) ------------------------------
            sk = alpha * d_k
            yk = g_kp1 - g_k

            # ---- 4. Hessian update  (eq. 10) --------------------------------
            if _dot(yk, sk).item() > 1e-10:
                tau, phi = self._compute_tau_phi(
                    sk=sk, yk=yk, Hk=H,
                    alpha_k=alpha, grad_k=g_k, n=n,
                )
                state["H"] = self._update_hessian(H, sk, yk, tau, phi)

            if sk.abs().max().item() < group["tolerance_change"]:
                break

            state["n_iter"] += 1

        return loss

    # ------------------------------------------------------------------
    # Line search
    # ------------------------------------------------------------------

    def _line_search_wolfe(
        self,
        *,
        params,
        x_k: Tensor,
        f_k: Tensor,
        g_k: Tensor,
        d_k: Tensor,
        closure: Callable,
        lr: float,
        max_eval: int,
    ) -> tuple[float, Tensor, Tensor]:
        """
        Strong-Wolfe line search.

        Tries PyTorch's internal _strong_wolfe (same as used by LBFGS), with a
        backtracking-Armijo fallback that is always available.

        Returns
        -------
        alpha   : float   accepted step length
        f_new   : Tensor  loss at the new parameters
        g_new   : Tensor  flat gradient at the new parameters
        """
        # -- Attempt PyTorch's _strong_wolfe ---------------------------------
        try:
            from torch.optim.lbfgs import _strong_wolfe  # type: ignore[import]

            gtd = _dot(g_k, d_k).item()

            def obj_func(x, t, d):
                # Evaluate at x + t*d, then restore to x (PyTorch convention).
                _set_params(params, x + t * d)
                with torch.enable_grad():
                    f = closure()
                g = _gather_flat_grad(params).clone()
                _set_params(params, x)
                return f, g

            alpha, _, _, f_new, _, g_new = _strong_wolfe(
                obj_func, x_k, lr, d_k, f_k, g_k, gtd,
                max_ls=max_eval,
            )

            # Apply the accepted step and run closure to leave grads populated.
            _set_params(params, x_k + alpha * d_k)
            with torch.enable_grad():
                f_new = closure()
            g_new = _gather_flat_grad(params).clone()
            return float(alpha), f_new, g_new

        except Exception:
            pass

        # -- Backtracking Armijo fallback ------------------------------------
        c1    = 1e-4
        alpha = lr
        gtd   = _dot(g_k, d_k).item()
        f0    = f_k.item() if isinstance(f_k, Tensor) else float(f_k)

        for _ in range(max_eval):
            _set_params(params, x_k + alpha * d_k)
            with torch.enable_grad():
                f_new = closure()
            if f_new.item() <= f0 + c1 * alpha * gtd:
                g_new = _gather_flat_grad(params).clone()
                return alpha, f_new, g_new
            alpha *= 0.5

        _set_params(params, x_k + alpha * d_k)
        with torch.enable_grad():
            f_new = closure()
        g_new = _gather_flat_grad(params).clone()
        return alpha, f_new, g_new


# ---------------------------------------------------------------------------
# Concrete subclass 1: SSBFGS
# ---------------------------------------------------------------------------

class SSBFGS(SelfScaledQuasiNewton):
    """
    Self-Scaled BFGS (Urbán et al. 2025, Section 3.2; Al-Baali 1998).

    Uses:
        tau_k = min{ 1,  (y_k . s_k) / (s_k . H_k^{-1} s_k) }    (eq. 11)

    Computed efficiently without inverting H_k via eq. B.1
    (H_k^{-1} s_k = -alpha_k * grad_J(theta_k)):

        tau_k = min{ 1,  -(y_k . s_k) / (alpha_k  s_k . grad_J(theta_k)) }
                                                                    (eq. B.2)
        phi_k = 1                                                   (eq. 12)

    When tau_k = 1 this reduces exactly to standard BFGS. The self-scaling
    kicks in (tau_k < 1) only when the curvature ratio falls below 1.
    """

    def _compute_tau_phi(
        self, *, sk, yk, Hk, alpha_k, grad_k, n,
    ) -> tuple[float, float]:
        """tau_k via eq. B.2, phi_k = 1 (eq. 12)."""
        ys    = _dot(yk, sk).item()          # y_k . s_k
        sg    = _dot(sk, grad_k).item()      # s_k . grad_J(theta_k)
        denom = -alpha_k * sg
        tau   = min(1.0, ys / denom) if abs(denom) > 1e-30 else 1.0
        return tau, 1.0


# ---------------------------------------------------------------------------
# Concrete subclass 2: SSBroyden
# ---------------------------------------------------------------------------

class SSBroyden(SelfScaledQuasiNewton):
    """
    Self-Scaled Broyden (Urbán et al. 2025, Section 3.2; Al-Baali & Khalfan 2005).

    Extends SSBFGS by also varying phi_k. Auxiliary scalars (eqs. 15-23):

    b_k = -alpha_k (s_k . grad_J(theta_k)) / (y_k . s_k)   (eq. 15, via B.1)
    h_k = (y_k . H_k y_k) / (y_k . s_k)                    (eq. 16)
    a_k = h_k b_k - 1                                        (eq. 17)
    c_k = sqrt(a_k / (a_k + 1))                              (eq. 18)
    rho_k^- = min(1,  h_k (1 - c_k))                         (eq. 19)
    theta_k^- = (rho_k^- - 1) / a_k                          (eq. 20)
    theta_k^+ = 1 / rho_k^-                                  (eq. 21)
    theta_k = max(theta_k^-, min(theta_k^+, (1-b_k)/b_k))    (eq. 22)
    sigma_k = 1 + a_k theta_k                                (eq. 23)

    Updating parameter (eq. 14):
        phi_k = 1 - theta_k / (1 + a_k theta_k)

    Scaling parameter (eq. 13):
        if theta_k > 0:  tau_k = tau_k^(1) * min(sigma_k^{-1/(n-1)}, 1/theta_k)
        else:            tau_k = min(tau_k^(1) * sigma_k^{-1/(n-1)},  sigma_k)
    """

    def _compute_tau_phi(
        self, *, sk, yk, Hk, alpha_k, grad_k, n,
    ) -> tuple[float, float]:
        """tau_k and phi_k via eqs. 13-23."""
        ys  = _dot(yk, sk).item()
        sg  = _dot(sk, grad_k).item()
        Hy  = _matvec(Hk, yk)
        yHy = _dot(yk, Hy).item()

        # b_k (eq. 15)
        b_k = (-alpha_k * sg) / ys if abs(ys) > 1e-30 else 1.0

        # h_k (eq. 16)
        h_k = yHy / ys if abs(ys) > 1e-30 else 1.0

        # a_k (eq. 17)
        a_k = h_k * b_k - 1.0

        # c_k (eq. 18) — clamp argument to [0, inf) for safety
        ratio = a_k / (a_k + 1.0) if abs(a_k + 1.0) > 1e-30 else 0.0
        c_k   = math.sqrt(max(ratio, 0.0))

        # rho_k^- (eq. 19)
        rho_minus = min(1.0, h_k * (1.0 - c_k))

        # theta_k^- (eq. 20)
        theta_minus = (rho_minus - 1.0) / a_k if abs(a_k) > 1e-30 else 0.0

        # theta_k^+ (eq. 21)
        theta_plus = 1.0 / rho_minus if abs(rho_minus) > 1e-30 else float("inf")

        # theta_k (eq. 22)
        mid     = (1.0 - b_k) / b_k if abs(b_k) > 1e-30 else 0.0
        theta_k = max(theta_minus, min(theta_plus, mid))

        # sigma_k (eq. 23)
        sigma_k = 1.0 + a_k * theta_k

        # tau_k^(1) from SSBFGS (eq. B.2)
        denom_tau1 = -alpha_k * sg
        tau1 = min(1.0, ys / denom_tau1) if abs(denom_tau1) > 1e-30 else 1.0

        # tau_k (eq. 13)
        exp_val     = -1.0 / max(n - 1, 1)
        sigma_power = abs(sigma_k) ** exp_val if abs(sigma_k) > 1e-30 else 1.0

        if theta_k > 0.0:
            inv_theta = 1.0 / theta_k if abs(theta_k) > 1e-30 else float("inf")
            tau = tau1 * min(sigma_power, inv_theta)
        else:
            tau = min(tau1 * sigma_power, sigma_k)

        # phi_k (eq. 14)
        denom_phi = 1.0 + a_k * theta_k
        phi = 1.0 - theta_k / denom_phi if abs(denom_phi) > 1e-30 else 1.0

        return float(tau), float(phi)