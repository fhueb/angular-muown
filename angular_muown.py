"""AngularMuown optimizer for hidden neural-network weight matrices.

AngularMuown is intended for 2D hidden weight matrices such as transformer attention
and MLP projection weights. It represents each optimized matrix as
``W = diag(g) @ U``: ``g`` stores one learned row scale, while every row of
``U`` has unit Euclidean norm. This separates magnitude updates from directional
updates.

Given a weight gradient ``grad_W``, AngularMuown computes

    grad_g = <grad_W, U>_row
    grad_U = g * (grad_W - U * grad_g)

where ``grad_U`` is tangent to the product of row spheres. Adam updates the row
scales ``g``. A momentum buffer, optional Nesterov lookahead, Polar Express
orthogonalization, and row-normalization retraction update the directions
``U``:

    M = momentum * M + grad_U
    Q = polar(grad_U + momentum * M)  # with nesterov=True
    U = row_norm(U - adjust_lr(lr, shape(W)) * angular_lr_multiplier * Q)
    g = Adam(g, grad_g, lr)

Use AngularMuown together with a separate AdamW optimizer for embeddings, language
model heads, classifier heads, biases, normalization parameters, and other
non-hidden-matrix parameters. The optimizer keeps an internal
``angular_lr_multiplier`` that decays only the directional step while the
ordinary optimizer learning rate controls Adam on ``g``.
"""

import math
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer


EPS = 1e-7
ZERO_ROW_SCALE = 0.33**0.5

POLAR_EXPRESS_COEFFS = (
    (8.237312490495555, -23.157747414558198, 16.680568411445915),
    (4.082441999064836, -2.8930477353325887, 0.5252849256975651),
    (3.9263479922546556, -2.8547468034765293, 0.5318022422894989),
    (3.2982187133085143, -2.4245419810267062, 0.48632008358844075),
    (2.320007312889811, -1.6862169729967622, 0.42068027340235137),
)

ADJUST_LR_FNS = {"original", "match_rms_adamw"}


__all__ = [
    "AngularMuown",
    "AngularMuownDP",
    "zeropower_via_polar_express",
]


@torch.compile
def zeropower_via_polar_express(
    G: Tensor,
    steps: int = 5,
    eps: float = EPS,
) -> Tensor:
    """Polar Express quintic polynomial iteration for Muon-style orthogonalization."""
    if len(G.shape) != 2:
        raise ValueError("Polar Express input must be a 2D matrix")
    if steps < 0:
        raise ValueError(f"Invalid orthogonalization steps: {steps}")

    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X.div_(X.norm() * 1.01 + eps)
    for step in range(steps):
        a, b, c = POLAR_EXPRESS_COEFFS[min(step, len(POLAR_EXPRESS_COEFFS) - 1)]
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def _u_gradients(
    w: Tensor,
    g: Tensor,
    grad_w: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return U, dL/dg, and the Riemannian U gradient."""
    g_safe = torch.copysign(g.abs().clamp_min(EPS), g)
    u = w / g_safe
    grad_g = (grad_w * u).sum(dim=1, keepdim=True)
    grad_u = g * (grad_w - u * grad_g)
    return u, grad_g, grad_u


@torch.compile
def _u_recompose(w: Tensor, g: Tensor, u_step: Tensor, eps: float = EPS) -> None:
    """Normalize U and write W = g U."""
    u_step_norm = u_step.norm(dim=1, keepdim=True).clamp_min(eps)
    w.copy_(g * (u_step / u_step_norm))


def _shape_for_u_step(p: Tensor) -> tuple[int, int]:
    """Return the matrix shape used for sqrt(max(1, m/n)) U-step scaling.

    Packed QKV matrices are orthogonalized as three square chunks, so they use
    the square chunk shape for scaling.
    """
    if p.size(0) == 3 * p.size(1):
        return p.size(1), p.size(1)
    return p.size(0), p[0].numel()


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: tuple[int, int]) -> float:
    """Return the Muon-style shape-adjusted learning rate for the U update."""
    m, n = param_shape
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1.0, m / n))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(m, n))
    raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")


def _angular_lr_multiplier(
    angular_step: int,
    warmup_steps: int,
    decay_scale: float,
    decay_degree: float,
) -> float:
    """Return the inverse-polynomial angular learning-rate multiplier."""
    steps_after_warmup = max(0, angular_step - warmup_steps)
    if decay_degree == 0.0:
        return 1.0
    return (1.0 + decay_scale * steps_after_warmup) ** (-decay_degree)


def _orthogonalize_update(update: Tensor, orthogonalization_steps: int) -> Tensor:
    """Return the Polar Express direction, handling packed QKV matrices by chunk."""
    if update.size(0) != 3 * update.size(1):
        return zeropower_via_polar_express(update, steps=orthogonalization_steps)

    chunk_size = update.size(1)
    return torch.cat(
        [
            zeropower_via_polar_express(chunk, steps=orthogonalization_steps)
            for chunk in update.split(chunk_size, dim=0)
        ],
        dim=0,
    )


class AngularMuown(Optimizer):
    """Optimizer for 2D hidden weight matrices in normalized-row coordinates.

    AngularMuown stores one row magnitude ``g`` per row and treats the normalized
    rows ``U`` as points on a product of unit spheres. The row magnitudes are
    optimized with Adam. The row directions are optimized with a Muon-style
    spectral direction followed by row normalization.

    Pass only 2D hidden matrices to this optimizer. Use a separate AdamW
    optimizer for embeddings, language-model heads, classifier heads, biases,
    normalization parameters, and other 1D or scalar parameters.

    Args:
        params: Iterable of 2D parameters or parameter-group dictionaries.
        lr: Base learning rate used by Adam on ``g`` and by the ``U`` update.
        momentum: Momentum coefficient for the ``U``-gradient buffer.
        nesterov: Whether to use Nesterov-style lookahead for the ``U`` update.
        betas: Adam betas for the row magnitude ``g``.
        adam_eps: Adam epsilon for the row magnitude ``g``.
        orthogonalization_steps: Polar Express iteration count.
        warmup_steps: Number of optimizer steps before angular decay starts.
        decay_scale: Scale ``c`` in ``(1 + c * (t - warmup_steps)_+) ** (-p)``.
        decay_degree: Degree ``p`` in the angular decay.
        adjust_lr_fn: Shape adjustment for the ``U`` learning rate. ``None`` or
            ``"original"`` uses ``sqrt(max(1, m / n))``. ``"match_rms_adamw"``
            uses ``0.2 * sqrt(max(m, n))``.

    Row magnitudes whose absolute value would fall below ``EPS`` are projected
    back to magnitude ``EPS`` with their proposed sign. This keeps the explicit
    ``W = diag(g) @ U`` coordinates well-defined.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        orthogonalization_steps: int = 5,
        warmup_steps: int = 0,
        decay_scale: float = 0.001,
        decay_degree: float = 1.0,
        adjust_lr_fn: str | None = None,
    ):
        if not isinstance(params, list):
            params = list(params)
        if not params:
            raise ValueError("AngularMuown got an empty parameter list")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if adam_eps < 0.0:
            raise ValueError(f"Invalid adam_eps: {adam_eps}")
        if orthogonalization_steps < 0:
            raise ValueError(f"Invalid orthogonalization_steps: {orthogonalization_steps}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps: {warmup_steps}")
        if not math.isfinite(decay_scale) or decay_scale <= 0.0:
            raise ValueError(f"Invalid decay_scale: {decay_scale}")
        if not math.isfinite(decay_degree) or decay_degree < 0.0:
            raise ValueError(f"Invalid decay_degree: {decay_degree}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ADJUST_LR_FNS:
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            adam_eps=adam_eps,
            orthogonalization_steps=orthogonalization_steps,
            warmup_steps=warmup_steps,
            decay_scale=decay_scale,
            decay_degree=decay_degree,
            adjust_lr_fn=adjust_lr_fn,
            angular_step=0,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p_param in group["params"]:
                if p_param.ndim != 2:
                    raise ValueError(
                        "AngularMuown only supports 2D parameters, but found "
                        f"a parameter with size: {p_param.size()}",
                    )

    @torch.no_grad()
    def _init_state_2d(self, p: Tensor, state: dict) -> None:
        """Initialize AngularMuown state for a 2D parameter."""
        # Zero rows have no direction for U = W / g. Give just those rows the
        # expected row scale of the default linear init so the first U update
        # can create a unit direction without starting Adam's g near zero.
        w_norm = p.detach().norm(dim=1, keepdim=True)
        zero_rows = w_norm <= EPS
        if zero_rows.any():
            w_norm = torch.where(
                zero_rows,
                w_norm.new_full(w_norm.shape, ZERO_ROW_SCALE),
                w_norm,
            )
        state["g"] = w_norm.clone()
        state["m_u"] = torch.zeros_like(p)
        state["m_g"] = torch.zeros_like(w_norm)
        state["v_g"] = torch.zeros_like(w_norm)
        state["step"] = 0

    def _compute_u_step(
        self,
        update: Tensor,
        lr: float,
        orthogonalization_steps: int,
        angular_lr_multiplier: float,
        adjust_lr_fn: str | None,
    ) -> tuple[Tensor, float]:
        """Orthogonalize the U-gradient update and return its step size."""
        direction = _orthogonalize_update(update, orthogonalization_steps)
        step_size = (
            _adjust_lr(lr, adjust_lr_fn, _shape_for_u_step(direction))
            * angular_lr_multiplier
        )
        return direction, step_size

    @torch.no_grad()
    def _step_param(
        self,
        p: Tensor,
        grad: Tensor,
        lr: float,
        momentum: float,
        nesterov: bool,
        betas: tuple[float, float],
        adam_eps: float,
        orthogonalization_steps: int,
        angular_lr_multiplier: float,
        adjust_lr_fn: str | None,
    ) -> None:
        """Apply one AngularMuown update to a single 2D parameter."""
        if torch.is_complex(p):
            raise RuntimeError("AngularMuown does not support complex parameters")
        if grad.is_sparse:
            raise RuntimeError("AngularMuown does not support sparse gradients")
        if grad.ndim != 2:
            raise ValueError("AngularMuown parameter gradient must be a 2D matrix")

        state = self.state[p]
        if len(state) == 0:
            self._init_state_2d(p, state)

        state["step"] += 1
        step = state["step"]

        g = state["g"]
        m_u = state["m_u"]
        m_g = state["m_g"]
        v_g = state["v_g"]

        u, grad_g, grad_u = _u_gradients(p, g, grad)

        m_u.mul_(momentum).add_(grad_u)
        if nesterov:
            update = grad_u.add(m_u, alpha=momentum)
        else:
            update = m_u.clone()

        update, u_step_size = self._compute_u_step(
            update,
            lr,
            orthogonalization_steps,
            angular_lr_multiplier,
            adjust_lr_fn,
        )
        u_step = u.add(update, alpha=-u_step_size)

        beta1, beta2 = betas
        m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
        v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step
        g_update = (m_g / bc1) / (v_g / bc2).sqrt().add_(adam_eps)
        g_candidate = g.add(g_update, alpha=-lr)
        g.copy_(torch.copysign(g_candidate.abs().clamp_min(EPS), g_candidate))

        _u_recompose(p, g, u_step)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            betas = group["betas"]
            adam_eps = group["adam_eps"]
            orthogonalization_steps = group["orthogonalization_steps"]
            adjust_lr_fn = group.get("adjust_lr_fn")
            angular_lr_multiplier = _angular_lr_multiplier(
                group["angular_step"],
                group["warmup_steps"],
                group["decay_scale"],
                group["decay_degree"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue

                self._step_param(
                    p,
                    p.grad,
                    lr,
                    momentum,
                    nesterov,
                    betas,
                    adam_eps,
                    orthogonalization_steps,
                    angular_lr_multiplier,
                    adjust_lr_fn,
                )
            group["angular_step"] += 1

        return loss


def _param_to_complexity(p: Tensor) -> int:
    """Approximate orthogonalization cost for load-balanced sorting."""
    m, n = p.shape[0], p[0].numel()
    return 2 * (m**2) * n + m**3


class AngularMuownDP(AngularMuown):
    """Distributed AngularMuown optimizer for 2D parameter tensors.

    Parameters are sorted by estimated orthogonalization cost and processed in
    blocks of ``world_size``. Each rank updates one parameter from each block,
    then ``all_gather`` synchronizes the updated weights across ranks.
    """

    rank_sharded = True

    def __init__(self, params, **kwargs):
        if not isinstance(params, list):
            params = list(params)
        if not params:
            raise ValueError("AngularMuownDP got an empty parameter list")

        if not dist.is_initialized():
            raise ValueError("Using AngularMuownDP in a non-distributed run.")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if isinstance(params[0], dict):
            params = [
                {**group, "params": sorted(group["params"], key=_param_to_complexity, reverse=True)}
                for group in params
                if group["params"]
            ]
            if not params:
                raise ValueError("AngularMuownDP got only empty parameter groups")
        else:
            params = sorted(params, key=_param_to_complexity, reverse=True)

        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        allgather_handles = []
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            betas = group["betas"]
            adam_eps = group["adam_eps"]
            orthogonalization_steps = group["orthogonalization_steps"]
            adjust_lr_fn = group.get("adjust_lr_fn")
            angular_lr_multiplier = _angular_lr_multiplier(
                group["angular_step"],
                group["warmup_steps"],
                group["decay_scale"],
                group["decay_degree"],
            )
            params = group["params"]

            pad = (self.world_size - len(params) % self.world_size) % self.world_size
            params_pad = params + [torch.empty_like(params[-1]) for _ in range(pad)]

            for block_start in range(0, len(params), self.world_size):
                rank_param_index = block_start + self.rank
                if rank_param_index < len(params):
                    p = params[rank_param_index]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    self._step_param(
                        p,
                        p.grad,
                        lr,
                        momentum,
                        nesterov,
                        betas,
                        adam_eps,
                        orthogonalization_steps,
                        angular_lr_multiplier,
                        adjust_lr_fn,
                    )

                handle = dist.all_gather(
                    params_pad[block_start : block_start + self.world_size],
                    params_pad[block_start + self.rank],
                    async_op=True,
                )
                allgather_handles.append(handle)

        for handle in allgather_handles:
            handle.wait()

        for group in self.param_groups:
            group["angular_step"] += 1

        return loss

