"""Optional GPU backend for the E-step chunk math (torch; auto-detected).

Division of labor per chunk when a GPU is active:

  CPU (worker threads, unchanged)   pair building on the sorted event stream,
                                    candidate binary search, α/z gathers,
                                    time-slack weights, all sufficient-stat
                                    scatters (float64 np.add.at)
  GPU (this module)                 the per-pair kernel math — softplus, the
                                    exponential kernel score, per-event rate,
                                    responsibilities, the chunk's Σ log rate

so every EM accumulator stays float64 on the CPU regardless of the GPU
compute dtype, and the GPU is a pure math accelerator with numpy in/out.

Detection (resolve_estep_backend): torch importable AND a CUDA GPU present.
``estep_device="auto"`` selects CUDA when available; without torch or a CUDA
GPU it silently resolves to None and the CPU path runs exactly as before.
Explicitly requesting ``cuda`` raises if that device is unavailable.

CUDA uses float64, so numerics track the CPU path closely. GPU results are
statistically equivalent to CPU results but not bit-identical because CUDA
scatter atomics are not run-to-run deterministic, unlike the CPU path which is
exactly reproducible for any estep_workers. Pin estep_device="cpu" when
bitwise reproducibility matters more than speed.

Thread-safety: workers may call chunk_scores concurrently; a lock serializes
the GPU section (kernels on one device serialize anyway) while pair building
keeps overlapping on the CPU threads.
"""

from __future__ import annotations

import threading

import numpy as np

_EPS = 1e-12

_RESOLVE_CACHE: dict = {}


class TorchEstepBackend:
    """numpy-in / numpy-out chunk scorer on a torch device."""

    def __init__(self, torch_mod, device: str):
        self._torch = torch_mod
        self.device = torch_mod.device(device)
        self.dtype = torch_mod.float64
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.device.type}/{'f32' if self.dtype == self._torch.float32 else 'f64'}"

    def _to(self, arr, dtype):
        return self._torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype).to(
            self.device, non_blocking=True
        )

    def chunk_scores(
        self,
        *,
        a: np.ndarray,
        softplus_a: bool,
        beta,
        dt: np.ndarray,
        late_weight,
        tlocal: np.ndarray,
        mu_chunk: np.ndarray,
        csize: int,
    ):
        """Exp-kernel E-step math for one chunk.

        score_p = A_p · β_p · exp(-β_p · dt_p) · late_weight_p, where
        A = softplus(a) when softplus_a else a, and β is a scalar or per-pair.
        ``tlocal`` maps each pair to its target event in [0, csize); pairs with
        zero score for other targets may simply be omitted by the caller.

        Returns (p_self (csize,) f64, p_ij (P,) f64, ll float) as numpy/host
        values, where p_ij = score / rate[tlocal] and ll = Σ log rate.
        """
        torch = self._torch
        with self._lock, torch.no_grad():
            a_t = self._to(a, self.dtype)
            if softplus_a:
                a_t = torch.nn.functional.softplus(a_t)   # log1p(exp) — same as CPU softplus
            dt_t = self._to(dt, self.dtype)
            if np.ndim(beta) == 0:
                score = a_t * float(beta) * torch.exp(dt_t * (-float(beta)))
            else:
                beta_t = self._to(beta, self.dtype)
                score = a_t * beta_t * torch.exp(-beta_t * dt_t)
            if late_weight is not None:
                score = score * self._to(late_weight, self.dtype)
            tl = self._to(tlocal, torch.long)
            sum_score = torch.zeros(csize, dtype=self.dtype, device=self.device)
            sum_score.index_add_(0, tl, score)
            mu_t = self._to(mu_chunk, self.dtype)
            rate = torch.clamp(mu_t + sum_score, min=_EPS)
            p_self = mu_t / rate
            p_ij = score / rate.index_select(0, tl)
            ll = torch.log(rate).sum()
            return (
                p_self.cpu().numpy().astype(np.float64, copy=False),
                p_ij.cpu().numpy().astype(np.float64, copy=False),
                float(ll.cpu()),
            )


def resolve_estep_backend(config) -> "TorchEstepBackend | None":
    """Map config.estep_device to a backend instance (cached) or None (CPU).

    "auto" → CUDA if available, else None — never raises, so the default
    config runs anywhere. "cuda" → that device or ValueError.
    "cpu" → None.
    """
    setting = str(getattr(config, "estep_device", "cpu") or "cpu").strip().lower()
    if setting == "cpu":
        return None
    if setting not in ("auto", "cuda"):
        raise ValueError(f"estep_device must be auto|cpu|cuda (got {setting!r})")
    if setting in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[setting]

    torch_mod = None
    try:
        import torch as torch_mod  # noqa: F401
    except ImportError:
        pass

    backend = None
    if torch_mod is None:
        if setting != "auto":
            raise ValueError(
                f"estep_device={setting!r} requires PyTorch (pip install torch)"
            )
    elif setting == "cuda" or (setting == "auto" and torch_mod.cuda.is_available()):
        if not torch_mod.cuda.is_available():
            raise ValueError("estep_device='cuda' but torch reports no CUDA device")
        backend = TorchEstepBackend(torch_mod, "cuda")
    _RESOLVE_CACHE[setting] = backend
    return backend
