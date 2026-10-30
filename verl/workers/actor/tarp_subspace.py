from __future__ import annotations

import contextlib
import gc
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
except Exception:  # pragma: no cover
    FSDP = None  # type: ignore

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except Exception:  # pragma: no cover
    StoppingCriteria = object  # type: ignore
    StoppingCriteriaList = list  # type: ignore


@dataclass
class MatBlockMeta:
    name: str
    param: torch.nn.Parameter
    out_dim: int
    in_dim: int
    block_size: int
    proj_dim: int


@dataclass
class VecMeta:
    name: str
    param: torch.nn.Parameter
    dim: int


class BlockExactSubspaceTARP:
    """Practical exact subspace TARP (block-wise):

    - Matrix params: block-wise covariance over output dimension, exact eigh per block.
    - Input dimension compressed by multiple Rademacher projections (proj_dim = d).
    - Vector params (bias, LayerNorm): diagonal Fisher.
    - CPU accumulation + optional GPU caching for training-time penalty.

    Notes on projection dimension d:
        d columns are treated as independent observations, no mean-collapsing.
        average_over_proj_dim controls only the scalar normalization (1/d).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        param_names: Optional[List[str]] = None,
        rank_per_block: int = 16,
        block_size: int = 1024,
        proj_dim: int = 8,
        seed: int = 42,
        cov_dtype: torch.dtype = torch.float32,
        store_dtype: torch.dtype = torch.float16,
        normalize_proj_columns: bool = True,
        average_over_proj_dim: bool = True,
        build_accumulators: bool = True,
    ):
        self.rank_per_block = int(rank_per_block)
        self.block_size = int(block_size)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)

        self.cov_dtype = cov_dtype
        self.store_dtype = store_dtype
        self.normalize_proj_columns = bool(normalize_proj_columns)
        self.average_over_proj_dim = bool(average_over_proj_dim)
        self._build_accumulators = bool(build_accumulators)

        named_params = [(n, p) for n, p in model.named_parameters() if getattr(p, "requires_grad", False)]
        if param_names is not None:
            allow = set(param_names)
            named_params = [(n, p) for n, p in named_params if n in allow]

        if not named_params:
            raise ValueError("No trainable parameters selected.")

        self.mat_metas: List[MatBlockMeta] = []
        self.vec_metas: List[VecMeta] = []

        # CPU storage
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.sample_count: int = 0

        # matrix params
        self.R_cpu: Dict[str, torch.Tensor] = {}  # (in_dim, d)
        self.cov_acc: Dict[str, List[torch.Tensor]] = {}  # list of (b,b)
        self.subspaces_cpu: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}  # list of (U(b,r), lam(r))

        # vector params
        self.fisher_diag_cpu: Dict[str, torch.Tensor] = {}

        # GPU caches (optional)
        self._cached_device: Optional[torch.device] = None
        self.R_gpu: Dict[str, torch.Tensor] = {}
        self.subspaces_gpu: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.fisher_diag_gpu: Dict[str, torch.Tensor] = {}
        self.theta_star_gpu: Dict[str, torch.Tensor] = {}

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        for name, p in named_params:
            shape = tuple(p.shape)
            if len(shape) >= 2:
                out_dim = int(shape[0])
                # flatten remaining dims as input dimension
                in_dim = int(torch.tensor(shape[1:]).prod().item())

                # Rademacher projection matrix R: (in_dim, d)
                R = torch.empty((in_dim, self.proj_dim), dtype=torch.float32, device="cpu")
                R.bernoulli_(0.5, generator=gen).mul_(2.0).sub_(1.0)
                R = R / (in_dim ** 0.5)
                self.R_cpu[name] = R

                if self._build_accumulators:
                    n_blocks = (out_dim + self.block_size - 1) // self.block_size
                    self.cov_acc[name] = []
                    for bi in range(n_blocks):
                        b0 = bi * self.block_size
                        b1 = min(out_dim, (bi + 1) * self.block_size)
                        b = b1 - b0
                        self.cov_acc[name].append(torch.zeros((b, b), dtype=self.cov_dtype, device="cpu"))
                else:
                    self.cov_acc[name] = []

                self.mat_metas.append(
                    MatBlockMeta(
                        name=name,
                        param=p,
                        out_dim=out_dim,
                        in_dim=in_dim,
                        block_size=self.block_size,
                        proj_dim=self.proj_dim,
                    )
                )
            else:
                dim = int(p.numel())
                self.vec_metas.append(VecMeta(name=name, param=p, dim=dim))
                self.fisher_diag_cpu[name] = torch.zeros((dim,), dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def snapshot_theta_star(self):
        self.theta_star = {}
        for m in self.mat_metas:
            self.theta_star[m.name] = m.param.detach().cpu().clone().to(dtype=self.store_dtype)
        for v in self.vec_metas:
            self.theta_star[v.name] = v.param.detach().cpu().clone().to(dtype=self.store_dtype)

        # invalidate GPU cache
        self._cached_device = None
        self.R_gpu.clear()
        self.subspaces_gpu.clear()
        self.fisher_diag_gpu.clear()
        self.theta_star_gpu.clear()

    @torch.no_grad()
    def accumulate_fisher_after_backward(self):
        """Call after backward to accumulate statistics.

        Accumulate:
          - vector params: diagonal fisher via grad^2
          - matrix params: block covariances using projected gradients (out_dim, d)
        """
        self.sample_count += 1

        # vector diagonal fisher
        for v in self.vec_metas:
            if v.param.grad is None:
                continue
            g = v.param.grad.detach().reshape(-1).float().cpu()
            self.fisher_diag_cpu[v.name] += g.pow(2)

        if not self._build_accumulators:
            # when used only for penalty, no need to accumulate
            return

        alpha_proj = (1.0 / float(self.proj_dim)) if self.average_over_proj_dim else 1.0

        # matrix block covariances
        for m in self.mat_metas:
            if m.param.grad is None:
                continue

            g = m.param.grad.detach()
            if g.numel() != m.out_dim * m.in_dim:
                # best effort: if shape mismatch (e.g., FSDP flat param), skip
                continue

            g2 = g.reshape(m.out_dim, m.in_dim)

            # project: (out, in) @ (in, d) -> (out, d)
            R = self.R_cpu[m.name].to(device=g2.device, dtype=g2.dtype, non_blocking=True)
            proj = torch.matmul(g2, R)  # (out_dim, d)

            proj_cpu = proj.float().cpu()

            if self.normalize_proj_columns:
                col_norm = proj_cpu.norm(dim=0, keepdim=True).clamp_min(1e-6)
                proj_cpu = proj_cpu / col_norm

            blocks = self.cov_acc.get(m.name, None)
            if blocks is None:
                continue

            n_blocks = len(blocks)
            for bi in range(n_blocks):
                b0 = bi * m.block_size
                b1 = min(m.out_dim, (bi + 1) * m.block_size)
                block = proj_cpu[b0:b1, :]  # (b, d)
                blocks[bi].addmm_(block, block.t(), beta=1.0, alpha=alpha_proj)

    @torch.no_grad()
    def finalize_fisher(self):
        if self.sample_count <= 0:
            raise RuntimeError("No samples accumulated.")

        # normalize vector fisher
        for name in list(self.fisher_diag_cpu.keys()):
            self.fisher_diag_cpu[name] = (self.fisher_diag_cpu[name] / float(self.sample_count)).to(
                dtype=self.store_dtype
            )

        if not self._build_accumulators:
            return

        # eigen per block covariance
        self.subspaces_cpu = {}
        for m in self.mat_metas:
            blocks = self.cov_acc.get(m.name, None)
            if not blocks:
                continue

            self.subspaces_cpu[m.name] = []
            for cov in blocks:
                cov = cov / float(self.sample_count)
                e, v = torch.linalg.eigh(cov)  # ascending
                r = min(self.rank_per_block, v.shape[1])
                top_e = e[-r:].flip(0).clamp_min(1e-6).to(dtype=self.store_dtype)
                top_v = v[:, -r:].flip(1).to(dtype=self.store_dtype)
                self.subspaces_cpu[m.name].append((top_v, top_e))

            # free cov blocks
            self.cov_acc[m.name] = []

        # release accumulators
        self.cov_acc = {}
        gc.collect()

        # invalidate GPU cache
        self._cached_device = None
        self.R_gpu.clear()
        self.subspaces_gpu.clear()
        self.fisher_diag_gpu.clear()
        self.theta_star_gpu.clear()

    @torch.no_grad()
    def cache_to_device(self, device: Union[str, torch.device], dtype: torch.dtype = torch.float16):
        """Cache R, U, lam, theta_star, fisher_diag to GPU to avoid per-step transfers."""
        dev = torch.device(device)
        self._cached_device = dev

        self.R_gpu = {m.name: self.R_cpu[m.name].to(device=dev, dtype=dtype, non_blocking=True) for m in self.mat_metas}

        self.subspaces_gpu = {}
        for name, blocks in self.subspaces_cpu.items():
            self.subspaces_gpu[name] = []
            for (U, lam) in blocks:
                self.subspaces_gpu[name].append(
                    (
                        U.to(device=dev, dtype=dtype, non_blocking=True),
                        lam.to(device=dev, dtype=dtype, non_blocking=True),
                    )
                )

        self.fisher_diag_gpu = {name: fd.to(device=dev, dtype=dtype, non_blocking=True) for name, fd in self.fisher_diag_cpu.items()}
        self.theta_star_gpu = {name: ts.to(device=dev, dtype=dtype, non_blocking=True) for name, ts in self.theta_star.items()}

    def penalty(self, coef: float = 1.0) -> torch.Tensor:
        """Differentiable penalty.

        - vectors: diagonal fisher penalty
        - matrices: Frobenius penalty over all d projected columns (no mean-collapsing)

        If cache_to_device() was called, this avoids CPU->GPU copies.
        """
        coef = float(coef)
        total: Optional[torch.Tensor] = None

        print(f"coef: {coef}")
        # vector diagonal penalty
        for v in self.vec_metas:
            p = v.param
            if self._cached_device is not None:
                ts = self.theta_star_gpu[v.name]
                fd = self.fisher_diag_gpu[v.name]
            else:
                ts = self.theta_star[v.name].to(p.device)
                fd = self.fisher_diag_cpu[v.name].to(p.device)

            delta = p.reshape(-1) - ts.to(dtype=p.dtype)
            term = 0.5 * (fd.to(dtype=delta.dtype) * delta.pow(2)).sum()
            total = term if total is None else (total + term)

        # matrix block subspace penalty
        alpha_proj = (1.0 / float(self.proj_dim)) if self.average_over_proj_dim else 1.0

        for m in self.mat_metas:
            name = m.name
            blocks = self.subspaces_gpu.get(name, None) if self._cached_device is not None else self.subspaces_cpu.get(name, None)
            if not blocks:
                continue

            p = m.param
            if self._cached_device is not None:
                ts = self.theta_star_gpu[name]
                R = self.R_gpu[name]
            else:
                ts = self.theta_star[name].to(p.device)
                R = self.R_cpu[name].to(p.device)

            delta = (p - ts.to(dtype=p.dtype)).reshape(m.out_dim, m.in_dim)
            delta_proj = torch.matmul(delta, R.to(dtype=delta.dtype))  # (out_dim, d)

            for bi, (U, lam) in enumerate(blocks):
                b0 = bi * m.block_size
                b1 = min(m.out_dim, (bi + 1) * m.block_size)
                Z = delta_proj[b0:b1, :]  # (b, d)

                U_ = U.to(dtype=Z.dtype)
                lam_ = lam.to(dtype=Z.dtype)

                P = torch.matmul(U_.transpose(0, 1), Z)  # (r, d)
                term = 0.5 * (lam_.unsqueeze(1) * P.pow(2)).sum() * alpha_proj
                total = term if total is None else (total + term)

        if total is None:
            # fallback scalar on the right device
            dev = self.mat_metas[0].param.device if self.mat_metas else self.vec_metas[0].param.device
            return torch.zeros((), device=dev)

        return total * coef

    def state_dict(self) -> Dict[str, Any]:
        """Pure-tensor state for saving/loading."""
        state: Dict[str, Any] = {
            "kind": "block_exact_subspace_tarp",
            "version": 1,
            "rank_per_block": self.rank_per_block,
            "block_size": self.block_size,
            "proj_dim": self.proj_dim,
            "seed": self.seed,
            "cov_dtype": str(self.cov_dtype),
            "store_dtype": str(self.store_dtype),
            "normalize_proj_columns": self.normalize_proj_columns,
            "average_over_proj_dim": self.average_over_proj_dim,
            "sample_count": int(self.sample_count),
            "theta_star": {k: v.cpu() for k, v in self.theta_star.items()},
            "R_cpu": {k: v.cpu() for k, v in self.R_cpu.items()},
            "subspaces_cpu": {
                k: [(U.cpu(), lam.cpu()) for (U, lam) in blocks]
                for k, blocks in self.subspaces_cpu.items()
            },
            "fisher_diag_cpu": {k: v.cpu() for k, v in self.fisher_diag_cpu.items()},
            "param_shapes": {m.name: tuple(m.param.shape) for m in self.mat_metas}
            | {v.name: tuple(v.param.shape) for v in self.vec_metas},
        }
        return state

    @classmethod
    def from_state_dict(
        cls,
        model: torch.nn.Module,
        state: Dict[str, Any],
        param_names: Optional[List[str]] = None,
    ) -> "BlockExactSubspaceTARP":
        if state.get("kind", None) != "block_exact_subspace_tarp":
            raise ValueError("Not a block_exact_subspace_tarp state.")

        obj = cls(
            model=model,
            param_names=param_names,
            rank_per_block=int(state.get("rank_per_block", 16)),
            block_size=int(state.get("block_size", 1024)),
            proj_dim=int(state.get("proj_dim", 8)),
            seed=int(state.get("seed", 42)),
            normalize_proj_columns=bool(state.get("normalize_proj_columns", True)),
            average_over_proj_dim=bool(state.get("average_over_proj_dim", True)),
            build_accumulators=False,
        )

        obj.sample_count = int(state.get("sample_count", 0))
        obj.theta_star = {k: v.cpu() for k, v in state.get("theta_star", {}).items()}
        obj.R_cpu = {k: v.cpu() for k, v in state.get("R_cpu", {}).items()}
        obj.subspaces_cpu = {
            k: [(U.cpu(), lam.cpu()) for (U, lam) in blocks]
            for k, blocks in state.get("subspaces_cpu", {}).items()
        }
        obj.fisher_diag_cpu = {k: v.cpu() for k, v in state.get("fisher_diag_cpu", {}).items()}

        return obj


# =========================
# TARP Weighter
# =========================

class EndAnswerCriteria(StoppingCriteria):
    def __init__(self, end_ids: List[int]):
        super().__init__()
        self.end_ids = list(end_ids)

    def __call__(self, seq_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if seq_ids is None or seq_ids.numel() == 0:
            return False
        if not self.end_ids:
            return False
        cur = seq_ids[0].tolist()
        n = len(self.end_ids)
        return len(cur) >= n and cur[-n:] == self.end_ids


class CTSWeighter:
    def __init__(
        self,
        sem_tau: float = 0.2,
        sem_power: float = 1.0,
        sem_agg: str = "mean",
        exclude_strings: Optional[List[str]] = None,
    ):
        self.sem_tau = float(sem_tau)
        self.sem_power = float(sem_power)
        self.sem_agg = str(sem_agg)
        self.exclude_strings = exclude_strings or [
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
            "<|im_start|>",
            "<|im_end|>",
            "<|",
            "|>",
            "\\",
            "\\(",
            "\\)",
            "\\[",
            "\\]",
        ]

    @staticmethod
    def _to_list_1d(x: Union[torch.Tensor, List[int]]) -> List[int]:
        if isinstance(x, list):
            return x
        return x.detach().cpu().tolist()

    @staticmethod
    def _extract_between(text: str, left: Optional[str], right: Optional[str]) -> str:
        if text is None:
            return ""
        if right:
            if right in text:
                text = text.split(right, 1)[0]
        if left:
            if left in text:
                text = text.split(left, 1)[1]
        return text

    @staticmethod
    def _trim_end_by_pad(gen_1d: torch.Tensor, pad_id: Optional[int]) -> int:
        if pad_id is None:
            return int(gen_1d.numel())
        if (gen_1d == pad_id).any():
            idx = (gen_1d == pad_id).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                return int(idx[0].item())
        return int(gen_1d.numel())

    @staticmethod
    def _find_subseq(haystack: List[int], needle: List[int]) -> Optional[int]:
        if not needle:
            return None
        n = len(needle)
        if len(haystack) < n:
            return None
        for i in range(0, len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                return i + n
        return None

    @staticmethod
    def _split_reason_sentences(text: str) -> List[str]:
        if not text:
            return []
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 2:
            return lines
        parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()] or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _locate_line_spans(tokenizer, full_ids_list: List[int], lines: List[str]) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        cursor = 0
        for ln in lines:
            ln_ids = tokenizer(ln, add_special_tokens=False)["input_ids"]
            if not ln_ids:
                continue
            start = -1
            max_i = len(full_ids_list) - len(ln_ids)
            for i in range(max(cursor, 0), max_i + 1):
                if full_ids_list[i : i + len(ln_ids)] == ln_ids:
                    start = i
                    break
            if start == -1:
                ln2 = re.sub(r"\s+", " ", ln).strip()
                ln_ids2 = tokenizer(ln2, add_special_tokens=False)["input_ids"]
                if ln_ids2:
                    max_i2 = len(full_ids_list) - len(ln_ids2)
                    for i in range(max(cursor, 0), max_i2 + 1):
                        if full_ids_list[i : i + len(ln_ids2)] == ln_ids2:
                            start = i
                            ln_ids = ln_ids2
                            break
            if start == -1:
                continue
            end = start + len(ln_ids)
            spans.append((start, end, ln))
            cursor = end
        return spans

    @staticmethod
    def _isin_mask(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        try:
            return torch.isin(x, values)
        except Exception:
            return (x.unsqueeze(-1) == values.view(1, 1, -1)).any(dim=-1)

    def _make_exclude_ids(self, tokenizer, device: torch.device) -> Optional[torch.Tensor]:
        ids: List[int] = []
        for s in self.exclude_strings:
            try:
                ids.extend(tokenizer.encode(s, add_special_tokens=False))
            except Exception:
                pass
        ids = list(set(ids))
        return torch.tensor(ids, device=device, dtype=torch.long) if ids else None

    @torch.no_grad()
    def _encode_sentence_embeds(self, actor_module: torch.nn.Module, seq_ids_2d: torch.Tensor, spans, exclude_ids=None) -> Optional[torch.Tensor]:
        dev = seq_ids_2d.device
        attn = torch.ones_like(seq_ids_2d, device=dev)
        out = actor_module(
            input_ids=seq_ids_2d,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states[-1].float()  # (1, T, H)

        embs: List[torch.Tensor] = []
        for (s, e, _) in spans:
            if e <= s:
                continue
            seg_ids = seq_ids_2d[:, s:e]
            if exclude_ids is not None and exclude_ids.numel() > 0:
                mask = (~self._isin_mask(seg_ids, exclude_ids)).float()
            else:
                mask = torch.ones_like(seg_ids, dtype=torch.float32)
            vec = hs[:, s:e, :]
            denom = mask.sum().clamp_min(1e-6)
            pooled = (vec * mask.unsqueeze(-1)).sum(dim=1) / denom
            embs.append(pooled.squeeze(0))

        if not embs:
            return None

        embs_t = torch.stack(embs, dim=0)
        return torch.nn.functional.normalize(embs_t, p=2, dim=-1)

    @staticmethod
    def _semantic_consistency(anchor_embs: torch.Tensor, other_embs_list: List[Optional[torch.Tensor]], agg: str = "mean") -> torch.Tensor:
        Sa = anchor_embs.size(0)
        sims: List[torch.Tensor] = []
        for embs in other_embs_list:
            if embs is None or embs.numel() == 0:
                continue
            sim = anchor_embs @ embs.t()  # (Sa, Sb)
            best = sim.max(dim=1).values.clamp(-1.0, 1.0)
            best01 = (best + 1.0) * 0.5
            sims.append(best01)
        if not sims:
            return torch.full((Sa,), 0.35, device=anchor_embs.device, dtype=torch.float32)
        stack = torch.stack(sims, dim=0)
        if agg == "min":
            sem = stack.min(dim=0).values
        elif agg == "max":
            sem = stack.max(dim=0).values
        else:
            sem = stack.mean(dim=0)
        return sem.clamp(0.0, 1.0)

    @torch.no_grad()
    def build_token_weights(
        self,
        actor_module: torch.nn.Module,
        tokenizer,
        anchor_ids_1d_cpu: torch.Tensor,
        prompt_len: int,
        pad_token_id: Optional[int],
        valid_start_idx: int,
        other_ids_1d_cpu_list: Optional[List[torch.Tensor]] = None,
        response_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, str]]]:
        """Build token weights for the (Prompt+Reasoning) truncated sequence.

        Returns:
            token_weight: shape (1, T-1) float32 on GPU, aligned with token_log_probs
            spans: sentence spans used (token index on full sequence)
        """
        dev = next(actor_module.parameters()).device

        # Determine effective sequence end (trim pad in generation part)
        prompt_len = int(prompt_len)
        gen_part = anchor_ids_1d_cpu[prompt_len:]
        rel_end = self._trim_end_by_pad(gen_part, pad_token_id)
        seq_end = prompt_len + rel_end
        seq_trimmed_cpu = anchor_ids_1d_cpu[:seq_end]

        # Prepare text for sentence splitting
        if response_text is None:
            resp_ids = seq_trimmed_cpu[prompt_len:]
            response_text = tokenizer.decode(resp_ids.tolist(), skip_special_tokens=True)

        reasoning_text = self._extract_between(response_text, "<think>", "</think>")
        if reasoning_text == response_text:
            # no tag, try remove answer area
            reasoning_text = self._extract_between(response_text, None, "<answer>")

        lines = self._split_reason_sentences(reasoning_text)
        full_ids_list = self._to_list_1d(seq_trimmed_cpu)
        spans = self._locate_line_spans(tokenizer, full_ids_list, lines)

        # token_weight lives on GPU
        seq_ids_gpu = seq_trimmed_cpu.unsqueeze(0).to(device=dev, non_blocking=True)
        T = seq_ids_gpu.size(1)
        token_weight = torch.zeros((1, max(T - 1, 1)), device=dev, dtype=torch.float32)

        # fallback: no spans => uniform weights on response tokens
        if not spans:
            # weights correspond to token_log_probs positions (t = 0..T-2) predicting token t+1
            s0 = max(0, min(valid_start_idx, T - 1))
            token_weight[:, s0:] = 1.0
            return token_weight, spans

        exclude_ids = self._make_exclude_ids(tokenizer, device=dev)

        # semantic consistency score per sentence
        anchor_embs = self._encode_sentence_embeds(actor_module, seq_ids_gpu, spans, exclude_ids=exclude_ids)
        other_embs_list: List[Optional[torch.Tensor]] = []
        if other_ids_1d_cpu_list:
            for other_cpu in other_ids_1d_cpu_list:
                other_gpu = other_cpu.unsqueeze(0).to(device=dev, non_blocking=True)
                other_txt = tokenizer.decode(other_cpu.tolist(), skip_special_tokens=True)
                other_reason = self._extract_between(other_txt, "<think>", "</think>")
                if other_reason == other_txt:
                    other_reason = self._extract_between(other_txt, None, "<answer>")
                other_lines = self._split_reason_sentences(other_reason)
                other_spans = self._locate_line_spans(tokenizer, self._to_list_1d(other_cpu), other_lines)
                if other_spans:
                    other_embs_list.append(self._encode_sentence_embeds(actor_module, other_gpu, other_spans, exclude_ids=exclude_ids))

        if anchor_embs is None:
            s0 = max(0, min(valid_start_idx, T - 1))
            token_weight[:, s0:] = 1.0
            return token_weight, spans

        sem_scores = self._semantic_consistency(anchor_embs, other_embs_list, agg=self.sem_agg)

        # compute stability from logits (detached) once
        out = actor_module(
            input_ids=seq_ids_gpu,
            attention_mask=torch.ones_like(seq_ids_gpu, device=dev),
            use_cache=False,
            return_dict=True,
        )
        logits_detached = out.logits[:, :-1, :].detach().float()

        # add sentence contributions
        Tm1 = token_weight.size(1)
        for si, (s_tok, e_tok, _) in enumerate(spans):
            if e_tok <= s_tok:
                continue
            sem_i = float(sem_scores[si].item())

            if other_ids_1d_cpu_list and len(other_ids_1d_cpu_list) >= 1:
                if sem_i <= self.sem_tau:
                    sem_i = 0.0
                else:
                    sem_i = (sem_i - self.sem_tau) / (1.0 - self.sem_tau + 1e-9)
            sem_i = max(0.0, min(1.0, sem_i)) ** self.sem_power

            # map token indices to logprob indices: token_log_probs[t] predicts token t+1
            t_start = max(0, s_tok - 1)
            t_end = max(0, e_tok - 1)
            t_start = min(t_start, Tm1)
            t_end = min(t_end, Tm1)
            if t_end <= t_start:
                continue

            logits_slice = logits_detached[:, t_start:t_end, :]
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1)
            stability = torch.exp(-ent).clamp(min=0.05, max=1.0)

            sent_token_ids = seq_ids_gpu[:, s_tok:e_tok]
            if exclude_ids is not None and exclude_ids.numel() > 0:
                valid_mask = (~self._isin_mask(sent_token_ids, exclude_ids)).float()
            else:
                valid_mask = torch.ones_like(sent_token_ids, dtype=torch.float32)

            L = t_end - t_start
            token_weight[:, t_start:t_end] += sem_i * stability[:, :L] * valid_mask[:, :L]

        # mask prompt contributions
        token_weight[:, : max(0, min(valid_start_idx, Tm1))] = 0.0

        return token_weight, spans


# =========================
# Fisher Estimation Orchestrator
# =========================

def _to_float_score(x: Any) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("score", "reward", "value", "acc", "correct"):
            if k in x:
                v = x[k]
                if isinstance(v, bool):
                    return 1.0 if v else 0.0
                if isinstance(v, (int, float)):
                    return float(v)
    return 0.0


@torch.no_grad()
def _reason_slice_with_prev(
    full_ids_1d_cpu: torch.Tensor,
    prompt_len: int,
    pad_token_id: int,
    end_think_ids: List[int],
    end_answer_ids: List[int],
) -> Tuple[Optional[int], Optional[int], Optional[torch.Tensor]]:
    gen_part = full_ids_1d_cpu[prompt_len:]
    rel_end = CTSWeighter._trim_end_by_pad(gen_part, pad_token_id)
    seq_end = prompt_len + rel_end

    gen_list = full_ids_1d_cpu[prompt_len:seq_end].tolist()
    think_end = CTSWeighter._find_subseq(gen_list, end_think_ids)
    ans_end = CTSWeighter._find_subseq(gen_list, end_answer_ids)

    reason_start = prompt_len
    if think_end is not None:
        reason_end = min(prompt_len + think_end, seq_end)
    elif ans_end is not None:
        reason_end = min(prompt_len + ans_end, seq_end)
    else:
        reason_end = seq_end

    if reason_end <= reason_start + 1:
        return None, None, None

    # include prompt: slice from 0
    return 0, reason_end, full_ids_1d_cpu[0:reason_end]


def _weighted_logprob_backward(
    actor_module: torch.nn.Module,
    seq_ids_2d: torch.Tensor,
    weight_2d: torch.Tensor,
    device_name: str,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Forward + backward of weighted joint logprob.

    weight_2d is float32, shape (1, T-1), aligned with token_log_probs.
    Returns float(valid_token_count).
    """
    dev = seq_ids_2d.device
    attn = torch.ones_like(seq_ids_2d, device=dev)

    actor_module.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device_name, dtype=autocast_dtype):
        out = actor_module(input_ids=seq_ids_2d, attention_mask=attn, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1, :]
        labels = seq_ids_2d[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        w = weight_2d.to(device=dev, dtype=torch.float32)
        valid_cnt = (w > 0).float().sum().clamp(min=1.0)
        obj = (token_log_probs.float() * w).sum() / valid_cnt

    if obj is not None and obj.requires_grad:
        obj.backward()

    return float(valid_cnt.item())


def run_tarp_subspace_fisher_estimation(
    actor_module: torch.nn.Module,
    config: Any,
    dataloader,
    tokenizer,
    save_dir: str,
    rollout=None,
    rollout_sharding_manager=None,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Estimate Fisher for TARP and store it as a BlockExactSubspaceTARP state.

    This function is verl-friendly and expects batch_dict compatible with DataProto.
    """
    from verl.utils.device import get_device_name, get_torch_device

    # reward scorers (verl)
    from verl.utils.reward_score.countdown import compute_score as compute_score_cd
    from verl.utils.reward_score.kk import compute_score as compute_score_kk
    from verl.utils.reward_score.coder1 import compute_score as compute_score_r1

    from verl import DataProto

    pid = os.getpid()
    os.makedirs(save_dir, exist_ok=True)

    tarp_mode = str(getattr(config, "tarp_mode", "TARP"))

    # block subspace hyper-params
    rank_per_block = int(getattr(config, "tarp_rank_per_block", 16))
    block_size = int(getattr(config, "tarp_block_size", 1024))
    proj_dim = int(getattr(config, "tarp_proj_dim", 8))
    seed = int(getattr(config, "tarp_seed", 42))

    # sampling/generation params
    micro_batch_size = int(getattr(config, "tarp_micro_batch_size", 1))
    max_new_tokens = int(getattr(config, "tarp_max_new_tokens", 1024))
    target_valid_samples = int(getattr(config, "tarp_target_valid_samples", 64))
    n_traj = int(getattr(config, "tarp_n_traj", 4))
    score_threshold = float(getattr(config, "tarp_score_threshold", 0.0))

    temperature = float(getattr(config, "tarp_temperature", 1.0))
    top_k = int(getattr(config, "tarp_top_k", 50))
    top_p = float(getattr(config, "tarp_top_p", 0.7))

    use_rollout_for_generation = int(getattr(config, "use_rollout_for_generation", 0))

    # TARP params
    sem_tau = float(getattr(config, "tarp_sem_tau", 0.2))
    sem_power = float(getattr(config, "tarp_sem_power", 1.0))
    sem_agg = str(getattr(config, "tarp_sem_agg", "mean"))

    weighter = CTSWeighter(sem_tau=sem_tau, sem_power=sem_power, sem_agg=sem_agg)

    dataset_name_hint = str(getattr(config, "dataset_name", "countdown")).lower()

    def _custom_reward_fn(resp_str: str, ground_truth: Any, extra_info: Any) -> float:
        if ground_truth is None:
            return 0.0
        try:
            if "countdown" in dataset_name_hint:
                return 1.0 if compute_score_cd(solution_str=resp_str, ground_truth=ground_truth, format_score=0) == 1 else 0.0
            if "math" in dataset_name_hint or "kk" in dataset_name_hint:
                return float(compute_score_kk(solution_str=resp_str, ground_truth=ground_truth, format_score=1))
            if "code" in dataset_name_hint:
                score = compute_score_r1(solution_str=resp_str, ground_truth=ground_truth, extra_info=extra_info)
                return 1.0 if score == 1.1 else 0.0
            return 1.0 if compute_score_cd(solution_str=resp_str, ground_truth=ground_truth, format_score=0) == 1 else 0.0
        except Exception:
            return 0.0

    # init subspace TARP accumulator
    sub_tarp = BlockExactSubspaceTARP(
        model=actor_module,
        param_names=param_names,
        rank_per_block=rank_per_block,
        block_size=block_size,
        proj_dim=proj_dim,
        seed=seed,
        cov_dtype=torch.float32,
        store_dtype=torch.float16,
        normalize_proj_columns=bool(getattr(config, "tarp_normalize_proj_columns", True)),
        average_over_proj_dim=bool(getattr(config, "tarp_average_over_proj_dim", True)),
        build_accumulators=True,
    )
    sub_tarp.snapshot_theta_star()

    # token ids
    eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = 0

    end_answer_ids = tokenizer("</answer>", add_special_tokens=False)["input_ids"]
    end_think_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    stopping_criteria = StoppingCriteriaList([EndAnswerCriteria(end_answer_ids)])

    device_name = get_device_name()
    dev = get_torch_device().current_device()

    actor_module.eval()

    step_count = 0
    valid_samples_count = 0

    with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        for i, batch_dict in enumerate(dataloader):
            if valid_samples_count >= target_valid_samples:
                break

            original_batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_keys = ["raw_prompt_ids"]
            gen_batch = original_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_keys)
            batch_data = {**gen_batch.batch.to(device=dev), **gen_batch.non_tensor_batch}

            batch_size = batch_data["input_ids"].size(0)

            for start_idx in range(0, batch_size, micro_batch_size):
                if valid_samples_count >= target_valid_samples:
                    break

                end_idx = start_idx + micro_batch_size

                seq_prompt_ids = batch_data["input_ids"][start_idx:end_idx]
                seq_attn = batch_data["attention_mask"][start_idx:end_idx]
                prompt_len = int(seq_prompt_ids.size(1))

                # reward_model extraction
                reward_model_item = None
                ground_truth = None
                cur_extra_info = None
                if "reward_model" in original_batch.non_tensor_batch:
                    cur_rm = original_batch.non_tensor_batch["reward_model"][start_idx:end_idx]
                    if len(cur_rm) > 0:
                        reward_model_item = cur_rm[0]
                        if isinstance(reward_model_item, dict):
                            ground_truth = reward_model_item.get("ground_truth", None)
                if "extra_info" in original_batch.non_tensor_batch:
                    cur_extra = original_batch.non_tensor_batch["extra_info"][start_idx:end_idx]
                    if len(cur_extra) > 0:
                        cur_extra_info = cur_extra[0]

                # Step 1: generate trajectories
                all_full_ids_cpu: List[torch.Tensor] = []
                all_resp_str: List[str] = []
                all_scores: List[float] = []
                correct_indices: List[int] = []

                if use_rollout_for_generation:
                    raise NotImplementedError("use_rollout_for_generation is not implemented in this helper.")

                is_fsdp = (FSDP is not None) and isinstance(actor_module, FSDP)
                param_ctx = (
                    FSDP.summon_full_params(actor_module, writeback=False, recurse=False)
                    if is_fsdp
                    else contextlib.nullcontext()
                )

                with param_ctx, torch.no_grad():
                    gen_cfg = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "do_sample": True,
                        "num_beams": 1,
                        "num_return_sequences": n_traj,
                        "pad_token_id": pad_token_id,
                        "eos_token_id": eos_token_id,
                        "use_cache": False,
                        "return_dict_in_generate": True,
                    }
                    out = actor_module.generate(
                        input_ids=seq_prompt_ids,
                        attention_mask=seq_attn,
                        stopping_criteria=stopping_criteria,
                        **gen_cfg,
                    )
                    seqs = out.sequences.detach().cpu()

                for k in range(n_traj):
                    full_ids = seqs[k]
                    all_full_ids_cpu.append(full_ids)
                    gen_ids = full_ids[prompt_len:]
                    resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    all_resp_str.append(resp)
                    sc = float(_custom_reward_fn(resp, ground_truth=ground_truth, extra_info=cur_extra_info))
                    all_scores.append(sc)
                    if sc > score_threshold:
                        correct_indices.append(k)

                del seqs
                torch.cuda.empty_cache()

                if len(correct_indices) == 0:
                    del all_full_ids_cpu, all_resp_str, all_scores, correct_indices
                    gc.collect()
                    continue

                anchor_k = correct_indices[0]

                # Step 2: slice reasoning region (with prompt)
                a_s, a_e, anchor_with_prev_cpu = _reason_slice_with_prev(
                    all_full_ids_cpu[anchor_k],
                    prompt_len=prompt_len,
                    pad_token_id=pad_token_id,
                    end_think_ids=end_think_ids,
                    end_answer_ids=end_answer_ids,
                )
                if anchor_with_prev_cpu is None:
                    del all_full_ids_cpu, all_resp_str, all_scores, correct_indices
                    gc.collect()
                    continue

                other_with_prev_cpu_list: List[torch.Tensor] = []
                for k in correct_indices:
                    if k == anchor_k:
                        continue
                    _, _, ids_k = _reason_slice_with_prev(
                        all_full_ids_cpu[k],
                        prompt_len=prompt_len,
                        pad_token_id=pad_token_id,
                        end_think_ids=end_think_ids,
                        end_answer_ids=end_answer_ids,
                    )
                    if ids_k is not None:
                        other_with_prev_cpu_list.append(ids_k)

                valid_samples_count += 1

                # Step 3: Fisher accumulation for TARP
                valid_start_idx = max(0, prompt_len - 1)

                if tarp_mode.upper() != "TARP":
                    raise ValueError(f"Unknown tarp_mode: {tarp_mode}")

                if len(correct_indices) < 2:
                    seq_ids_gpu = anchor_with_prev_cpu.unsqueeze(0).to(device=dev, non_blocking=True)
                    T = int(seq_ids_gpu.size(1))
                    weight = torch.zeros((1, max(T - 1, 1)), device=dev, dtype=torch.float32)
                    weight[:, max(0, min(valid_start_idx, T - 1)) :] = 1.0
                    _weighted_logprob_backward(actor_module, seq_ids_gpu, weight, device_name=device_name)
                else:
                    resp_text = all_resp_str[anchor_k]
                    token_weight, _ = weighter.build_token_weights(
                        actor_module=actor_module,
                        tokenizer=tokenizer,
                        anchor_ids_1d_cpu=anchor_with_prev_cpu,
                        prompt_len=prompt_len,
                        pad_token_id=pad_token_id,
                        valid_start_idx=valid_start_idx,
                        other_ids_1d_cpu_list=other_with_prev_cpu_list,
                        response_text=resp_text,
                    )
                    seq_ids_gpu = anchor_with_prev_cpu.unsqueeze(0).to(device=dev, non_blocking=True)
                    _weighted_logprob_backward(actor_module, seq_ids_gpu, token_weight, device_name=device_name)

                # accumulate and clear grads
                sub_tarp.accumulate_fisher_after_backward()
                actor_module.zero_grad(set_to_none=True)

                step_count += 1

                del all_full_ids_cpu, all_resp_str, all_scores, correct_indices
                gc.collect()
                torch.cuda.empty_cache()

    sub_tarp.finalize_fisher()

    state = sub_tarp.state_dict()
    state["tarp_mode"] = tarp_mode
    state["valid_samples_used"] = int(valid_samples_count)
    state["traj_per_prompt"] = int(n_traj)
    state["score_threshold"] = float(score_threshold)

    save_path = os.path.join(save_dir, f"Seq_TARP_Subspace_{tarp_mode}_valid{target_valid_samples}.pt")
    torch.save(state, save_path)

    print(
        f"[TARP-Subspace pid={pid}] saved state: mode={tarp_mode}, valid={valid_samples_count}/{target_valid_samples}, steps={step_count}, path={save_path}",
        flush=True,
    )

    return state
