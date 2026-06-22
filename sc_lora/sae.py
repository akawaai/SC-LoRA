from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SAETrainLog:
    epoch: int
    l_rec: float
    l_sparse: float
    dead_feature_rate: float


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int = 4, normalize_inputs: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = input_dim * expansion_factor
        self.normalize_inputs = normalize_inputs

        self.encoder = nn.Linear(input_dim, self.feature_dim)
        self.decoder = nn.Linear(self.feature_dim, input_dim, bias=False)
        self.input_norm = nn.LayerNorm(input_dim)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)

    def _input_ref(self) -> torch.Tensor:
        # Encoder weights are the canonical dtype/device reference for SAE math.
        return self.encoder.weight

    def _match_input(self, x: torch.Tensor) -> torch.Tensor:
        ref = self._input_ref()
        if x.device != ref.device or x.dtype != ref.dtype:
            return x.to(device=ref.device, dtype=ref.dtype)
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return x
        ref = self.input_norm.weight
        if ref is not None and (x.device != ref.device or x.dtype != ref.dtype):
            x = x.to(device=ref.device, dtype=ref.dtype)
        return self.input_norm(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._normalize(self._match_input(x))
        z = F.relu(self.encoder(x_in))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat



class IdentitySparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, normalize_inputs: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = input_dim
        self.normalize_inputs = normalize_inputs
        self.input_norm = nn.LayerNorm(input_dim)

    def _input_ref(self) -> torch.Tensor:
        ref = self.input_norm.weight
        if ref is not None:
            return ref
        if self.input_norm.bias is not None:
            return self.input_norm.bias
        raise RuntimeError('IdentitySparseAutoencoder LayerNorm has no affine parameters to infer dtype/device from.')

    def _match_input(self, x: torch.Tensor) -> torch.Tensor:
        ref = self._input_ref()
        if x.device != ref.device or x.dtype != ref.dtype:
            return x.to(device=ref.device, dtype=ref.dtype)
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return self._match_input(x)
        x = self._match_input(x)
        return self.input_norm(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

class RandomProjectionSparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        normalize_inputs: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.normalize_inputs = normalize_inputs
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Linear(input_dim, feature_dim)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        weight = torch.randn(feature_dim, input_dim, generator=gen) / max(1.0, input_dim**0.5)
        bias = torch.zeros(feature_dim)

        with torch.no_grad():
            self.encoder.weight.copy_(weight)
            self.encoder.bias.copy_(bias)

        for p in self.encoder.parameters():
            p.requires_grad = False

    def _input_ref(self) -> torch.Tensor:
        ref = self.input_norm.weight
        if ref is not None:
            return ref
        if self.input_norm.bias is not None:
            return self.input_norm.bias
        return self.encoder.weight

    def _match_input(self, x: torch.Tensor) -> torch.Tensor:
        ref = self._input_ref()
        if x.device != ref.device or x.dtype != ref.dtype:
            return x.to(device=ref.device, dtype=ref.dtype)
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return self._match_input(x)
        x = self._match_input(x)
        return self.input_norm(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._normalize(x)
        ref = self.encoder.weight
        if x_in.device != ref.device or x_in.dtype != ref.dtype:
            x_in = x_in.to(device=ref.device, dtype=ref.dtype)
        return F.relu(self.encoder(x_in))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*z.shape[:-1], self.input_dim, device=z.device, dtype=z.dtype)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

class PermutedSparseAutoencoder(nn.Module):
    def __init__(self, base_sae: nn.Module, permutation: torch.Tensor):
        super().__init__()
        if permutation.ndim != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        self.base_sae = base_sae
        self.input_dim = int(getattr(base_sae, "input_dim"))
        self.feature_dim = int(getattr(base_sae, "feature_dim"))
        self.normalize_inputs = bool(getattr(base_sae, "normalize_inputs", True))
        self.register_buffer("permutation", permutation.long())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base_sae.encode(x)
        return z.index_select(dim=-1, index=self.permutation.to(z.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_sae.decode(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)

def sparse_penalty(z: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    if mode == "l1":
        return z.abs().mean()
    if mode == "hoyer":
        l1 = z.abs().sum(dim=-1)
        l2 = torch.norm(z, p=2, dim=-1) + 1e-8
        return (l1 / l2).mean()
    raise ValueError(f"Unknown sparse mode: {mode}")


def dead_feature_penalty(z: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    feature_mean = z.mean(dim=0)
    # Smooth surrogate: penalize only features whose mean activation falls below threshold.
    # This keeps gradients informative, unlike a hard indicator.
    return F.relu(threshold - feature_mean).mean()


def dead_feature_rate(z: torch.Tensor, threshold: float = 1e-4) -> float:
    feature_mean = z.mean(dim=0)
    return float((feature_mean < threshold).float().mean().item())


def decoder_norm_penalty(decoder_weight: torch.Tensor) -> torch.Tensor:
    # nn.Linear(feature_dim -> input_dim) stores weight as [input_dim, feature_dim].
    # Decoder atoms are columns (one per feature), so normalize along dim=0.
    norms = torch.norm(decoder_weight, dim=0)
    return ((norms - 1.0) ** 2).mean()


def train_single_sae(
    x: torch.Tensor,
    sae: SparseAutoencoder,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_sparse: float,
    lambda_aux: float,
    sparse_mode: str = "l1",
    dead_threshold: float = 1e-4,
    decoder_norm_weight: float = 1e-3,
    device: torch.device | str = "cpu",
) -> list[SAETrainLog]:
    sae = sae.to(device)
    sae.train()

    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)

    logs: list[SAETrainLog] = []

    for epoch in range(1, epochs + 1):
        rec_values: list[float] = []
        sparse_values: list[float] = []
        dead_values: list[float] = []

        for (batch_x,) in loader:
            ref = sae.encoder.weight
            batch_x = batch_x.to(device=ref.device, dtype=ref.dtype)
            z, x_hat = sae(batch_x)

            l_rec = F.mse_loss(x_hat, batch_x)
            l_sparse = sparse_penalty(z, mode=sparse_mode)
            l_aux = dead_feature_penalty(z, threshold=dead_threshold) + decoder_norm_weight * decoder_norm_penalty(
                sae.decoder.weight
            )
            loss = l_rec + lambda_sparse * l_sparse + lambda_aux * l_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            rec_values.append(float(l_rec.detach().cpu().item()))
            sparse_values.append(float(l_sparse.detach().cpu().item()))
            dead_values.append(dead_feature_rate(z.detach(), threshold=dead_threshold))

        logs.append(
            SAETrainLog(
                epoch=epoch,
                l_rec=sum(rec_values) / max(1, len(rec_values)),
                l_sparse=sum(sparse_values) / max(1, len(sparse_values)),
                dead_feature_rate=sum(dead_values) / max(1, len(dead_values)),
            )
        )

    return logs


def encode_with_sae(sae: SparseAutoencoder, x: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    sae.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i : i + chunk_size]
            outputs.append(sae.encode(chunk))
    return torch.cat(outputs, dim=0)
