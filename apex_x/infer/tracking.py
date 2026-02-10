from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.nn import functional as f


def _empty_long_tensor(device: torch.device) -> Tensor:
    return torch.zeros((0,), dtype=torch.int64, device=device)


def _box_iou_matrix(
    track_boxes_xyxy: Tensor, detection_boxes_xyxy: Tensor, *, eps: float = 1e-9
) -> Tensor:
    """Compute pairwise IoU matrix [T, N] between track and detection boxes."""
    if track_boxes_xyxy.ndim != 2 or track_boxes_xyxy.shape[1] != 4:
        raise ValueError("track_boxes_xyxy must be [T,4]")
    if detection_boxes_xyxy.ndim != 2 or detection_boxes_xyxy.shape[1] != 4:
        raise ValueError("detection_boxes_xyxy must be [N,4]")
    if track_boxes_xyxy.shape[0] == 0 or detection_boxes_xyxy.shape[0] == 0:
        return torch.zeros(
            (track_boxes_xyxy.shape[0], detection_boxes_xyxy.shape[0]),
            dtype=track_boxes_xyxy.dtype,
            device=track_boxes_xyxy.device,
        )

    tl = torch.maximum(track_boxes_xyxy[:, None, :2], detection_boxes_xyxy[None, :, :2])
    br = torch.minimum(track_boxes_xyxy[:, None, 2:], detection_boxes_xyxy[None, :, 2:])
    wh = (br - tl).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    track_wh = (track_boxes_xyxy[:, 2:] - track_boxes_xyxy[:, :2]).clamp(min=0.0)
    det_wh = (detection_boxes_xyxy[:, 2:] - detection_boxes_xyxy[:, :2]).clamp(min=0.0)
    track_area = track_wh[:, 0] * track_wh[:, 1]
    det_area = det_wh[:, 0] * det_wh[:, 1]

    union = track_area[:, None] + det_area[None, :] - inter
    return inter / (union + eps)


def _hungarian_solve_dense(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Hungarian algorithm for rectangular cost matrix (minimization)."""
    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return []

    transposed = False
    work = cost_matrix
    n = rows
    m = cols
    if n > m:
        transposed = True
        work = [[cost_matrix[r][c] for r in range(rows)] for c in range(cols)]
        n, m = m, n

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = work[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta or (minv[j] == delta and j < j1):
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] == 0:
            continue
        row = p[j] - 1
        col = j - 1
        if transposed:
            assignment.append((col, row))
        else:
            assignment.append((row, col))
    assignment.sort(key=lambda pair: (pair[0], pair[1]))
    return assignment


def hungarian_assignment(cost_matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Return row/column assignment minimizing global cost."""
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be [R,C]")
    if not torch.isfinite(cost_matrix).all():
        raise ValueError("cost_matrix must be finite")

    rows = int(cost_matrix.shape[0])
    cols = int(cost_matrix.shape[1])
    if rows == 0 or cols == 0:
        device = cost_matrix.device
        return _empty_long_tensor(device), _empty_long_tensor(device)

    pairs = _hungarian_solve_dense(cost_matrix.detach().cpu().tolist())
    device = cost_matrix.device
    out_rows = torch.tensor([pair[0] for pair in pairs], dtype=torch.int64, device=device)
    out_cols = torch.tensor([pair[1] for pair in pairs], dtype=torch.int64, device=device)
    return out_rows, out_cols


@dataclass(frozen=True, slots=True)
class TrackState:
    """Track state contract for frame-to-frame association."""

    track_ids: Tensor  # [T] int64
    embeddings: Tensor  # [T,D] float
    boxes_xyxy: Tensor  # [T,4] float
    scores: Tensor  # [T] float
    ages: Tensor  # [T] int64
    frame_index: int
    hit_counts: Tensor | None = None  # [T] int64
    memory_bank: Tensor | None = None  # [T,M,D] float
    memory_counts: Tensor | None = None  # [T] int64

    def __post_init__(self) -> None:
        if self.track_ids.ndim != 1:
            raise ValueError("track_ids must be [T]")
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be [T,D]")
        if self.boxes_xyxy.ndim != 2 or self.boxes_xyxy.shape[1] != 4:
            raise ValueError("boxes_xyxy must be [T,4]")
        if self.scores.ndim != 1:
            raise ValueError("scores must be [T]")
        if self.ages.ndim != 1:
            raise ValueError("ages must be [T]")

        tracks = int(self.track_ids.shape[0])
        if self.embeddings.shape[0] != tracks:
            raise ValueError("embeddings first dim must match number of tracks")
        if self.boxes_xyxy.shape[0] != tracks:
            raise ValueError("boxes first dim must match number of tracks")
        if self.scores.shape[0] != tracks:
            raise ValueError("scores dim must match number of tracks")
        if self.ages.shape[0] != tracks:
            raise ValueError("ages dim must match number of tracks")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")

        device = self.track_ids.device
        if self.embeddings.device != device:
            raise ValueError("embeddings must share device with track_ids")
        if self.boxes_xyxy.device != device:
            raise ValueError("boxes_xyxy must share device with track_ids")
        if self.scores.device != device:
            raise ValueError("scores must share device with track_ids")
        if self.ages.device != device:
            raise ValueError("ages must share device with track_ids")

        if self.track_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("track_ids must be integer typed")
        if self.ages.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("ages must be integer typed")

        if torch.any(self.track_ids < 0):
            raise ValueError("track_ids must be non-negative")
        if torch.any(self.ages < 0):
            raise ValueError("ages must be non-negative")
        if not torch.isfinite(self.embeddings).all():
            raise ValueError("embeddings must be finite")
        if not torch.isfinite(self.boxes_xyxy).all():
            raise ValueError("boxes_xyxy must be finite")
        if not torch.isfinite(self.scores).all():
            raise ValueError("scores must be finite")

        if self.hit_counts is None:
            default_hits = torch.ones((tracks,), dtype=torch.int64, device=device)
            object.__setattr__(self, "hit_counts", default_hits)
        else:
            if self.hit_counts.ndim != 1 or self.hit_counts.shape[0] != tracks:
                raise ValueError("hit_counts must be [T]")
            if self.hit_counts.device != device:
                raise ValueError("hit_counts must share device with track_ids")
            if self.hit_counts.dtype not in {torch.int16, torch.int32, torch.int64, torch.uint8}:
                raise ValueError("hit_counts must be integer typed")
            if torch.any(self.hit_counts < 0):
                raise ValueError("hit_counts must be non-negative")

        if self.memory_bank is None:
            emb_dim = max(int(self.embeddings.shape[1]), 1)
            bank = torch.zeros((tracks, 1, emb_dim), dtype=torch.float32, device=device)
            if tracks > 0 and int(self.embeddings.shape[1]) > 0:
                bank[:, 0, :] = f.normalize(self.embeddings, p=2.0, dim=1, eps=1e-6)
            object.__setattr__(self, "memory_bank", bank)
        else:
            if self.memory_bank.ndim != 3:
                raise ValueError("memory_bank must be [T,M,D]")
            if self.memory_bank.shape[0] != tracks:
                raise ValueError("memory_bank first dim must match number of tracks")
            if self.memory_bank.device != device:
                raise ValueError("memory_bank must share device with track_ids")
            if self.memory_bank.shape[2] != self.embedding_dim:
                raise ValueError("memory_bank last dim must match embedding dim")
            if self.memory_bank.shape[1] <= 0:
                raise ValueError("memory_bank must have positive bank size")
            if not torch.isfinite(self.memory_bank).all():
                raise ValueError("memory_bank must be finite")

        current_bank = self.memory_bank
        if current_bank is None:
            raise RuntimeError("memory_bank was not initialized")
        bank_size = int(current_bank.shape[1])

        if self.memory_counts is None:
            if tracks == 0:
                counts = torch.zeros((0,), dtype=torch.int64, device=device)
            else:
                counts = torch.ones((tracks,), dtype=torch.int64, device=device)
            object.__setattr__(self, "memory_counts", counts)
        else:
            if self.memory_counts.ndim != 1 or self.memory_counts.shape[0] != tracks:
                raise ValueError("memory_counts must be [T]")
            if self.memory_counts.device != device:
                raise ValueError("memory_counts must share device with track_ids")
            if self.memory_counts.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError("memory_counts must be integer typed")
            if torch.any(self.memory_counts < 0):
                raise ValueError("memory_counts must be non-negative")
            if torch.any(self.memory_counts > bank_size):
                raise ValueError("memory_counts must be <= bank size")

    @property
    def num_tracks(self) -> int:
        return int(self.track_ids.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    @property
    def next_track_id(self) -> int:
        if self.num_tracks == 0:
            return 0
        return int(torch.max(self.track_ids).item()) + 1

    @property
    def bank_size(self) -> int:
        current_bank = self.memory_bank
        if current_bank is None:
            return 0
        return int(current_bank.shape[1])

    def association_embeddings(self) -> Tensor:
        """Embedding references for association (mean over memory-bank history)."""
        if self.num_tracks == 0:
            return self.embeddings

        bank = self.memory_bank
        counts = self.memory_counts
        if bank is None or counts is None:
            return f.normalize(self.embeddings, p=2.0, dim=1, eps=1e-6)

        refs: list[Tensor] = []
        normalized_current = f.normalize(self.embeddings, p=2.0, dim=1, eps=1e-6)
        for idx in range(self.num_tracks):
            count = int(counts[idx].item())
            if count <= 0:
                refs.append(normalized_current[idx : idx + 1])
                continue
            refs.append(bank[idx, :count].mean(dim=0, keepdim=True))
        stacked = torch.cat(refs, dim=0)
        return f.normalize(stacked, p=2.0, dim=1, eps=1e-6)

    @classmethod
    def empty(
        cls,
        *,
        embedding_dim: int,
        frame_index: int = 0,
        device: torch.device | None = None,
        bank_size: int = 1,
    ) -> TrackState:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if bank_size <= 0:
            raise ValueError("bank_size must be > 0")
        dev = torch.device("cpu") if device is None else device
        return cls(
            track_ids=torch.zeros((0,), dtype=torch.int64, device=dev),
            embeddings=torch.zeros((0, embedding_dim), dtype=torch.float32, device=dev),
            boxes_xyxy=torch.zeros((0, 4), dtype=torch.float32, device=dev),
            scores=torch.zeros((0,), dtype=torch.float32, device=dev),
            ages=torch.zeros((0,), dtype=torch.int64, device=dev),
            frame_index=frame_index,
            hit_counts=torch.zeros((0,), dtype=torch.int64, device=dev),
            memory_bank=torch.zeros((0, bank_size, embedding_dim), dtype=torch.float32, device=dev),
            memory_counts=torch.zeros((0,), dtype=torch.int64, device=dev),
        )


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Association result between detections and prior tracks."""

    matched_detection_indices: Tensor  # [M] int64
    matched_track_indices: Tensor  # [M] int64 indices into previous state
    new_detection_indices: Tensor  # [K] int64
    next_state: TrackState
    terminated_track_indices: Tensor = field(
        default_factory=lambda: torch.zeros((0,), dtype=torch.int64)
    )
    terminated_track_ids: Tensor = field(
        default_factory=lambda: torch.zeros((0,), dtype=torch.int64)
    )
    created_track_ids: Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.int64))


@runtime_checkable
class AssociationProtocol(Protocol):
    """Association interface for tracker state updates."""

    def associate(
        self,
        detection_boxes_xyxy: Tensor,  # [N,4]
        detection_embeddings: Tensor,  # [N,D]
        detection_scores: Tensor,  # [N]
        state: TrackState | None,
        *,
        frame_index: int,
    ) -> AssociationResult:
        """Associate detections to existing tracks and return next state."""
        ...


class HungarianAssociator(AssociationProtocol):
    """Deterministic Hungarian association with IoU + embedding-distance gating."""

    def __init__(
        self,
        *,
        iou_gate: float = 0.0,
        embedding_distance_gate: float = 0.7,
        iou_weight: float = 0.5,
        embedding_weight: float = 0.5,
        score_weight: float = 0.0,
        max_age: int = 30,
        memory_bank_size: int = 8,
        invalid_cost: float = 1e6,
    ) -> None:
        if not (-1.0 <= iou_gate <= 1.0):
            raise ValueError("iou_gate must be within [-1, 1]")
        if embedding_distance_gate < 0.0:
            raise ValueError("embedding_distance_gate must be >= 0")
        if iou_weight < 0.0:
            raise ValueError("iou_weight must be >= 0")
        if embedding_weight < 0.0:
            raise ValueError("embedding_weight must be >= 0")
        if score_weight < 0.0:
            raise ValueError("score_weight must be >= 0")
        if iou_weight + embedding_weight + score_weight <= 0.0:
            raise ValueError("at least one cost weight must be > 0")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if memory_bank_size <= 0:
            raise ValueError("memory_bank_size must be > 0")
        if invalid_cost <= 0.0:
            raise ValueError("invalid_cost must be > 0")

        self.iou_gate = float(iou_gate)
        self.embedding_distance_gate = float(embedding_distance_gate)
        self.iou_weight = float(iou_weight)
        self.embedding_weight = float(embedding_weight)
        self.score_weight = float(score_weight)
        self.max_age = int(max_age)
        self.memory_bank_size = int(memory_bank_size)
        self.invalid_cost = float(invalid_cost)

    def _validate_detections(
        self,
        detection_boxes_xyxy: Tensor,
        detection_embeddings: Tensor,
        detection_scores: Tensor,
    ) -> tuple[int, int]:
        if detection_boxes_xyxy.ndim != 2 or detection_boxes_xyxy.shape[1] != 4:
            raise ValueError("detection_boxes_xyxy must be [N,4]")
        if detection_embeddings.ndim != 2:
            raise ValueError("detection_embeddings must be [N,D]")
        if detection_scores.ndim != 1:
            raise ValueError("detection_scores must be [N]")

        n = int(detection_boxes_xyxy.shape[0])
        if detection_embeddings.shape[0] != n or detection_scores.shape[0] != n:
            raise ValueError("detection tensors must share first dimension N")
        if n > 0 and not torch.isfinite(detection_boxes_xyxy).all():
            raise ValueError("detection_boxes_xyxy must be finite")
        if n > 0 and not torch.isfinite(detection_embeddings).all():
            raise ValueError("detection_embeddings must be finite")
        if n > 0 and not torch.isfinite(detection_scores).all():
            raise ValueError("detection_scores must be finite")

        dim = int(detection_embeddings.shape[1])
        return n, dim

    def _pairwise_cost(
        self,
        prev_state: TrackState,
        detection_boxes_xyxy: Tensor,
        detection_embeddings: Tensor,
        detection_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        det_norm = f.normalize(detection_embeddings, p=2.0, dim=1, eps=1e-6)
        if prev_state.num_tracks == 0 or detection_embeddings.shape[0] == 0:
            empty = detection_embeddings.new_zeros(
                (prev_state.num_tracks, detection_embeddings.shape[0])
            )
            return empty, det_norm

        track_ref = prev_state.association_embeddings()
        cosine = (track_ref @ det_norm.t()).clamp(min=-1.0, max=1.0)
        emb_distance = (1.0 - cosine).clamp(min=0.0)
        iou = _box_iou_matrix(prev_state.boxes_xyxy, detection_boxes_xyxy).to(
            dtype=emb_distance.dtype
        )

        gate = emb_distance <= self.embedding_distance_gate
        if self.iou_gate >= 0.0:
            gate = gate & (iou >= self.iou_gate)

        score_term = (1.0 - detection_scores.clamp(min=0.0, max=1.0)).unsqueeze(0)
        cost = (
            self.iou_weight * (1.0 - iou)
            + self.embedding_weight * emb_distance
            + self.score_weight * score_term
        )
        masked = torch.full_like(cost, self.invalid_cost)
        return torch.where(gate, cost, masked), det_norm

    def _append_memory(
        self,
        previous_bank: Tensor,
        *,
        previous_count: int,
        previous_hits: int,
        new_embedding: Tensor,
    ) -> tuple[Tensor, int]:
        bank_size = int(previous_bank.shape[0])
        updated = previous_bank.clone()
        write_pos = previous_count if previous_count < bank_size else previous_hits % bank_size
        updated[write_pos] = new_embedding
        next_count = min(previous_count + 1, bank_size)
        return updated, next_count

    def _resize_memory_bank(self, state: TrackState) -> TrackState:
        if state.bank_size == self.memory_bank_size:
            return state

        tracks = state.num_tracks
        emb_dim = state.embedding_dim
        device = state.track_ids.device
        if tracks == 0:
            return TrackState.empty(
                embedding_dim=max(emb_dim, 1),
                frame_index=state.frame_index,
                device=device,
                bank_size=self.memory_bank_size,
            )

        bank = state.memory_bank
        counts = state.memory_counts
        hits = state.hit_counts
        if bank is None or counts is None or hits is None:
            raise RuntimeError("track state memory fields are not initialized")

        resized = torch.zeros(
            (tracks, self.memory_bank_size, emb_dim),
            dtype=bank.dtype,
            device=device,
        )
        copy_size = min(int(bank.shape[1]), self.memory_bank_size)
        if copy_size > 0:
            resized[:, :copy_size, :] = bank[:, :copy_size, :]
        resized_counts = counts.clamp(min=0, max=self.memory_bank_size)
        return TrackState(
            track_ids=state.track_ids,
            embeddings=state.embeddings,
            boxes_xyxy=state.boxes_xyxy,
            scores=state.scores,
            ages=state.ages,
            frame_index=state.frame_index,
            hit_counts=hits,
            memory_bank=resized,
            memory_counts=resized_counts,
        )

    def associate(
        self,
        detection_boxes_xyxy: Tensor,
        detection_embeddings: Tensor,
        detection_scores: Tensor,
        state: TrackState | None,
        *,
        frame_index: int,
    ) -> AssociationResult:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        n_dets, emb_dim = self._validate_detections(
            detection_boxes_xyxy,
            detection_embeddings,
            detection_scores,
        )
        device = detection_embeddings.device

        if state is None:
            prev_state = TrackState.empty(
                embedding_dim=max(emb_dim, 1),
                frame_index=max(0, frame_index - 1),
                device=device,
                bank_size=self.memory_bank_size,
            )
        else:
            prev_state = state
            if prev_state.num_tracks > 0 and prev_state.embedding_dim != emb_dim:
                raise ValueError("detection embedding dim must match track state embedding dim")
            prev_state = self._resize_memory_bank(prev_state)

        if n_dets == 0 and prev_state.num_tracks == 0:
            return AssociationResult(
                matched_detection_indices=_empty_long_tensor(device),
                matched_track_indices=_empty_long_tensor(device),
                new_detection_indices=_empty_long_tensor(device),
                next_state=TrackState.empty(
                    embedding_dim=max(
                        emb_dim, prev_state.embedding_dim if prev_state.embedding_dim > 0 else 1
                    ),
                    frame_index=frame_index,
                    device=device,
                    bank_size=self.memory_bank_size,
                ),
                terminated_track_indices=_empty_long_tensor(device),
                terminated_track_ids=_empty_long_tensor(device),
                created_track_ids=_empty_long_tensor(device),
            )

        if prev_state.num_tracks == 0:
            det_norm = f.normalize(detection_embeddings, p=2.0, dim=1, eps=1e-6)
            track_ids = torch.arange(n_dets, dtype=torch.int64, device=device)
            memory_bank = torch.zeros(
                (n_dets, self.memory_bank_size, emb_dim),
                dtype=torch.float32,
                device=device,
            )
            if n_dets > 0:
                memory_bank[:, 0, :] = det_norm
            next_state = TrackState(
                track_ids=track_ids,
                embeddings=det_norm,
                boxes_xyxy=detection_boxes_xyxy,
                scores=detection_scores,
                ages=torch.zeros((n_dets,), dtype=torch.int64, device=device),
                frame_index=frame_index,
                hit_counts=torch.ones((n_dets,), dtype=torch.int64, device=device),
                memory_bank=memory_bank,
                memory_counts=torch.ones((n_dets,), dtype=torch.int64, device=device),
            )
            return AssociationResult(
                matched_detection_indices=_empty_long_tensor(device),
                matched_track_indices=_empty_long_tensor(device),
                new_detection_indices=torch.arange(n_dets, dtype=torch.int64, device=device),
                next_state=next_state,
                terminated_track_indices=_empty_long_tensor(device),
                terminated_track_ids=_empty_long_tensor(device),
                created_track_ids=track_ids,
            )

        if n_dets == 0:
            terminated_indices: list[int] = []
            terminated_ids: list[int] = []
            keep_mask = (prev_state.ages + 1) <= self.max_age
            for idx in range(prev_state.num_tracks):
                if bool(keep_mask[idx].item()):
                    continue
                terminated_indices.append(idx)
                terminated_ids.append(int(prev_state.track_ids[idx].item()))

            next_state = TrackState(
                track_ids=prev_state.track_ids[keep_mask],
                embeddings=prev_state.embeddings[keep_mask],
                boxes_xyxy=prev_state.boxes_xyxy[keep_mask],
                scores=prev_state.scores[keep_mask],
                ages=prev_state.ages[keep_mask] + 1,
                frame_index=frame_index,
                hit_counts=(
                    prev_state.hit_counts[keep_mask]
                    if prev_state.hit_counts is not None
                    else torch.zeros((0,), dtype=torch.int64, device=device)
                ),
                memory_bank=(
                    prev_state.memory_bank[keep_mask]
                    if prev_state.memory_bank is not None
                    else torch.zeros(
                        (0, self.memory_bank_size, emb_dim), dtype=torch.float32, device=device
                    )
                ),
                memory_counts=(
                    prev_state.memory_counts[keep_mask]
                    if prev_state.memory_counts is not None
                    else torch.zeros((0,), dtype=torch.int64, device=device)
                ),
            )
            return AssociationResult(
                matched_detection_indices=_empty_long_tensor(device),
                matched_track_indices=_empty_long_tensor(device),
                new_detection_indices=_empty_long_tensor(device),
                next_state=next_state,
                terminated_track_indices=torch.tensor(
                    terminated_indices,
                    dtype=torch.int64,
                    device=device,
                ),
                terminated_track_ids=torch.tensor(
                    terminated_ids,
                    dtype=torch.int64,
                    device=device,
                ),
                created_track_ids=_empty_long_tensor(device),
            )

        cost, det_norm = self._pairwise_cost(
            prev_state,
            detection_boxes_xyxy,
            detection_embeddings,
            detection_scores,
        )

        row_idx, col_idx = hungarian_assignment(cost)
        matched_pairs: list[tuple[int, int]] = []
        for row, col in zip(row_idx.tolist(), col_idx.tolist(), strict=True):
            if float(cost[row, col].item()) >= (self.invalid_cost * 0.5):
                continue
            matched_pairs.append((row, col))
        matched_pairs.sort(key=lambda pair: (pair[0], pair[1]))

        matched_track_idxs = [pair[0] for pair in matched_pairs]
        matched_det_idxs = [pair[1] for pair in matched_pairs]

        used_tracks = set(matched_track_idxs)
        used_dets = set(matched_det_idxs)
        unmatched_track_idxs = [
            idx for idx in range(prev_state.num_tracks) if idx not in used_tracks
        ]
        unmatched_det_idxs = [idx for idx in range(n_dets) if idx not in used_dets]

        previous_hits = prev_state.hit_counts
        previous_bank = prev_state.memory_bank
        previous_counts = prev_state.memory_counts
        if previous_hits is None or previous_bank is None or previous_counts is None:
            raise RuntimeError("track state memory fields are not initialized")

        records: list[tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        terminated_track_indices_list: list[int] = []
        terminated_track_ids_list: list[int] = []
        created_track_ids_list: list[int] = []

        for track_idx, det_idx in matched_pairs:
            track_id = int(prev_state.track_ids[track_idx].item())
            hits = int(previous_hits[track_idx].item())
            count = int(previous_counts[track_idx].item())
            updated_bank, updated_count = self._append_memory(
                previous_bank[track_idx],
                previous_count=count,
                previous_hits=hits,
                new_embedding=det_norm[det_idx],
            )
            records.append(
                (
                    track_id,
                    det_norm[det_idx : det_idx + 1],
                    detection_boxes_xyxy[det_idx : det_idx + 1],
                    detection_scores[det_idx : det_idx + 1],
                    torch.zeros((1,), dtype=torch.int64, device=device),
                    torch.tensor([hits + 1], dtype=torch.int64, device=device),
                    updated_bank.unsqueeze(0),
                    torch.tensor([updated_count], dtype=torch.int64, device=device),
                )
            )

        for track_idx in unmatched_track_idxs:
            age_next = int(prev_state.ages[track_idx].item()) + 1
            if age_next > self.max_age:
                terminated_track_indices_list.append(track_idx)
                terminated_track_ids_list.append(int(prev_state.track_ids[track_idx].item()))
                continue
            records.append(
                (
                    int(prev_state.track_ids[track_idx].item()),
                    prev_state.embeddings[track_idx : track_idx + 1],
                    prev_state.boxes_xyxy[track_idx : track_idx + 1],
                    prev_state.scores[track_idx : track_idx + 1],
                    torch.tensor([age_next], dtype=torch.int64, device=device),
                    previous_hits[track_idx : track_idx + 1],
                    previous_bank[track_idx : track_idx + 1],
                    previous_counts[track_idx : track_idx + 1],
                )
            )

        next_track_id = prev_state.next_track_id
        for det_idx in unmatched_det_idxs:
            created_track_ids_list.append(next_track_id)
            new_bank = torch.zeros(
                (1, self.memory_bank_size, emb_dim), dtype=torch.float32, device=device
            )
            new_bank[0, 0, :] = det_norm[det_idx]
            records.append(
                (
                    next_track_id,
                    det_norm[det_idx : det_idx + 1],
                    detection_boxes_xyxy[det_idx : det_idx + 1],
                    detection_scores[det_idx : det_idx + 1],
                    torch.zeros((1,), dtype=torch.int64, device=device),
                    torch.ones((1,), dtype=torch.int64, device=device),
                    new_bank,
                    torch.ones((1,), dtype=torch.int64, device=device),
                )
            )
            next_track_id += 1

        if records:
            records.sort(key=lambda entry: entry[0])
            next_ids = torch.tensor(
                [entry[0] for entry in records], dtype=torch.int64, device=device
            )
            next_embeddings = torch.cat([entry[1] for entry in records], dim=0)
            next_boxes = torch.cat([entry[2] for entry in records], dim=0)
            next_scores = torch.cat([entry[3] for entry in records], dim=0)
            next_ages = torch.cat([entry[4] for entry in records], dim=0)
            next_hits = torch.cat([entry[5] for entry in records], dim=0)
            next_memory_bank = torch.cat([entry[6] for entry in records], dim=0)
            next_memory_counts = torch.cat([entry[7] for entry in records], dim=0)
            next_state = TrackState(
                track_ids=next_ids,
                embeddings=next_embeddings,
                boxes_xyxy=next_boxes,
                scores=next_scores,
                ages=next_ages,
                frame_index=frame_index,
                hit_counts=next_hits,
                memory_bank=next_memory_bank,
                memory_counts=next_memory_counts,
            )
        else:
            next_state = TrackState.empty(
                embedding_dim=max(prev_state.embedding_dim, emb_dim, 1),
                frame_index=frame_index,
                device=device,
                bank_size=self.memory_bank_size,
            )

        return AssociationResult(
            matched_detection_indices=torch.tensor(
                matched_det_idxs, dtype=torch.int64, device=device
            ),
            matched_track_indices=torch.tensor(
                matched_track_idxs, dtype=torch.int64, device=device
            ),
            new_detection_indices=torch.tensor(
                unmatched_det_idxs, dtype=torch.int64, device=device
            ),
            next_state=next_state,
            terminated_track_indices=torch.tensor(
                terminated_track_indices_list,
                dtype=torch.int64,
                device=device,
            ),
            terminated_track_ids=torch.tensor(
                terminated_track_ids_list, dtype=torch.int64, device=device
            ),
            created_track_ids=torch.tensor(
                created_track_ids_list, dtype=torch.int64, device=device
            ),
        )


class GreedyCosineAssociator(HungarianAssociator):
    """Backward-compatible associator using cosine-only gating/cost."""

    def __init__(self, *, match_threshold: float = 0.3, max_age: int = 30) -> None:
        if not (-1.0 <= match_threshold <= 1.0):
            raise ValueError("match_threshold must be within [-1, 1]")
        distance_gate = min(2.0, max(0.0, 1.0 - match_threshold))
        super().__init__(
            iou_gate=-1.0,
            embedding_distance_gate=distance_gate,
            iou_weight=0.0,
            embedding_weight=1.0,
            score_weight=0.0,
            max_age=max_age,
            memory_bank_size=8,
        )
        self.match_threshold = float(match_threshold)


# Backward-compatible aliases
TrackAssociatorProtocol = AssociationProtocol
TrackAssociator = HungarianAssociator


__all__ = [
    "TrackState",
    "AssociationResult",
    "AssociationProtocol",
    "TrackAssociatorProtocol",
    "TrackAssociator",
    "GreedyCosineAssociator",
    "HungarianAssociator",
    "hungarian_assignment",
]
