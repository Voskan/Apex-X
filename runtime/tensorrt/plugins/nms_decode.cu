#include "nms_decode.h"

#if APEXX_ENABLE_TENSORRT

#include <algorithm>
#include <cstdint>
#include <limits>

#include "cuda_fp16.h"

namespace apexx::trt::plugins {

namespace {

__device__ __forceinline__ float clamp_f32(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0F / (1.0F + expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x) {
  const float a = fabsf(x);
  return log1pf(expf(-a)) + fmaxf(x, 0.0F);
}

__device__ __forceinline__ float iou_xyxy(const float* box_a, const float* box_b) {
  const float inter_x1 = fmaxf(box_a[0], box_b[0]);
  const float inter_y1 = fmaxf(box_a[1], box_b[1]);
  const float inter_x2 = fminf(box_a[2], box_b[2]);
  const float inter_y2 = fminf(box_a[3], box_b[3]);
  const float inter_w = fmaxf(inter_x2 - inter_x1, 0.0F);
  const float inter_h = fmaxf(inter_y2 - inter_y1, 0.0F);
  const float inter = inter_w * inter_h;
  const float area_a = fmaxf(box_a[2] - box_a[0], 0.0F) * fmaxf(box_a[3] - box_a[1], 0.0F);
  const float area_b = fmaxf(box_b[2] - box_b[0], 0.0F) * fmaxf(box_b[3] - box_b[1], 0.0F);
  const float denom = fmaxf(area_a + area_b - inter, 1.0e-9F);
  return inter / denom;
}

__device__ __forceinline__ bool better_candidate(
    float score_a,
    int32_t pair_id_a,
    float score_b,
    int32_t pair_id_b) {
  if (score_a > score_b) {
    return true;
  }
  if (score_a < score_b) {
    return false;
  }
  return pair_id_a < pair_id_b;
}

__device__ __forceinline__ void swap_candidates(
    float* scores,
    float* boxes,
    int32_t* class_ids,
    int32_t* pair_ids,
    int32_t i,
    int32_t j) {
  const float s = scores[i];
  scores[i] = scores[j];
  scores[j] = s;
  const int32_t cid = class_ids[i];
  class_ids[i] = class_ids[j];
  class_ids[j] = cid;
  const int32_t pid = pair_ids[i];
  pair_ids[i] = pair_ids[j];
  pair_ids[j] = pid;
  for (int d = 0; d < 4; ++d) {
    const float v = boxes[i * 4 + d];
    boxes[i * 4 + d] = boxes[j * 4 + d];
    boxes[j * 4 + d] = v;
  }
}

__device__ __forceinline__ void insert_candidate_sorted(
    float score,
    const float* box4,
    int32_t class_id,
    int32_t pair_id,
    float* scores,
    float* boxes,
    int32_t* class_ids,
    int32_t* pair_ids,
    int32_t& count,
    int32_t cap) {
  if (cap <= 0) {
    return;
  }
  if (count < cap) {
    int32_t pos = count;
    scores[pos] = score;
    class_ids[pos] = class_id;
    pair_ids[pos] = pair_id;
    for (int d = 0; d < 4; ++d) {
      boxes[pos * 4 + d] = box4[d];
    }
    ++count;
    while (pos > 0 &&
           better_candidate(scores[pos], pair_ids[pos], scores[pos - 1], pair_ids[pos - 1])) {
      swap_candidates(scores, boxes, class_ids, pair_ids, pos, pos - 1);
      --pos;
    }
    return;
  }

  if (!better_candidate(score, pair_id, scores[count - 1], pair_ids[count - 1])) {
    return;
  }
  int32_t pos = count - 1;
  scores[pos] = score;
  class_ids[pos] = class_id;
  pair_ids[pos] = pair_id;
  for (int d = 0; d < 4; ++d) {
    boxes[pos * 4 + d] = box4[d];
  }
  while (pos > 0 &&
         better_candidate(scores[pos], pair_ids[pos], scores[pos - 1], pair_ids[pos - 1])) {
    swap_candidates(scores, boxes, class_ids, pair_ids, pos, pos - 1);
    --pos;
  }
}

__device__ __forceinline__ size_t align_up(size_t value, size_t align) {
  const size_t rem = value % align;
  return rem == 0 ? value : value + (align - rem);
}

__device__ __forceinline__ size_t per_batch_workspace_bytes(int32_t cap) {
  size_t offset = 0;
  offset = align_up(offset, alignof(float));
  offset += static_cast<size_t>(cap) * sizeof(float);      // scores
  offset = align_up(offset, alignof(float));
  offset += static_cast<size_t>(cap) * 4 * sizeof(float);  // boxes
  offset = align_up(offset, alignof(int32_t));
  offset += static_cast<size_t>(cap) * sizeof(int32_t);  // class_ids
  offset += static_cast<size_t>(cap) * sizeof(int32_t);  // pair_ids
  offset += static_cast<size_t>(cap) * sizeof(int32_t);  // suppressed
  offset += static_cast<size_t>(cap) * sizeof(int32_t);  // keep_ids
  return offset;
}

__global__ void nms_decode_kernel_fp16(
    const __half* cls_logits,
    const __half* box_reg,
    const __half* quality,
    const __half* centers,
    const __half* strides,
    __half* boxes_out,
    __half* scores_out,
    int32_t* class_ids_out,
    int32_t* valid_counts_out,
    uint8_t* workspace,
    int32_t anchors,
    int32_t classes,
    int32_t max_detections,
    int32_t candidate_cap,
    float score_threshold,
    float iou_threshold) {
  const int32_t b = static_cast<int32_t>(blockIdx.x);
  if (threadIdx.x != 0) {
    return;
  }

  const size_t per_batch = per_batch_workspace_bytes(candidate_cap);
  uint8_t* batch_ws = workspace + static_cast<size_t>(b) * per_batch;
  size_t offset = 0;
  offset = align_up(offset, alignof(float));
  float* cand_scores = reinterpret_cast<float*>(batch_ws + offset);
  offset += static_cast<size_t>(candidate_cap) * sizeof(float);
  offset = align_up(offset, alignof(float));
  float* cand_boxes = reinterpret_cast<float*>(batch_ws + offset);
  offset += static_cast<size_t>(candidate_cap) * 4 * sizeof(float);
  offset = align_up(offset, alignof(int32_t));
  int32_t* cand_class_ids = reinterpret_cast<int32_t*>(batch_ws + offset);
  offset += static_cast<size_t>(candidate_cap) * sizeof(int32_t);
  int32_t* cand_pair_ids = reinterpret_cast<int32_t*>(batch_ws + offset);
  offset += static_cast<size_t>(candidate_cap) * sizeof(int32_t);
  int32_t* suppressed = reinterpret_cast<int32_t*>(batch_ws + offset);
  offset += static_cast<size_t>(candidate_cap) * sizeof(int32_t);
  int32_t* keep_ids = reinterpret_cast<int32_t*>(batch_ws + offset);

  for (int32_t i = 0; i < max_detections; ++i) {
    const int64_t out_box_off =
        (static_cast<int64_t>(b) * max_detections + i) * 4;
    boxes_out[out_box_off + 0] = __float2half(0.0F);
    boxes_out[out_box_off + 1] = __float2half(0.0F);
    boxes_out[out_box_off + 2] = __float2half(0.0F);
    boxes_out[out_box_off + 3] = __float2half(0.0F);
    scores_out[static_cast<int64_t>(b) * max_detections + i] = __float2half(0.0F);
    class_ids_out[static_cast<int64_t>(b) * max_detections + i] = -1;
  }
  valid_counts_out[b] = 0;

  int32_t cand_count = 0;
  for (int32_t a = 0; a < anchors; ++a) {
    const float q = sigmoid_f32(clamp_f32(
        __half2float(quality[static_cast<int64_t>(b) * anchors + a]),
        -60.0F,
        60.0F));

    const float stride = __half2float(strides[a]);
    const float cx = __half2float(centers[static_cast<int64_t>(a) * 2 + 0]);
    const float cy = __half2float(centers[static_cast<int64_t>(a) * 2 + 1]);

    const int64_t box_off = (static_cast<int64_t>(b) * anchors + a) * 4;
    const float l = softplus_f32(clamp_f32(__half2float(box_reg[box_off + 0]), -20.0F, 20.0F)) * stride;
    const float t = softplus_f32(clamp_f32(__half2float(box_reg[box_off + 1]), -20.0F, 20.0F)) * stride;
    const float r = softplus_f32(clamp_f32(__half2float(box_reg[box_off + 2]), -20.0F, 20.0F)) * stride;
    const float btm = softplus_f32(clamp_f32(__half2float(box_reg[box_off + 3]), -20.0F, 20.0F)) * stride;
    const float decoded[4] = {cx - l, cy - t, cx + r, cy + btm};

    for (int32_t c = 0; c < classes; ++c) {
      const int64_t cls_off = (static_cast<int64_t>(b) * anchors + a) * classes + c;
      const float cls = sigmoid_f32(clamp_f32(__half2float(cls_logits[cls_off]), -60.0F, 60.0F));
      const float score = cls * q;
      if (score < score_threshold) {
        continue;
      }
      const int32_t pair_id = a * classes + c;
      insert_candidate_sorted(
          score,
          decoded,
          c,
          pair_id,
          cand_scores,
          cand_boxes,
          cand_class_ids,
          cand_pair_ids,
          cand_count,
          candidate_cap);
    }
  }

  for (int32_t i = 0; i < cand_count; ++i) {
    suppressed[i] = 0;
  }
  int32_t keep_count = 0;
  for (int32_t cls = 0; cls < classes; ++cls) {
    for (int32_t i = 0; i < cand_count; ++i) {
      if (suppressed[i] != 0 || cand_class_ids[i] != cls) {
        continue;
      }
      keep_ids[keep_count++] = i;
      for (int32_t j = i + 1; j < cand_count; ++j) {
        if (suppressed[j] != 0 || cand_class_ids[j] != cls) {
          continue;
        }
        const float iou = iou_xyxy(&cand_boxes[i * 4], &cand_boxes[j * 4]);
        if (iou > iou_threshold) {
          suppressed[j] = 1;
        }
      }
    }
  }

  for (int32_t i = 1; i < keep_count; ++i) {
    int32_t cur = keep_ids[i];
    int32_t j = i - 1;
    while (j >= 0) {
      const int32_t prev = keep_ids[j];
      if (better_candidate(
              cand_scores[cur],
              cand_pair_ids[cur],
              cand_scores[prev],
              cand_pair_ids[prev])) {
        keep_ids[j + 1] = keep_ids[j];
        --j;
      } else {
        break;
      }
    }
    keep_ids[j + 1] = cur;
  }

  const int32_t valid = keep_count < max_detections ? keep_count : max_detections;
  valid_counts_out[b] = valid;
  for (int32_t out_i = 0; out_i < valid; ++out_i) {
    const int32_t k = keep_ids[out_i];
    const int64_t out_box_off = (static_cast<int64_t>(b) * max_detections + out_i) * 4;
    boxes_out[out_box_off + 0] = __float2half(cand_boxes[k * 4 + 0]);
    boxes_out[out_box_off + 1] = __float2half(cand_boxes[k * 4 + 1]);
    boxes_out[out_box_off + 2] = __float2half(cand_boxes[k * 4 + 2]);
    boxes_out[out_box_off + 3] = __float2half(cand_boxes[k * 4 + 3]);
    scores_out[static_cast<int64_t>(b) * max_detections + out_i] = __float2half(cand_scores[k]);
    class_ids_out[static_cast<int64_t>(b) * max_detections + out_i] = cand_class_ids[k];
  }
}

}  // namespace

cudaError_t launch_nms_decode_fp16(
    const void* cls_logits,
    const void* box_reg,
    const void* quality,
    const void* centers,
    const void* strides,
    void* boxes_out,
    void* scores_out,
    int32_t* class_ids_out,
    int32_t* valid_counts_out,
    void* workspace,
    int32_t batch,
    int32_t anchors,
    int32_t classes,
    int32_t max_detections,
    int32_t pre_nms_topk,
    float score_threshold,
    float iou_threshold,
    cudaStream_t stream) noexcept {
  if (cls_logits == nullptr || box_reg == nullptr || quality == nullptr || centers == nullptr ||
      strides == nullptr || boxes_out == nullptr || scores_out == nullptr ||
      class_ids_out == nullptr || valid_counts_out == nullptr || batch <= 0 || anchors <= 0 ||
      classes <= 0 || max_detections <= 0 || pre_nms_topk <= 0) {
    return cudaErrorInvalidValue;
  }

  const int64_t candidate_cap_64 =
      std::min(static_cast<int64_t>(pre_nms_topk), static_cast<int64_t>(anchors) * classes);
  if (candidate_cap_64 <= 0 || candidate_cap_64 > std::numeric_limits<int32_t>::max()) {
    return cudaErrorInvalidValue;
  }
  const int32_t candidate_cap = static_cast<int32_t>(candidate_cap_64);
  if (workspace == nullptr) {
    return cudaErrorInvalidValue;
  }

  nms_decode_kernel_fp16<<<batch, 1, 0, stream>>>(
      static_cast<const __half*>(cls_logits),
      static_cast<const __half*>(box_reg),
      static_cast<const __half*>(quality),
      static_cast<const __half*>(centers),
      static_cast<const __half*>(strides),
      static_cast<__half*>(boxes_out),
      static_cast<__half*>(scores_out),
      class_ids_out,
      valid_counts_out,
      static_cast<uint8_t*>(workspace),
      anchors,
      classes,
      max_detections,
      candidate_cap,
      score_threshold,
      iou_threshold);
  return cudaGetLastError();
}

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
