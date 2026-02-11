#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if APEXX_ENABLE_TENSORRT && APEXX_ENABLE_CUDA && APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN

#include "NvInfer.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

#include "nms_decode.h"

namespace {

using apexx::trt::plugins::NMSDecodePluginCreator;

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
  return false;
}

float sigmoid_f32(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

float softplus_f32(float x) {
  const float a = std::fabs(x);
  return std::log1p(std::exp(-a)) + std::max(x, 0.0F);
}

float iou_xyxy(const float* a, const float* b) {
  const float ix1 = std::max(a[0], b[0]);
  const float iy1 = std::max(a[1], b[1]);
  const float ix2 = std::min(a[2], b[2]);
  const float iy2 = std::min(a[3], b[3]);
  const float iw = std::max(ix2 - ix1, 0.0F);
  const float ih = std::max(iy2 - iy1, 0.0F);
  const float inter = iw * ih;
  const float aa = std::max(a[2] - a[0], 0.0F) * std::max(a[3] - a[1], 0.0F);
  const float bb = std::max(b[2] - b[0], 0.0F) * std::max(b[3] - b[1], 0.0F);
  const float u = std::max(aa + bb - inter, 1.0e-9F);
  return inter / u;
}

struct RefOutputs {
  std::vector<__half> boxes;
  std::vector<__half> scores;
  std::vector<int32_t> class_ids;
  std::vector<int32_t> valid;
};

RefOutputs reference_decode_nms(
    const std::vector<__half>& cls_logits,
    const std::vector<__half>& box_reg,
    const std::vector<__half>& quality,
    const std::vector<__half>& centers,
    const std::vector<__half>& strides,
    int32_t batch,
    int32_t anchors,
    int32_t classes,
    int32_t max_det,
    int32_t pre_topk,
    float score_threshold,
    float iou_threshold) {
  RefOutputs out;
  out.boxes.assign(static_cast<std::size_t>(batch * max_det * 4), __float2half(0.0F));
  out.scores.assign(static_cast<std::size_t>(batch * max_det), __float2half(0.0F));
  out.class_ids.assign(static_cast<std::size_t>(batch * max_det), -1);
  out.valid.assign(static_cast<std::size_t>(batch), 0);

  const int32_t cap = std::min(pre_topk, anchors * classes);
  for (int32_t b = 0; b < batch; ++b) {
    std::vector<float> cand_scores;
    std::vector<float> cand_boxes;
    std::vector<int32_t> cand_cls;
    std::vector<int32_t> cand_pair;
    cand_scores.reserve(static_cast<std::size_t>(cap));
    cand_boxes.reserve(static_cast<std::size_t>(cap) * 4U);
    cand_cls.reserve(static_cast<std::size_t>(cap));
    cand_pair.reserve(static_cast<std::size_t>(cap));

    for (int32_t a = 0; a < anchors; ++a) {
      const float q = sigmoid_f32(std::clamp(
          __half2float(quality[static_cast<std::size_t>(b * anchors + a)]),
          -60.0F,
          60.0F));
      const float stride = __half2float(strides[static_cast<std::size_t>(a)]);
      const float cx = __half2float(centers[static_cast<std::size_t>(a * 2 + 0)]);
      const float cy = __half2float(centers[static_cast<std::size_t>(a * 2 + 1)]);
      const std::size_t boff = static_cast<std::size_t>((b * anchors + a) * 4);
      const float l =
          softplus_f32(std::clamp(__half2float(box_reg[boff + 0]), -20.0F, 20.0F)) * stride;
      const float t =
          softplus_f32(std::clamp(__half2float(box_reg[boff + 1]), -20.0F, 20.0F)) * stride;
      const float r =
          softplus_f32(std::clamp(__half2float(box_reg[boff + 2]), -20.0F, 20.0F)) * stride;
      const float bb =
          softplus_f32(std::clamp(__half2float(box_reg[boff + 3]), -20.0F, 20.0F)) * stride;
      const float box4[4] = {cx - l, cy - t, cx + r, cy + bb};

      for (int32_t c = 0; c < classes; ++c) {
        const std::size_t cls_off = static_cast<std::size_t>(((b * anchors + a) * classes) + c);
        const float cls = sigmoid_f32(
            std::clamp(__half2float(cls_logits[cls_off]), -60.0F, 60.0F));
        const float score = cls * q;
        if (score < score_threshold) {
          continue;
        }
        cand_scores.push_back(score);
        cand_cls.push_back(c);
        cand_pair.push_back(a * classes + c);
        cand_boxes.insert(cand_boxes.end(), box4, box4 + 4);
      }
    }

    std::vector<int32_t> order(static_cast<int32_t>(cand_scores.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(order.size()); ++i) {
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
      const float sl = cand_scores[static_cast<std::size_t>(lhs)];
      const float sr = cand_scores[static_cast<std::size_t>(rhs)];
      if (sl > sr) {
        return true;
      }
      if (sl < sr) {
        return false;
      }
      return cand_pair[static_cast<std::size_t>(lhs)] < cand_pair[static_cast<std::size_t>(rhs)];
    });
    if (static_cast<int32_t>(order.size()) > cap) {
      order.resize(static_cast<std::size_t>(cap));
    }

    std::vector<int32_t> suppressed(order.size(), 0);
    std::vector<int32_t> kept;
    kept.reserve(order.size());
    for (int32_t cls = 0; cls < classes; ++cls) {
      for (int32_t oi = 0; oi < static_cast<int32_t>(order.size()); ++oi) {
        if (suppressed[static_cast<std::size_t>(oi)] != 0) {
          continue;
        }
        const int32_t cand_i = order[static_cast<std::size_t>(oi)];
        if (cand_cls[static_cast<std::size_t>(cand_i)] != cls) {
          continue;
        }
        kept.push_back(cand_i);
        const float* box_i = &cand_boxes[static_cast<std::size_t>(cand_i) * 4];
        for (int32_t oj = oi + 1; oj < static_cast<int32_t>(order.size()); ++oj) {
          if (suppressed[static_cast<std::size_t>(oj)] != 0) {
            continue;
          }
          const int32_t cand_j = order[static_cast<std::size_t>(oj)];
          if (cand_cls[static_cast<std::size_t>(cand_j)] != cls) {
            continue;
          }
          const float* box_j = &cand_boxes[static_cast<std::size_t>(cand_j) * 4];
          if (iou_xyxy(box_i, box_j) > iou_threshold) {
            suppressed[static_cast<std::size_t>(oj)] = 1;
          }
        }
      }
    }

    std::stable_sort(kept.begin(), kept.end(), [&](int32_t lhs, int32_t rhs) {
      const float sl = cand_scores[static_cast<std::size_t>(lhs)];
      const float sr = cand_scores[static_cast<std::size_t>(rhs)];
      if (sl > sr) {
        return true;
      }
      if (sl < sr) {
        return false;
      }
      return cand_pair[static_cast<std::size_t>(lhs)] < cand_pair[static_cast<std::size_t>(rhs)];
    });

    const int32_t valid = std::min<int32_t>(static_cast<int32_t>(kept.size()), max_det);
    out.valid[static_cast<std::size_t>(b)] = valid;
    for (int32_t i = 0; i < valid; ++i) {
      const int32_t k = kept[static_cast<std::size_t>(i)];
      out.scores[static_cast<std::size_t>(b * max_det + i)] =
          __float2half(cand_scores[static_cast<std::size_t>(k)]);
      out.class_ids[static_cast<std::size_t>(b * max_det + i)] =
          cand_cls[static_cast<std::size_t>(k)];
      const std::size_t out_off = static_cast<std::size_t>((b * max_det + i) * 4);
      const std::size_t box_off = static_cast<std::size_t>(k) * 4;
      out.boxes[out_off + 0] = __float2half(cand_boxes[box_off + 0]);
      out.boxes[out_off + 1] = __float2half(cand_boxes[box_off + 1]);
      out.boxes[out_off + 2] = __float2half(cand_boxes[box_off + 2]);
      out.boxes[out_off + 3] = __float2half(cand_boxes[box_off + 3]);
    }
  }
  return out;
}

bool run_case(float score_threshold, float iou_threshold) {
  constexpr int32_t batch = 1;
  constexpr int32_t anchors = 24;
  constexpr int32_t classes = 3;
  constexpr int32_t max_det = 10;
  constexpr int32_t pre_topk = 50;
  constexpr float tol = 6.0e-2F;

  std::vector<__half> h_cls(static_cast<std::size_t>(batch * anchors * classes));
  std::vector<__half> h_box(static_cast<std::size_t>(batch * anchors * 4));
  std::vector<__half> h_quality(static_cast<std::size_t>(batch * anchors));
  std::vector<__half> h_centers(static_cast<std::size_t>(anchors * 2));
  std::vector<__half> h_strides(static_cast<std::size_t>(anchors));

  for (int32_t a = 0; a < anchors; ++a) {
    h_centers[static_cast<std::size_t>(a * 2 + 0)] = __float2half(4.0F + 2.0F * static_cast<float>(a));
    h_centers[static_cast<std::size_t>(a * 2 + 1)] = __float2half(5.0F + 1.5F * static_cast<float>(a % 7));
    h_strides[static_cast<std::size_t>(a)] = __float2half(8.0F);
  }
  for (std::size_t i = 0; i < h_box.size(); ++i) {
    h_box[i] = __float2half(1.5F + 0.01F * static_cast<float>(i % 13));
  }
  for (std::size_t i = 0; i < h_quality.size(); ++i) {
    h_quality[i] = __float2half(0.6F + 0.1F * std::sin(static_cast<float>(i)));
  }
  for (std::size_t i = 0; i < h_cls.size(); ++i) {
    h_cls[i] = __float2half(2.0F * std::sin(static_cast<float>(i) * 0.1F));
  }

  const auto ref = reference_decode_nms(
      h_cls,
      h_box,
      h_quality,
      h_centers,
      h_strides,
      batch,
      anchors,
      classes,
      max_det,
      pre_topk,
      score_threshold,
      iou_threshold);

  __half* d_cls = nullptr;
  __half* d_box = nullptr;
  __half* d_quality = nullptr;
  __half* d_centers = nullptr;
  __half* d_strides = nullptr;
  __half* d_boxes = nullptr;
  __half* d_scores = nullptr;
  int32_t* d_class_ids = nullptr;
  int32_t* d_valid = nullptr;
  void* d_workspace = nullptr;

  const std::size_t cls_bytes = h_cls.size() * sizeof(__half);
  const std::size_t box_bytes = h_box.size() * sizeof(__half);
  const std::size_t quality_bytes = h_quality.size() * sizeof(__half);
  const std::size_t centers_bytes = h_centers.size() * sizeof(__half);
  const std::size_t strides_bytes = h_strides.size() * sizeof(__half);
  const std::size_t out_boxes_bytes = static_cast<std::size_t>(batch * max_det * 4) * sizeof(__half);
  const std::size_t out_scores_bytes = static_cast<std::size_t>(batch * max_det) * sizeof(__half);
  const std::size_t out_class_bytes = static_cast<std::size_t>(batch * max_det) * sizeof(int32_t);
  const std::size_t out_valid_bytes = static_cast<std::size_t>(batch) * sizeof(int32_t);

  if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cls), cls_bytes), "cudaMalloc(cls)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_box), box_bytes), "cudaMalloc(box)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_quality), quality_bytes), "cudaMalloc(quality)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_centers), centers_bytes), "cudaMalloc(centers)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_strides), strides_bytes), "cudaMalloc(strides)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_boxes), out_boxes_bytes), "cudaMalloc(out boxes)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scores), out_scores_bytes), "cudaMalloc(out scores)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_class_ids), out_class_bytes), "cudaMalloc(out class)") ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_valid), out_valid_bytes), "cudaMalloc(out valid)")) {
    return false;
  }

  if (!check_cuda(cudaMemcpy(d_cls, h_cls.data(), cls_bytes, cudaMemcpyHostToDevice), "copy cls") ||
      !check_cuda(cudaMemcpy(d_box, h_box.data(), box_bytes, cudaMemcpyHostToDevice), "copy box") ||
      !check_cuda(cudaMemcpy(d_quality, h_quality.data(), quality_bytes, cudaMemcpyHostToDevice), "copy quality") ||
      !check_cuda(cudaMemcpy(d_centers, h_centers.data(), centers_bytes, cudaMemcpyHostToDevice), "copy centers") ||
      !check_cuda(cudaMemcpy(d_strides, h_strides.data(), strides_bytes, cudaMemcpyHostToDevice), "copy strides")) {
    return false;
  }

  NMSDecodePluginCreator creator;
  nvinfer1::PluginField fields[4] = {
      nvinfer1::PluginField{"max_detections", &max_det, nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"pre_nms_topk", &pre_topk, nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"score_threshold", &score_threshold, nvinfer1::PluginFieldType::kFLOAT32, 1},
      nvinfer1::PluginField{"iou_threshold", &iou_threshold, nvinfer1::PluginFieldType::kFLOAT32, 1},
  };
  nvinfer1::PluginFieldCollection fc{};
  fc.nbFields = 4;
  fc.fields = fields;
  nvinfer1::IPluginV2* plugin_raw = creator.createPlugin("nms_decode_test", &fc);
  if (plugin_raw == nullptr) {
    std::cerr << "createPlugin failed\n";
    return false;
  }
  auto* plugin = static_cast<nvinfer1::IPluginV2DynamicExt*>(plugin_raw);
  if (plugin->initialize() != 0) {
    std::cerr << "plugin initialize failed\n";
    plugin->destroy();
    return false;
  }

  nvinfer1::PluginTensorDesc in_desc[5]{};
  in_desc[0].type = nvinfer1::DataType::kHALF;
  in_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[0].dims.nbDims = 3;
  in_desc[0].dims.d[0] = batch;
  in_desc[0].dims.d[1] = anchors;
  in_desc[0].dims.d[2] = classes;
  in_desc[1].type = nvinfer1::DataType::kHALF;
  in_desc[1].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[1].dims.nbDims = 3;
  in_desc[1].dims.d[0] = batch;
  in_desc[1].dims.d[1] = anchors;
  in_desc[1].dims.d[2] = 4;
  in_desc[2].type = nvinfer1::DataType::kHALF;
  in_desc[2].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[2].dims.nbDims = 2;
  in_desc[2].dims.d[0] = batch;
  in_desc[2].dims.d[1] = anchors;
  in_desc[3].type = nvinfer1::DataType::kHALF;
  in_desc[3].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[3].dims.nbDims = 2;
  in_desc[3].dims.d[0] = anchors;
  in_desc[3].dims.d[1] = 2;
  in_desc[4].type = nvinfer1::DataType::kHALF;
  in_desc[4].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[4].dims.nbDims = 1;
  in_desc[4].dims.d[0] = anchors;

  nvinfer1::PluginTensorDesc out_desc[4]{};
  out_desc[0].type = nvinfer1::DataType::kHALF;
  out_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[0].dims.nbDims = 3;
  out_desc[0].dims.d[0] = batch;
  out_desc[0].dims.d[1] = max_det;
  out_desc[0].dims.d[2] = 4;
  out_desc[1].type = nvinfer1::DataType::kHALF;
  out_desc[1].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[1].dims.nbDims = 2;
  out_desc[1].dims.d[0] = batch;
  out_desc[1].dims.d[1] = max_det;
  out_desc[2].type = nvinfer1::DataType::kINT32;
  out_desc[2].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[2].dims.nbDims = 2;
  out_desc[2].dims.d[0] = batch;
  out_desc[2].dims.d[1] = max_det;
  out_desc[3].type = nvinfer1::DataType::kINT32;
  out_desc[3].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[3].dims.nbDims = 1;
  out_desc[3].dims.d[0] = batch;

  const std::size_t ws = plugin->getWorkspaceSize(in_desc, 5, out_desc, 4);
  if (ws > 0 && !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_workspace), ws), "cudaMalloc(workspace)")) {
    plugin->terminate();
    plugin->destroy();
    return false;
  }

  const void* inputs[5] = {d_cls, d_box, d_quality, d_centers, d_strides};
  void* outputs[4] = {d_boxes, d_scores, d_class_ids, d_valid};
  const int status = plugin->enqueue(in_desc, out_desc, inputs, outputs, d_workspace, nullptr);
  if (status != 0 || !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    std::cerr << "plugin enqueue failed with status " << status << "\n";
    plugin->terminate();
    plugin->destroy();
    return false;
  }

  std::vector<__half> got_boxes(ref.boxes.size());
  std::vector<__half> got_scores(ref.scores.size());
  std::vector<int32_t> got_class_ids(ref.class_ids.size(), -1);
  std::vector<int32_t> got_valid(ref.valid.size(), 0);
  if (!check_cuda(cudaMemcpy(got_boxes.data(), d_boxes, out_boxes_bytes, cudaMemcpyDeviceToHost), "copy out boxes") ||
      !check_cuda(cudaMemcpy(got_scores.data(), d_scores, out_scores_bytes, cudaMemcpyDeviceToHost), "copy out scores") ||
      !check_cuda(cudaMemcpy(got_class_ids.data(), d_class_ids, out_class_bytes, cudaMemcpyDeviceToHost), "copy out class") ||
      !check_cuda(cudaMemcpy(got_valid.data(), d_valid, out_valid_bytes, cudaMemcpyDeviceToHost), "copy out valid")) {
    plugin->terminate();
    plugin->destroy();
    return false;
  }

  for (std::size_t i = 0; i < got_valid.size(); ++i) {
    if (got_valid[i] != ref.valid[i]) {
      std::cerr << "valid mismatch at batch " << i << ": got=" << got_valid[i]
                << " expected=" << ref.valid[i] << "\n";
      plugin->terminate();
      plugin->destroy();
      return false;
    }
  }
  for (std::size_t i = 0; i < got_scores.size(); ++i) {
    const float g = __half2float(got_scores[i]);
    const float e = __half2float(ref.scores[i]);
    if (std::fabs(g - e) > tol) {
      std::cerr << "score mismatch at " << i << ": got=" << g << " expected=" << e << "\n";
      plugin->terminate();
      plugin->destroy();
      return false;
    }
    if (got_class_ids[i] != ref.class_ids[i]) {
      std::cerr << "class mismatch at " << i << ": got=" << got_class_ids[i]
                << " expected=" << ref.class_ids[i] << "\n";
      plugin->terminate();
      plugin->destroy();
      return false;
    }
  }
  for (std::size_t i = 0; i < got_boxes.size(); ++i) {
    const float g = __half2float(got_boxes[i]);
    const float e = __half2float(ref.boxes[i]);
    if (std::fabs(g - e) > tol) {
      std::cerr << "box mismatch at " << i << ": got=" << g << " expected=" << e << "\n";
      plugin->terminate();
      plugin->destroy();
      return false;
    }
  }

  cudaEvent_t ev_start{};
  cudaEvent_t ev_end{};
  check_cuda(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&ev_end), "cudaEventCreate(end)");
  constexpr int warmup = 20;
  constexpr int iters = 200;
  for (int i = 0; i < warmup; ++i) {
    plugin->enqueue(in_desc, out_desc, inputs, outputs, d_workspace, nullptr);
  }
  check_cuda(cudaDeviceSynchronize(), "sync warmup");
  check_cuda(cudaEventRecord(ev_start, nullptr), "record start");
  for (int i = 0; i < iters; ++i) {
    plugin->enqueue(in_desc, out_desc, inputs, outputs, d_workspace, nullptr);
  }
  check_cuda(cudaEventRecord(ev_end, nullptr), "record end");
  check_cuda(cudaEventSynchronize(ev_end), "sync end");
  float elapsed_ms = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end), "elapsed");
  std::cout << "DecodeNMS avg_ms=" << (elapsed_ms / static_cast<float>(iters)) << "\n";
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  plugin->terminate();
  plugin->destroy();
  cudaFree(d_workspace);
  cudaFree(d_cls);
  cudaFree(d_box);
  cudaFree(d_quality);
  cudaFree(d_centers);
  cudaFree(d_strides);
  cudaFree(d_boxes);
  cudaFree(d_scores);
  cudaFree(d_class_ids);
  cudaFree(d_valid);
  return true;
}

}  // namespace

int main() {
  if (!run_case(0.2F, 0.5F)) {
    return 2;
  }
  // High threshold corner case: no output boxes expected.
  if (!run_case(0.9999F, 0.5F)) {
    return 3;
  }
  std::cout << "DecodeNMS plugin test passed.\n";
  return 0;
}

#else

int main() {
  std::cout << "DecodeNMS plugin test skipped (TensorRT/CUDA/real plugin unavailable).\n";
  return 0;
}

#endif
