#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace qkern {

// -----------------------------------------------------------------------------
// Compile-time INT4 GEMV kernel configuration
// -----------------------------------------------------------------------------
//
// Mapping (decode GEMV): W[N,K] @ x[K] -> y[N], batch M = 1.
// One thread owns one output row n. A block owns BLOCK_N consecutive rows
// (threads = BLOCK_N).
//
// BLOCK_M is fixed to 1: there is no batch/tile-M dimension in this GEMV.
// It is documented but not a free template parameter (forcing it would be
// meaningless for the current mapping).
//
// Parameters that do affect this kernel:
//   BLOCK_N   — threads per block / output rows per block
//   BLOCK_K   — activation tile length in shared memory; 0 disables smem tiling
//   VEC_SIZE  — packed INT4 bytes consumed per inner-step vectorized W load
//               (1, 2, or 4); tails always fall back to scalar bytes
//   NUM_STAGES— number of smem buffers for x tiles (only when BLOCK_K > 0);
//               raises smem footprint / can lower occupancy; enables later
//               software pipelining without changing the API

template <int BlockN_, int BlockK_, int VecSize_, int NumStages_>
struct Int4GemvConfig {
  static constexpr int BLOCK_M = 1;
  static constexpr int BLOCK_N = BlockN_;
  static constexpr int BLOCK_K = BlockK_;
  static constexpr int VEC_SIZE = VecSize_;
  static constexpr int NUM_STAGES = NumStages_;

  static_assert(BLOCK_N > 0 && BLOCK_N <= 1024, "BLOCK_N out of range");
  static_assert(VEC_SIZE == 1 || VEC_SIZE == 2 || VEC_SIZE == 4, "VEC_SIZE must be 1|2|4");
  static_assert(NUM_STAGES >= 1 && NUM_STAGES <= 4, "NUM_STAGES out of range");
  static_assert(BLOCK_K >= 0, "BLOCK_K must be >= 0");
  static_assert(BLOCK_K == 0 || (BLOCK_K % 2 == 0), "BLOCK_K must be even when > 0 (INT4 pairs)");
};

// Default = existing correctness-first path (no smem tile, scalar packed loads).
using Int4GemvConfigDefault = Int4GemvConfig<256, 0, 1, 1>;
using Int4GemvConfigBn128Bk128V1S1 = Int4GemvConfig<128, 128, 1, 1>;
using Int4GemvConfigBn256Bk256V4S2 = Int4GemvConfig<256, 256, 4, 2>;

enum class Int4GemvConfigId : int {
  Default = 0,
  Bn128Bk128V1S1 = 1,
  Bn256Bk256V4S2 = 2,
};

inline const char* int4_gemv_config_name(Int4GemvConfigId id) {
  switch (id) {
    case Int4GemvConfigId::Default:
      return "default";
    case Int4GemvConfigId::Bn128Bk128V1S1:
      return "bn128_bk128_v1_s1";
    case Int4GemvConfigId::Bn256Bk256V4S2:
      return "bn256_bk256_v4_s2";
    default:
      return "unknown";
  }
}

inline bool parse_int4_gemv_config_id(const std::string& name, Int4GemvConfigId* out) {
  if (name == "default" || name.empty()) {
    *out = Int4GemvConfigId::Default;
    return true;
  }
  if (name == "bn128_bk128_v1_s1") {
    *out = Int4GemvConfigId::Bn128Bk128V1S1;
    return true;
  }
  if (name == "bn256_bk256_v4_s2") {
    *out = Int4GemvConfigId::Bn256Bk256V4S2;
    return true;
  }
  return false;
}

struct Int4GemvConfigInfo {
  const char* name;
  int block_m;
  int block_n;
  int block_k;
  int vec_size;
  int num_stages;
  bool is_default;
};

inline Int4GemvConfigInfo int4_gemv_config_info(Int4GemvConfigId id) {
  switch (id) {
    case Int4GemvConfigId::Bn128Bk128V1S1:
      return {"bn128_bk128_v1_s1", 1, 128, 128, 1, 1, false};
    case Int4GemvConfigId::Bn256Bk256V4S2:
      return {"bn256_bk256_v4_s2", 1, 256, 256, 4, 2, false};
    case Int4GemvConfigId::Default:
    default:
      return {"default", 1, 256, 0, 1, 1, true};
  }
}

}  // namespace qkern
