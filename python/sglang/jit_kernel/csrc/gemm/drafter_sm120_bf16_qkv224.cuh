/* Copyright 2026 SGLang Team. All Rights Reserved. */
#pragma once

#include "drafter_sm120_bf16_qkv32.cuh"

// Retain the existing cp.async/MMA/epilogue/launch implementation. These two
// M224 schedules isolate larger-M reuse at the same four-warp, ~48-KiB budget.
inline void drafter_sm120_bf16_qkv224_topology(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<6, 32, 64, 64, 32, 32, 224>(output, activation, weight, workspace);
}

inline void drafter_sm120_bf16_qkv224_reuse_m128(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 128, 64, 64, 32, 224>(output, activation, weight, workspace);
}
