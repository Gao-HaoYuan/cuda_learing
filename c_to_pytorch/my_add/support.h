// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once
#include <cuda_fp16.h>
#include <torch/extension.h>
// clang-format off
#define CHECK_CUDA(x) TORCH_CHECK(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// clang-format on

template <typename U>
struct native_type {
    using T = U;
};

template <>
struct native_type<c10::Half> {
    using T = __half;
};

template <typename U>
typename native_type<U>::T *ptr(torch::Tensor t) {
    return reinterpret_cast<typename native_type<U>::T *>(t.data_ptr<U>());
}
