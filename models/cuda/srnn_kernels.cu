#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return z > 0 ? z : 0;
}

template <typename scalar_t>
__global__ void roll_sum_relu_first_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> hidden,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> res) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  const auto hidden_size = input.size(2);
  if (c < hidden_size) {
    int h_idx = c - 1;
    if (h_idx < 0) {
      h_idx = hidden_size - 1;
    }
    res[n][0][c] = relu(input[n][0][c] + hidden[n][h_idx]);
  }
}

template <typename scalar_t>
__global__ void roll_sum_relu_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> res,
    unsigned int seq_idx) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  const auto hidden_size = input.size(2);
  if (c < hidden_size) {
    int h_idx = c - 1;
    if (h_idx < 0) {
      h_idx = hidden_size - 1;
    }
    res[n][seq_idx][c] = relu(input[n][seq_idx][c] + res[n][seq_idx - 1][h_idx]);
  }
}

template <typename scalar_t>
__global__ void calc_roll_grad_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> loss_grad,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> loss_grad_seq,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> outputs_results,
    unsigned int s_len) {
  //batch index
  const int b = blockIdx.y;
  // column index
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  const auto hidden_size = outputs_results.size(2);
  if (c >= hidden_size) {
    return;
  }

  for (int seq_idx = s_len - 2; seq_idx >= 0 ; seq_idx--) {
    auto value = loss_grad_seq[b][seq_idx + 1][c];
    c--;
    if (c < 0) {
      c = hidden_size - 1;
    }
    auto out_res_value = outputs_results[b][seq_idx][c];

    if (loss_grad.size(1) != 1) {
        auto grad_loss_value = loss_grad[b][seq_idx][c];
        loss_grad_seq[b][seq_idx][c] = (value + grad_loss_value) * (out_res_value > 0);
    } else {
        loss_grad_seq[b][seq_idx][c] = value * (out_res_value > 0);
    }
  }
}

} // namespace

void calc_roll_grad_cuda(
    const torch::Tensor &loss_grad,
    torch::Tensor &loss_grad_seq,
    const torch::Tensor &outputs_results) {
  auto b_size = outputs_results.size(0);
  auto s_len = outputs_results.size(1);
  auto h_size = outputs_results.size(2);
  const int threads = 1024;
  const dim3 blocks((h_size + threads - 1) / threads, b_size);
  AT_DISPATCH_FLOATING_TYPES(loss_grad_seq.type(), "roll_sum_relu_cuda", ([&] {
    calc_roll_grad_cuda_kernel<scalar_t><<<blocks, threads>>>(
      loss_grad.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      loss_grad_seq.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      outputs_results.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      s_len);
  }));
  cudaDeviceSynchronize();
}

torch::Tensor roll_sum_relu_cuda(
    torch::Tensor input,
    torch::Tensor hidden) {

  const auto batch_size = input.size(0);
  const auto seq_len = input.size(1);
  const auto input_size = input.size(2);

  auto res = torch::empty_like(input);

  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "roll_sum_relu_cuda", ([&] {
      roll_sum_relu_first_cuda_kernel<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          hidden.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          res.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
    }));
  cudaDeviceSynchronize();
  for (auto seq_idx = 1; seq_idx < seq_len; seq_idx++) {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "roll_sum_relu_cuda", ([&] {
      roll_sum_relu_cuda_kernel<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          res.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          seq_idx);
    }));
    cudaDeviceSynchronize();
  }

  return res;
}