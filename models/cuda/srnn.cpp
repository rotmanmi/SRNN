#include <torch/script.h>

#include <vector>

// CUDA forward declarations

torch::Tensor roll_sum_relu_cuda(
    torch::Tensor input,
    torch::Tensor hidden);

void calc_roll_grad_cuda(
    const torch::Tensor &loss_grad,
    torch::Tensor &loss_grad_seq,
    const torch::Tensor &outputs_results);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); // CHECK_CONTIGUOUS(x)
#define CHECK_SAME_SIZE(x, h) AT_ASSERTM(x.is_same_size(h), #x " and " #h " must have same shape")

torch::Tensor roll_sum_relu(
    const torch::Tensor &input,
    const torch::Tensor &hidden) {
  CHECK_INPUT(input);
  CHECK_INPUT(hidden);
  // CHECK_SAME_SIZE(input, hidden);

  return roll_sum_relu_cuda(input, hidden);
}

torch::Tensor calc_roll_grad(
    const torch::Tensor &loss_grad,
    const torch::Tensor &outputs_results) {
  CHECK_INPUT(loss_grad);
  CHECK_INPUT(outputs_results);

  auto loss_grad_seq = torch::zeros_like(outputs_results);
  auto b_size = outputs_results.size(0);
  auto s_len = outputs_results.size(1);
  auto h_size = outputs_results.size(2);

  auto a = torch::narrow(loss_grad_seq, 1, s_len - 1, 1);
  auto b = torch::narrow(outputs_results, 1, s_len - 1, 1);
  auto c = torch::narrow(loss_grad, 1, loss_grad.size(1) - 1, 1);
  a += c * (b > 0);
  calc_roll_grad_cuda(loss_grad, loss_grad_seq, outputs_results);

  return std::move(loss_grad_seq);
}

static auto registry =
  torch::RegisterOperators("srnn::roll_sum_relu", &roll_sum_relu)
  .op("srnn::calc_roll_grad", &calc_roll_grad);