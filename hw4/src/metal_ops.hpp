#pragma once


namespace needle {
namespace metal {

// Kernel names as in ops.metal
namespace kernels {
static constexpr const char* kFill = "fill";
static constexpr const char* kCompact = "compact";
static constexpr const char* kEwiseSetitem = "ewise_setitem";
static constexpr const char* kScalarSetitem = "scalar_setitem";

static constexpr const char* kEwiseAdd = "ewise_add";
static constexpr const char* kScalarAdd = "scalar_add";
static constexpr const char* kEwiseMul = "ewise_mul";
static constexpr const char* kScalarMul = "scalar_mul";
static constexpr const char* kEwiseDiv = "ewise_div";
static constexpr const char* kScalarDiv = "scalar_div";
static constexpr const char* kScalarPower = "scalar_power";
static constexpr const char* kEwiseMaximum = "ewise_maximum";
static constexpr const char* kScalarMaximum = "scalar_maximum";
static constexpr const char* kEwiseEq = "ewise_eq";
static constexpr const char* kScalarEq = "scalar_eq";
static constexpr const char* kEwiseGe = "ewise_ge";
static constexpr const char* kScalarGe = "scalar_ge";

static constexpr const char* kEwiseLog = "ewise_log";
static constexpr const char* kEwiseExp = "ewise_exp";
static constexpr const char* kEwiseTanh = "ewise_tanh";

static constexpr const char* kMatmul = "matmul";
static constexpr const char* kMatmulTiled = "matmul_tiled";

static constexpr const char* kReduceMax = "reduce_max";
static constexpr const char* kReduceSum = "reduce_sum";
}

}
}
