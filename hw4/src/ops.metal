#include <metal_stdlib>
using namespace metal;

typedef float scalar_t;

#define MAX_VEC_SIZE 8
#define TILE_SIZE 8
#define BLOCK_DIM 32

// Helper function to calculate memory index from strides
size_t calc_index(uint index,
                 device const int32_t* shape,
                 device const int32_t* strides,
                 device const size_t* dim,
                 device const size_t* offset) {
    size_t res = *offset;
    size_t left = index;
    
    for (int dim_idx = (int)(*dim) - 1; dim_idx >= 0; dim_idx--) {
        size_t coord = left % shape[dim_idx];
        res += coord * strides[dim_idx];
        left /= shape[dim_idx];
    }
    return res;
}

////////////////////////////////////////////////////////////////////////////////
// Fill kernel
////////////////////////////////////////////////////////////////////////////////

kernel void fill(device scalar_t *out [[buffer(0)]],
                 constant scalar_t& val [[buffer(1)]],
                 uint index [[thread_position_in_grid]]) {
    out[index] = val;
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem kernels
////////////////////////////////////////////////////////////////////////////////

kernel void compact(device const scalar_t* a [[buffer(0)]],
                    device scalar_t* out [[buffer(1)]],
                    device const int32_t* shape [[buffer(2)]],
                    device const int32_t* strides [[buffer(3)]],
                    device const size_t* dim [[buffer(4)]],
                    device const size_t* offset [[buffer(5)]],
                    uint index [[thread_position_in_grid]])
{
    out[index] = a[calc_index(index, shape, strides, dim, offset)]; 
}

kernel void ewise_setitem(device const scalar_t* a [[buffer(0)]],
                          device scalar_t* out [[buffer(1)]],
                          device const int32_t* shape [[buffer(2)]],
                          device const int32_t* strides [[buffer(3)]],
                          device const size_t* dim [[buffer(4)]],
                          device const size_t* offset [[buffer(5)]],
                          uint index [[thread_position_in_grid]]) {
    size_t out_pos = calc_index(index, shape, strides, dim, offset);
    out[out_pos] = a[index];
}

kernel void scalar_setitem(constant scalar_t& val [[buffer(0)]],
                           device scalar_t* out [[buffer(1)]],
                           device const int32_t* shape [[buffer(2)]],
                           device const int32_t* strides [[buffer(3)]],
                           device const size_t* dim [[buffer(4)]],
                           device const size_t* offset [[buffer(5)]],
                           uint index [[thread_position_in_grid]]) {
    size_t out_pos = calc_index(index, shape, strides, dim, offset);
    out[out_pos] = val;
}

////////////////////////////////////////////////////////////////////////////////
// Element-wise binary operations
////////////////////////////////////////////////////////////////////////////////

kernel void ewise_add(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = a[index] + b[index];
}

kernel void ewise_mul(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = a[index] * b[index];
}

kernel void ewise_div(device const scalar_t* a [[buffer(0)]],
                      device const scalar_t* b [[buffer(1)]],
                      device scalar_t* out [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = a[index] / b[index];
}

kernel void ewise_maximum(device const scalar_t* a [[buffer(0)]],
                          device const scalar_t* b [[buffer(1)]],
                          device scalar_t* out [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    out[index] = fmax(a[index], b[index]);
}

kernel void ewise_eq(device const scalar_t* a [[buffer(0)]],
                     device const scalar_t* b [[buffer(1)]],
                     device scalar_t* out [[buffer(2)]],
                     uint index [[thread_position_in_grid]]) {
    out[index] = (a[index] == b[index]) ? 1.0f : 0.0f;
}

kernel void ewise_ge(device const scalar_t* a [[buffer(0)]],
                     device const scalar_t* b [[buffer(1)]],
                     device scalar_t* out [[buffer(2)]],
                     uint index [[thread_position_in_grid]]) {
    out[index] = (a[index] >= b[index]) ? 1.0f : 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
// Scalar operations
////////////////////////////////////////////////////////////////////////////////

kernel void scalar_add(device const scalar_t* a [[buffer(0)]],
                       constant scalar_t& val [[buffer(1)]],
                       device scalar_t* out [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    out[index] = a[index] + val;
}

kernel void scalar_mul(device const scalar_t* a [[buffer(0)]],
                       constant scalar_t& val [[buffer(1)]],
                       device scalar_t* out [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    out[index] = a[index] * val;
}

kernel void scalar_div(device const scalar_t* a [[buffer(0)]],
                       constant scalar_t& val [[buffer(1)]],
                       device scalar_t* out [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    out[index] = a[index] / val;
}

kernel void scalar_power(device const scalar_t* a [[buffer(0)]],
                         constant scalar_t& val [[buffer(1)]],
                         device scalar_t* out [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    out[index] = pow(a[index], val);
}

kernel void scalar_maximum(device const scalar_t* a [[buffer(0)]],
                           constant scalar_t& val [[buffer(1)]],
                           device scalar_t* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    out[index] = fmax(a[index], val);
}

kernel void scalar_eq(device const scalar_t* a [[buffer(0)]],
                      constant scalar_t& val [[buffer(1)]],
                      device scalar_t* out [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = (a[index] == val) ? 1.0f : 0.0f;
}

kernel void scalar_ge(device const scalar_t* a [[buffer(0)]],
                      constant scalar_t& val [[buffer(1)]],
                      device scalar_t* out [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = (a[index] >= val) ? 1.0f : 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
// Unary operations
////////////////////////////////////////////////////////////////////////////////

kernel void ewise_log(device const scalar_t* a [[buffer(0)]],
                      device scalar_t* out [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = log(a[index]);
}

kernel void ewise_exp(device const scalar_t* a [[buffer(0)]],
                      device scalar_t* out [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {
    out[index] = exp(a[index]);
}

kernel void ewise_tanh(device const scalar_t* a [[buffer(0)]],
                       device scalar_t* out [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    out[index] = tanh(a[index]);
}

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication
////////////////////////////////////////////////////////////////////////////////


kernel void matmul_tiled(device const scalar_t* a [[buffer(0)]],
                         device const scalar_t* b [[buffer(1)]],
                         device scalar_t* out [[buffer(2)]],
                         device const uint32_t* M [[buffer(3)]],
                         device const uint32_t* N [[buffer(4)]],
                         device const uint32_t* P [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]],
                         uint2 bid [[threadgroup_position_in_grid]]) {
    uint m = *M;
    uint n = *N;
    uint p = *P;

    uint globalRow = bid.x * BLOCK_DIM + tid.x;
    uint globalCol = bid.y * BLOCK_DIM + tid.y;

    threadgroup scalar_t As[BLOCK_DIM][BLOCK_DIM];
    threadgroup scalar_t Bs[BLOCK_DIM][BLOCK_DIM];

    scalar_t acc = 0.0f;
    uint numTiles = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * BLOCK_DIM + tid.y;
        uint bRow = t * BLOCK_DIM + tid.x;

        As[tid.x][tid.y] = (globalRow < m && aCol < n) ? a[globalRow * n + aCol] : 0.0f;
        Bs[tid.x][tid.y] = (bRow < n && globalCol < p) ? b[bRow * p + globalCol] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_DIM; ++k) {
            acc += As[tid.x][k] * Bs[k][tid.y];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (globalRow < m && globalCol < p) {
        out[globalRow * p + globalCol] = acc;
    }
}

// Optimized matmul with better memory coalescing and larger blocks
kernel void matmul(device const scalar_t* a [[buffer(0)]],
                             device const scalar_t* b [[buffer(1)]],
                             device scalar_t* out [[buffer(2)]],
                             device const uint32_t* M [[buffer(3)]],
                             device const uint32_t* N [[buffer(4)]],
                             device const uint32_t* P [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 bid [[threadgroup_position_in_grid]]) {
    uint m = *M;
    uint n = *N;
    uint p = *P;

    // Each thread computes multiple elements for better arithmetic intensity
    const uint TILE_M = 4;
    const uint TILE_N = 4;
    
    uint baseRow = bid.x * BLOCK_DIM * TILE_M + tid.x * TILE_M;
    uint baseCol = bid.y * BLOCK_DIM * TILE_N + tid.y * TILE_N;

    threadgroup scalar_t As[BLOCK_DIM * TILE_M][BLOCK_DIM];
    threadgroup scalar_t Bs[BLOCK_DIM][BLOCK_DIM * TILE_N];

    scalar_t acc[TILE_M][TILE_N];
    for (uint i = 0; i < TILE_M; i++) {
        for (uint j = 0; j < TILE_N; j++) {
            acc[i][j] = 0.0f;
        }
    }

    uint numTiles = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    for (uint t = 0; t < numTiles; ++t) {
        // Load A tile with coalesced access
        for (uint i = 0; i < TILE_M; i++) {
            uint row = baseRow + i;
            uint col = t * BLOCK_DIM + tid.y;
            As[tid.x * TILE_M + i][tid.y] = (row < m && col < n) ? a[row * n + col] : 0.0f;
        }

        // Load B tile with coalesced access
        for (uint j = 0; j < TILE_N; j++) {
            uint row = t * BLOCK_DIM + tid.x;
            uint col = baseCol + j;
            Bs[tid.x][tid.y * TILE_N + j] = (row < n && col < p) ? b[row * p + col] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute with higher arithmetic intensity
        for (uint k = 0; k < BLOCK_DIM; ++k) {
            for (uint i = 0; i < TILE_M; i++) {
                for (uint j = 0; j < TILE_N; j++) {
                    acc[i][j] += As[tid.x * TILE_M + i][k] * Bs[k][tid.y * TILE_N + j];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    for (uint i = 0; i < TILE_M; i++) {
        for (uint j = 0; j < TILE_N; j++) {
            uint row = baseRow + i;
            uint col = baseCol + j;
            if (row < m && col < p) {
                out[row * p + col] = acc[i][j];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reduction operations
////////////////////////////////////////////////////////////////////////////////

kernel void reduce_max(device const scalar_t* a [[buffer(0)]],
                       device scalar_t* out [[buffer(1)]],
                       device const size_t* reduce_size [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    size_t base = index * (*reduce_size);
    scalar_t curr_max = a[base];
    for (size_t j = 0; j < *reduce_size; j++) {
        curr_max = fmax(curr_max, a[base + j]);
    }
    out[index] = curr_max;
}

kernel void reduce_sum(device const scalar_t* a [[buffer(0)]],
                       device scalar_t* out [[buffer(1)]],
                       device const size_t* reduce_size [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    size_t base = index * (*reduce_size);
    scalar_t s = 0.0f;
    for (size_t j = 0; j < *reduce_size; j++) {
        s += a[base + j];
    }
    out[index] = s;
}