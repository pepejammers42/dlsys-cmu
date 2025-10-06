#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/NSString.hpp"
#include "Foundation/NSError.hpp"
#include "Metal/MTLDevice.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLLibrary.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLBuffer.hpp"
#include "Metal/MTLTypes.hpp"

#include "metal_ops.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <functional>

namespace needle {
namespace metal {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

MTL::Device *device = MTL::CreateSystemDefaultDevice();

static MTL::CommandQueue* g_queue = nullptr;

// Match ops.metal
#define BLOCK_DIM 32

struct MetalArray {
    MetalArray(const size_t size){
        buffer = device -> newBuffer(size * ELEM_SIZE, MTL::ResourceStorageModeShared);
        if (!buffer) throw std::bad_alloc();
        ptr = (scalar_t *) buffer -> contents();
        this -> size = size;
    }
    ~MetalArray() { buffer -> release(); }
    MTL::Buffer *buffer;
    size_t ptr_as_int() {return (size_t) ptr;}
    scalar_t *ptr;
    size_t size;
};

// ---------------- Pipeline cache implementation ----------------
namespace {
struct PipelineCacheImpl {
  std::unordered_map<std::string, MTL::ComputePipelineState*> name_to_ps;
  MTL::Library* library {nullptr};
  std::mutex mtx;
  ~PipelineCacheImpl(){
    for (auto& kv : name_to_ps) {
      if (kv.second) kv.second->release();
    }
    if (library) library->release();
  }
};

PipelineCacheImpl* g_cache = nullptr;

std::string read_file_text(const char* path){
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) throw std::runtime_error("Failed to open Metal source file");
  std::ostringstream ss; ss << ifs.rdbuf();
  return ss.str();
}

std::string dirname_of(const char* path) {
  std::string p(path);
  auto pos = p.find_last_of("/\\");
  if (pos == std::string::npos) return std::string(".");
  return p.substr(0, pos);
}

void ensure_initialized() {
  if (!device) throw std::runtime_error("Metal device not available");
  if (!g_queue) {
    g_queue = device->newCommandQueue();
  }
  if (!g_cache) {
    g_cache = new PipelineCacheImpl();

    // build library from ops.metal next to this file
    std::string dir = dirname_of(__FILE__);
    std::string source_path = dir + "/ops.metal";
    std::string src = read_file_text(source_path.c_str());
    NS::String* ns_src = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
    NS::Error* err = nullptr;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(true);
    // // Explicitly set a conservative Metal language version for wider compatibility
    opts->setLanguageVersion(MTL::LanguageVersion3_0);
    g_cache->library = device->newLibrary(ns_src, opts, &err);
    opts->release();
    if (!g_cache->library || err) {
      std::string msg = "Failed to compile Metal library from ops.metal";
      throw std::runtime_error(msg);
    }
  }
}

// ---------------- Simple async submission tracking & sync ----------------
static std::mutex g_pending_mtx;
static std::vector<MTL::CommandBuffer*> g_pending_cbs;

inline void track_cb(MTL::CommandBuffer* cb){
  if (!cb) return;
  std::lock_guard<std::mutex> lock(g_pending_mtx);
  cb->retain();
  g_pending_cbs.push_back(cb);
}

inline void synchronize_pending(){
  std::vector<MTL::CommandBuffer*> local;
  {
    std::lock_guard<std::mutex> lock(g_pending_mtx);
    local.swap(g_pending_cbs);
  }
  for (auto* cb : local){
    cb->waitUntilCompleted();
    cb->release();
  }
}

MTL::ComputePipelineState* get_pipeline(const char* name) {
  ensure_initialized();
  std::lock_guard<std::mutex> lock(g_cache->mtx);
  auto it = g_cache->name_to_ps.find(name);
  if (it != g_cache->name_to_ps.end()) return it->second;
  NS::Error* err = nullptr;
  NS::String* fn = NS::String::string(name, NS::UTF8StringEncoding);
  MTL::Function* f = g_cache->library->newFunction(fn);
  if (!f) {
    // Enumerate available function names to aid debugging
    std::string available;
    if (auto* arr = g_cache->library->functionNames()) {
      for (NS::UInteger i = 0; i < arr->count(); ++i) {
        auto* s = (NS::String*)arr->object(i);
        if (s && s->utf8String()) {
          if (!available.empty()) available += ", ";
          available += s->utf8String();
        }
      }
    }
    throw std::runtime_error(std::string("Metal kernel not found: ") + name + (available.empty() ? std::string("") : std::string(". Available: ") + available));
  }
  MTL::ComputePipelineState* ps = device->newComputePipelineState(f, &err);
  f->release();
  if (!ps || err) throw std::runtime_error(std::string("Failed to create pipeline: ") + name);
  g_cache->name_to_ps.emplace(name, ps);
  return ps;
}

inline MTL::Size threads_per_threadgroup_1d(MTL::ComputePipelineState* ps){
  NS::UInteger w = ps->threadExecutionWidth();
  return MTL::Size::Make(w, 1, 1);
}

inline void dispatch_1d(MTL::ComputePipelineState* ps, NS::UInteger total_threads, const std::function<void(MTL::ComputeCommandEncoder*)>& bind){
  if (total_threads == 0) return;
  auto* cb = g_queue->commandBuffer();
  auto* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(ps);
  bind(enc);
  MTL::Size grid = MTL::Size::Make(total_threads, 1, 1);
  MTL::Size tpg = threads_per_threadgroup_1d(ps);
  enc->dispatchThreads(grid, tpg);
  enc->endEncoding();
  cb->commit();
  track_cb(cb);
  enc->release();
  cb->release();
}

inline void set_scalar_bytes(MTL::ComputeCommandEncoder* enc, const void* data, size_t bytes, NS::UInteger index){
  enc->setBytes(data, bytes, index);
}

inline MTL::Buffer* make_buffer(const void* data, size_t bytes){
  if (bytes == 0) return nullptr;
  MTL::Buffer* buf = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
  std::memcpy(buf->contents(), data, bytes);
  return buf;
}
}

void Fill(MetalArray *out, scalar_t val){
  auto* ps = get_pipeline(kernels::kFill);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(out->buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
  });
}

void Compact(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
  std::vector<int32_t> strides, size_t offset) {
  auto* ps = get_pipeline(kernels::kCompact);
  size_t dim_val = shape.size();
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
    enc->setBytes(shape.data(), shape.size()*sizeof(int32_t), 2);
    enc->setBytes(strides.data(), strides.size()*sizeof(int32_t), 3);
    enc->setBytes(&dim_val, sizeof(size_t), 4);
    enc->setBytes(&offset, sizeof(size_t), 5);
  });
}

void EwiseSetitem(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
       std::vector<int32_t> strides, size_t offset) {
  auto* ps = get_pipeline(kernels::kEwiseSetitem);
  size_t dim_val = shape.size();
  dispatch_1d(ps, (NS::UInteger)a.size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
    enc->setBytes(shape.data(), shape.size()*sizeof(int32_t), 2);
    enc->setBytes(strides.data(), strides.size()*sizeof(int32_t), 3);
    enc->setBytes(&dim_val, sizeof(size_t), 4);
    enc->setBytes(&offset, sizeof(size_t), 5);
  });
}

void ScalarSetitem(const size_t size, scalar_t val, MetalArray* out, std::vector<int32_t> shape,
        std::vector<int32_t> strides, size_t offset) {
  auto* ps = get_pipeline(kernels::kScalarSetitem);
  size_t dim_val = shape.size();
  dispatch_1d(ps, (NS::UInteger)size, [&](MTL::ComputeCommandEncoder* enc){
    set_scalar_bytes(enc, &val, sizeof(val), 0);
    enc->setBuffer(out->buffer, 0, 1);
    enc->setBytes(shape.data(), shape.size()*sizeof(int32_t), 2);
    enc->setBytes(strides.data(), strides.size()*sizeof(int32_t), 3);
    enc->setBytes(&dim_val, sizeof(size_t), 4);
    enc->setBytes(&offset, sizeof(size_t), 5);
  });
}

void EwiseAdd(const MetalArray& a, const MetalArray& b, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseAdd);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(b.buffer, 0, 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarAdd(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarAdd);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void EwiseMul(const MetalArray& a, const MetalArray& b, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseMul);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(b.buffer, 0, 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarMul(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarMul);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void EwiseDiv(const MetalArray& a, const MetalArray& b, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseDiv);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(b.buffer, 0, 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarDiv(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarDiv);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarPower(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarPower);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void EwiseMaximum(const MetalArray& a, const MetalArray& b, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseMaximum);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(b.buffer, 0, 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarMaximum(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarMaximum);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void EwiseEq(const MetalArray& a, const MetalArray& b, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseEq);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(b.buffer, 0, 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarEq(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarEq);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void EwiseGe(const MetalArray& a, const MetalArray& b, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseGe);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(b.buffer, 0, 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void ScalarGe(const MetalArray& a, scalar_t val, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kScalarGe);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    set_scalar_bytes(enc, &val, sizeof(val), 1);
    enc->setBuffer(out->buffer, 0, 2);
  });
}

void EwiseLog(const MetalArray& a, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseLog);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
  });
}

void EwiseExp(const MetalArray& a, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseExp);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
  });
}

void EwiseTanh(const MetalArray& a, MetalArray* out) {
  auto* ps = get_pipeline(kernels::kEwiseTanh);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
  });
}

void Matmul(const MetalArray& a, const MetalArray& b, MetalArray* out, uint32_t m, uint32_t n,
 uint32_t p) {
  auto* ps = get_pipeline(kernels::kMatmul);
  auto* cb = g_queue->commandBuffer();
  auto* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(ps);
  enc->setBuffer(a.buffer, 0, 0);
  enc->setBuffer(b.buffer, 0, 1);
  enc->setBuffer(out->buffer, 0, 2);
  enc->setBytes(&m, sizeof(uint32_t), 3);
  enc->setBytes(&n, sizeof(uint32_t), 4);
  enc->setBytes(&p, sizeof(uint32_t), 5);
  // Each thread computes 4x4 elements, so we need fewer threadgroups
  NS::UInteger groups_x = (m + BLOCK_DIM * 4 - 1) / (BLOCK_DIM * 4);
  NS::UInteger groups_y = (p + BLOCK_DIM * 4 - 1) / (BLOCK_DIM * 4);
  MTL::Size tg = MTL::Size::Make(BLOCK_DIM, BLOCK_DIM, 1);
  MTL::Size tg_grid = MTL::Size::Make(groups_x, groups_y, 1);
  enc->dispatchThreadgroups(tg_grid, tg);
  enc->endEncoding();
  cb->commit();
  track_cb(cb);
  enc->release();
  cb->release();
}

void MatmulTiled(const MetalArray& a, const MetalArray& b, MetalArray* out, uint32_t m,
      uint32_t n, uint32_t p) {
  auto* ps = get_pipeline(kernels::kMatmulTiled);
  auto* cb = g_queue->commandBuffer();
  auto* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(ps);
  enc->setBuffer(a.buffer, 0, 0);
  enc->setBuffer(b.buffer, 0, 1);
  enc->setBuffer(out->buffer, 0, 2);
  enc->setBytes(&m, sizeof(uint32_t), 3);
  enc->setBytes(&n, sizeof(uint32_t), 4);
  enc->setBytes(&p, sizeof(uint32_t), 5);
  NS::UInteger groups_x = (m + BLOCK_DIM - 1) / BLOCK_DIM;
  NS::UInteger groups_y = (p + BLOCK_DIM - 1) / BLOCK_DIM;
  MTL::Size tg = MTL::Size::Make(BLOCK_DIM, BLOCK_DIM, 1);
  MTL::Size tg_grid = MTL::Size::Make(groups_x, groups_y, 1);
  enc->dispatchThreadgroups(tg_grid, tg);
  enc->endEncoding();
  cb->commit();
  track_cb(cb);
  enc->release();
  cb->release();
}


void ReduceMax(const MetalArray& a, MetalArray* out, size_t reduce_size) {
  auto* ps = get_pipeline(kernels::kReduceMax);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
    enc->setBytes(&reduce_size, sizeof(size_t), 2);
  });
}

void ReduceSum(const MetalArray& a, MetalArray* out, size_t reduce_size) {
  auto* ps = get_pipeline(kernels::kReduceSum);
  dispatch_1d(ps, (NS::UInteger)out->size, [&](MTL::ComputeCommandEncoder* enc){
    enc->setBuffer(a.buffer, 0, 0);
    enc->setBuffer(out->buffer, 0, 1);
    enc->setBytes(&reduce_size, sizeof(size_t), 2);
  });
}

}
}


PYBIND11_MODULE(ndarray_backend_metal, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace metal;
  
    m.attr("__device_name__") = "metal";
    m.attr("__tile_size__") = TILE;
  
    py::class_<MetalArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &MetalArray::ptr_as_int)
        .def_readonly("size", &MetalArray::size);
  
    // return numpy array (with copying for simplicity, otherwise garbage
    // collection is a pain)
    m.def("to_numpy", [](const MetalArray& a, std::vector<size_t> shape,
                         std::vector<size_t> strides, size_t offset) {
      // ensure GPU work that may write to 'a' is complete before host reads
      synchronize_pending();
      std::vector<size_t> numpy_strides = strides;
      std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                     [](size_t& c) { return c * ELEM_SIZE; });
      return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
    });
  
    // convert from numpy (with copying)
    m.def("from_numpy", [](py::array_t<scalar_t> a, MetalArray* out) {
      // synchronize first to avoid writing while GPU may still read
      synchronize_pending();
      std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
    });
  
    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);
  
    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);
  
    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);
  
    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);
  
    m.def("matmul", Matmul);
  
    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
  }
  