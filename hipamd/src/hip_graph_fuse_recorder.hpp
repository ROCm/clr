#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_graph_internal.hpp"
#include "platform/kernel.hpp"


namespace hip {
class GraphFuseRecorder {
  amd::Monitor fclock_{"Guards Graph Fuse-Recorder object", true};

 public:
  GraphFuseRecorder(hip::Graph* graph);
  void run();
  static bool isRecordingOn();
  static hipKernelNodeParams getKernelNodeParams(GraphNode* node);
  static amd::Kernel* getDeviceKernel(hipKernelNodeParams& nodeParams);
  static amd::Kernel* getDeviceKernel(GraphNode* node);

 private:
  struct ImageHandle {
    char* image_{};
    size_t imageSize_{};
    bool isAllocated_{};
    std::string fileName_{};
    bool operator==(const ImageHandle& other) const {
      return (this->image_ == other.image_) && (imageSize_ == other.imageSize_);
    }
  };
  struct ImageHash {
    template <class T> static void hashCombine(std::size_t& seed, const T& value) {
      std::hash<T> hasher;
      seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    size_t operator()(const ImageHandle& info) const {
      size_t result{0};
      hashCombine(result, info.image_);
      hashCombine(result, info.imageSize_);
      return result;
    }
  };
  using ImageCacheType = std::unordered_set<ImageHandle, ImageHash>;

  struct KernelDescription {
    std::string name{};
    std::string location{};
    dim3 gridDim{};
    std::vector<size_t> argsSizes{};
    std::vector<int> argsTypes{};
  };
  using KernelDescriptions = std::vector<KernelDescription>;


  static bool isInputOk();
  static std::string generateFilePath(const std::string& name);
  static std::string generateImagePath(size_t imageId);

  bool findCandidates(const std::vector<Node>& nodes);
  KernelDescriptions collectImages(const std::vector<Node>& nodes);
  void saveImageToDisk(ImageHandle& imageHandle);
  void saveFusionConfig(std::vector<KernelDescriptions>& groupDescriptions);

  hip::Graph* graph_;
  std::vector<std::vector<Node>> fusionGroups_{};
  std::vector<std::vector<size_t>> fusedExecutionOrder_{};
  size_t instanceId_{};

  struct Finalizer {
    void operator()(size_t*) const;
  };
  using CounterType = std::unique_ptr<size_t, Finalizer>;

  static CounterType instanceCounter_;
  static bool isRecordingStateQueried_;
  static bool isRecordingSwitchedOn_;
  static std::string tmpDirName_;
  static ImageCacheType imageCache_;
  static std::vector<std::string> savedFusionConfigs_;
};
}  // namespace hip
