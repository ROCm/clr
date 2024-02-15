#include "hip_graph_fuse_recorder.hpp"
#include "hip_global.hpp"
#include "hip_internal.hpp"
#include "utils/debug.hpp"
#include <stdlib.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <filesystem>
#include <yaml-cpp/emittermanip.h>
#include <yaml-cpp/yaml.h>


namespace {
void rtrim(std::string& str) { str.erase(std::find(str.begin(), str.end(), '\0'), str.end()); }

bool enabled(const std::string& value) {
  std::string lowercaseValue(value.size(), '\0');
  std::transform(value.begin(), value.end(), lowercaseValue.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  rtrim(lowercaseValue);
  static std::unordered_set<std::string> options{"1", "enable", "enabled", "yes", "true", "on"};
  if (options.find(lowercaseValue) != options.end()) {
    return true;
  }
  return false;
}

bool equal(const dim3& one, const dim3& two) {
  return (one.x == two.x) && (one.y == two.y) && (one.z == two.z);
}

template <typename T> void append(std::vector<T>& vec) { vec.push_back(T()); };
}  // namespace


namespace hip {
bool GraphFuseRecorder::isRecordingStateQueried_{false};
bool GraphFuseRecorder::isRecordingSwitchedOn_{false};
std::string GraphFuseRecorder::tmpDirName_{};
GraphFuseRecorder::ImageCacheType GraphFuseRecorder::imageCache_{};
std::vector<std::string> GraphFuseRecorder::savedFusionConfigs_{};
GraphFuseRecorder::CounterType GraphFuseRecorder::instanceCounter_{nullptr};

GraphFuseRecorder::GraphFuseRecorder(hip::Graph* graph) : graph_(graph) {
  amd::ScopedLock lock(fclock_);
  if (GraphFuseRecorder::instanceCounter_ == nullptr) {
    GraphFuseRecorder::instanceCounter_ = GraphFuseRecorder::CounterType{new size_t};
    *(GraphFuseRecorder::instanceCounter_) = 0;
  }
  instanceId_ = (*GraphFuseRecorder::instanceCounter_)++;
}

void GraphFuseRecorder::Finalizer::operator()(size_t* instanceCounter) const {
  std::filesystem::path path(GraphFuseRecorder::tmpDirName_);
  YAML::Emitter out;
  out << YAML::BeginSeq;
  for (const auto& item : GraphFuseRecorder::savedFusionConfigs_) {
    out << item;
  }
  out << YAML::EndSeq;

  auto filePath = generateFilePath("config.yaml");
  auto configFile = std::fstream(filePath.c_str(), std::ios_base::out);
  if (configFile) {
    configFile << out.c_str() << "\n";
  } else {
    std::stringstream msg;
    msg << "failed to write yaml manifest file to `" << filePath.c_str() << "`";
    LogPrintfError("%s", msg.str().c_str());
  }
  delete instanceCounter;
}

bool GraphFuseRecorder::isInputOk() {
  auto* env = getenv("AMD_FUSION_RECORDING");
  if (env == nullptr) {
    return false;
  }
  if (!enabled(std::string(env))) {
    return false;
  }

  env = getenv("AMD_FUSION_CONFIG");
  if (env == nullptr) {
    std::stringstream msg;
    LogPrintfError("%s", "fusion config is not specified; cannot proceed fusion recording");
    return false;
  }

  std::string configPathName(env);
  std::filesystem::path dirPath(configPathName);
  if (!std::filesystem::exists(dirPath)) {
    LogPrintfError("cannot open fusion config file: %s", configPathName.c_str());
    return false;
  }

  try {
    auto config = YAML::LoadFile(configPathName);
    auto tmpDirName = config["tmp_dir_path"].as<std::string>();
    std::filesystem::path dirPath(tmpDirName);
    dirPath /= "recording";
    if (!std::filesystem::exists(dirPath)) {
      auto isOk = std::filesystem::create_directories(dirPath);
      if (!isOk) {
        LogPrintfError("failed to create a tmp dir: %s", dirPath.c_str());
        return false;
      }
    }
    GraphFuseRecorder::tmpDirName_ = dirPath.string();
  } catch (const YAML::ParserException& ex) {
    LogPrintfError("error while parsing fusion config: %s", ex.what());
    return false;
  } catch (const std::runtime_error& ex) {
    LogPrintfError("error while parsing fusion config: %s", ex.what());
    return false;
  }
  LogPrintfInfo("%s", "graph fuse recorder is enabled");
  return true;
}

bool GraphFuseRecorder::isRecordingOn() {
  static amd::Monitor lock_;
  amd::ScopedLock lock(lock_);
  if (!isRecordingStateQueried_) {
    isRecordingSwitchedOn_ = GraphFuseRecorder::isInputOk();
    isRecordingStateQueried_ = true;
  }
  return isRecordingSwitchedOn_;
}

hipKernelNodeParams GraphFuseRecorder::getKernelNodeParams(GraphNode* node) {
  auto* kernelNode = dynamic_cast<GraphKernelNode*>(node);
  guarantee(kernelNode != nullptr, "failed to convert a graph node to `hipGraphKernelNode`");
  hipKernelNodeParams kernelParams;
  kernelNode->GetParams(&kernelParams);
  return kernelParams;
}

amd::Kernel* GraphFuseRecorder::getDeviceKernel(hipKernelNodeParams& nodeParams) {
  hipFunction_t hipFunc = GraphKernelNode::getFunc(nodeParams, ihipGetDevice());
  auto* deviceFunc = hip::DeviceFunc::asFunction(hipFunc);
  guarantee(deviceFunc != nullptr, "failed to retrieve the kernel of a graph node");

  auto* kernel = deviceFunc->kernel();
  return kernel;
}

amd::Kernel* GraphFuseRecorder::getDeviceKernel(GraphNode* node) {
  auto nodeParams = GraphFuseRecorder::getKernelNodeParams(node);
  return GraphFuseRecorder::getDeviceKernel(nodeParams);
}

void GraphFuseRecorder::run() {
  amd::ScopedLock lock(fclock_);
  const auto& nodes = graph_->GetNodes();
  if (!findCandidates(nodes)) {
    return;
  }

  std::vector<KernelDescriptions> groupDescriptions{};
  for (auto& group : fusionGroups_) {
    auto description = collectImages(group);
    groupDescriptions.push_back(description);
  }
  saveFusionConfig(groupDescriptions);
}

bool GraphFuseRecorder::findCandidates(const std::vector<Node>& nodes) {
  for (size_t i = 0; i < nodes.size() - 1; ++i) {
    auto& node = nodes[i];
    const auto outDegree = node->GetOutDegree();
    if (outDegree != 1) {
      std::stringstream msg;
      msg << "cannot perform fusion because node `" << i << "` contains multiple output edges. "
          << "Number of output edges equals " << outDegree;
      LogPrintfError("%s", msg.str().c_str());
      return false;
    }
  }

  fusionGroups_.push_back(std::vector<Node>());
  fusedExecutionOrder_.push_back(std::vector<size_t>());
  dim3 referenceBlockSize{};
  bool isRecording{true};
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    const auto type = node->GetType();
    const auto outDegree = node->GetOutDegree();

    if (type == hipGraphNodeTypeKernel) {
      if (!isRecording) {
        append(fusionGroups_);
        append(fusedExecutionOrder_);
      }
      isRecording = true;

      auto params = GraphFuseRecorder::getKernelNodeParams(node);
      auto* kernel = GraphFuseRecorder::getDeviceKernel(params);

      if (fusionGroups_.back().empty()) {
        referenceBlockSize = params.blockDim;
      }

      const bool isBlockSizeEqual = equal(referenceBlockSize, params.blockDim);
      if (!isBlockSizeEqual) {
        append(fusionGroups_);
        append(fusedExecutionOrder_);
      }
      fusionGroups_.back().push_back(node);
      fusedExecutionOrder_.back().push_back(i);
    }

    if (type != hipGraphNodeTypeKernel) {
      isRecording = false;

      append(fusedExecutionOrder_);
      fusedExecutionOrder_.back().push_back(i);
      continue;
    }
  }

  fusionGroups_.erase(std::remove_if(fusionGroups_.begin(), fusionGroups_.end(),
                                     [](auto& group) { return group.size() <= 1; }),
                      fusionGroups_.end());

  if (fusionGroups_.empty()) {
    LogPrintfError("%s", "could not find fusion candidates");
    return false;
  }

  fusedExecutionOrder_.erase(
      std::remove_if(fusedExecutionOrder_.begin(), fusedExecutionOrder_.end(),
                     [](auto& sequence) { return sequence.empty(); }),
      fusedExecutionOrder_.end());

  size_t nodeCounter{0};
  std::for_each(fusedExecutionOrder_.begin(), fusedExecutionOrder_.end(),
                [&nodeCounter](auto& item) { nodeCounter += item.size(); });
  guarantee(nodeCounter == nodes.size(), "failed to process execution sequences");
  return true;
}

GraphFuseRecorder::KernelDescriptions GraphFuseRecorder::collectImages(
    const std::vector<Node>& group) {
  const auto& devices = hip::getCurrentDevice()->devices();
  const auto currentDeviceId = ihipGetDevice();
  auto device = devices[currentDeviceId];

  KernelDescriptions descriptions{};
  for (size_t i = 0; i < group.size(); ++i) {
    KernelDescription descr{};

    const auto& node = group[i];
    auto params = GraphFuseRecorder::getKernelNodeParams(node);
    descr.gridDim = params.gridDim;

    auto* kernel = GraphFuseRecorder::getDeviceKernel(params);
    descr.name = kernel->name();
    rtrim(descr.name);

    auto& kernelargs = kernel->signature().parameters();
    for (uint32_t i = 0; i < kernel->signature().numParametersAll(); ++i) {
      auto& argDescriptor = kernelargs[i];
      descr.argsSizes.push_back(argDescriptor.size_);
      const auto& it = kernel->signature().at(i);
      descr.argsTypes.push_back(it.info_.oclObject_);
    }

    auto& program = kernel->program();
    auto [image, imageSize, isAllocated] = program.binary(*device);

    ImageHandle handle{};
    handle.image_ = reinterpret_cast<char*>(const_cast<uint8_t*>(image));
    handle.imageSize_ = static_cast<size_t>(imageSize);
    handle.isAllocated_ = isAllocated;

    if (imageCache_.find(handle) == imageCache_.end()) {
      const auto imageId = imageCache_.size();
      handle.fileName_ = GraphFuseRecorder::generateImagePath(imageId);
      imageCache_.insert(handle);
      saveImageToDisk(handle);
    }
    descr.location = imageCache_.find(handle)->fileName_;

    descriptions.push_back(descr);
  }
  return descriptions;
}

void GraphFuseRecorder::saveImageToDisk(ImageHandle& imageHandle) {
  if (imageHandle.imageSize_ > 0) {
    auto iamgeFile =
        std::fstream(imageHandle.fileName_.c_str(), std::ios_base::out | std::ios_base::binary);
    if (iamgeFile) {
      iamgeFile.write(imageHandle.image_, imageHandle.imageSize_);
    } else {
      std::stringstream msg;
      msg << "failed to write image file to `" << imageHandle.fileName_.c_str() << "`";
      LogPrintfError("%s", msg.str().c_str());
    }
  }
}

void GraphFuseRecorder::saveFusionConfig(std::vector<KernelDescriptions>& groupDescriptions) {
  const auto currentDeviceId = ihipGetDevice();
  hipDeviceProp_t props;
  auto status = hipGetDeviceProperties(&props, currentDeviceId);
  if (status != hipSuccess) {
    LogPrintfError("%s", "failed to call `hipGetDeviceProperties`");
  }

  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "device" << YAML::Value << std::string(props.gcnArchName);

  out << YAML::Key << "executionOrder" << YAML::Value << YAML::BeginSeq;
  for (auto& sequence : fusedExecutionOrder_) {
    out << YAML::Flow << YAML::BeginSeq;
    for (auto& item : sequence) {
      out << item;
    }
    out << YAML::EndSeq;
  }
  out << YAML::EndSeq;

  out << YAML::Key << "groups" << YAML::Value << YAML::BeginSeq;
  for (size_t id = 0; id < groupDescriptions.size(); ++id) {
    std::string groupName = std::string("group") + std::to_string(id);
    out << YAML::BeginMap << YAML::Key << groupName << YAML::Value << YAML::BeginSeq;
    const auto& descriptions = groupDescriptions[id];
    for (const auto& description : descriptions) {
      out << YAML::BeginMap;
      out << YAML::Key << "name" << YAML::Value << description.name;
      out << YAML::Key << "location" << YAML::Value << description.location;
      out << YAML::Key << "gridDim" << YAML::Value << YAML::Flow << YAML::BeginSeq
          << description.gridDim.x << description.gridDim.y << description.gridDim.z
          << YAML::EndSeq;

      out << YAML::Key << "argSizes" << YAML::Value << YAML::Flow << YAML::BeginSeq;
      for (const auto argSize : description.argsSizes) {
        out << argSize;
      }
      out << YAML::EndSeq;

      out << YAML::Key << "argTypes" << YAML::Value << YAML::Flow << YAML::BeginSeq;
      for (const auto argType : description.argsTypes) {
        out << argType;
      }
      out << YAML::EndSeq << YAML::EndMap;
    }
  }
  out << YAML::EndSeq << YAML::EndMap;

  auto fileName = std::string("graph") + std::to_string(instanceId_) + std::string(".yaml");
  auto configPath = GraphFuseRecorder::generateFilePath(fileName);
  auto configFile = std::fstream(configPath.c_str(), std::ios_base::out);
  if (configFile) {
    configFile << out.c_str() << "\n";
    GraphFuseRecorder::savedFusionConfigs_.push_back(fileName);
  } else {
    std::stringstream msg;
    msg << "failed to write yaml config file to `" << configPath.c_str() << "`";
    LogPrintfError("%s", msg.str().c_str());
  }
}

std::string GraphFuseRecorder::generateFilePath(const std::string& name) {
  auto path = std::filesystem::path(GraphFuseRecorder::tmpDirName_) / std::filesystem::path(name);
  return std::filesystem::weakly_canonical(path).string();
}

std::string GraphFuseRecorder::generateImagePath(size_t imageId) {
  auto name = std::string("img") + std::to_string(imageId) + std::string(".bin");
  return GraphFuseRecorder::generateFilePath(name);
}
}  // namespace hip
