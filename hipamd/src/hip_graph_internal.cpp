/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hip_graph_internal.hpp"
#include <queue>

#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    case_string = #C;                                                                              \
    break;
const char* GetGraphNodeTypeString(uint32_t op) {
  const char* case_string;
  switch (static_cast<hipGraphNodeType>(op)) {
    CASE_STRING(hipGraphNodeTypeKernel, KernelNode)
    CASE_STRING(hipGraphNodeTypeMemcpy, MemcpyNode)
    CASE_STRING(hipGraphNodeTypeMemset, MemsetNode)
    CASE_STRING(hipGraphNodeTypeHost, HostNode)
    CASE_STRING(hipGraphNodeTypeGraph, GraphNode)
    CASE_STRING(hipGraphNodeTypeEmpty, EmptyNode)
    CASE_STRING(hipGraphNodeTypeWaitEvent, WaitEventNode)
    CASE_STRING(hipGraphNodeTypeEventRecord, EventRecordNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreSignal, ExtSemaphoreSignalNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreWait, ExtSemaphoreWaitNode)
    CASE_STRING(hipGraphNodeTypeMemcpyFromSymbol, MemcpyFromSymbolNode)
    CASE_STRING(hipGraphNodeTypeMemcpyToSymbol, MemcpyToSymbolNode)
    default:
      case_string = "Unknown node type";
  };
  return case_string;
};

int hipGraphNode::nextID = 0;
int ihipGraph::nextID = 0;
std::unordered_set<hipGraphNode*> hipGraphNode::nodeSet_;
amd::Monitor hipGraphNode::nodeSetLock_{"Guards global node set"};
std::unordered_set<ihipGraph*> ihipGraph::graphSet_;
amd::Monitor ihipGraph::graphSetLock_{"Guards global graph set"};
std::unordered_set<hipGraphExec*> hipGraphExec::graphExecSet_;
amd::Monitor hipGraphExec::graphExecSetLock_{"Guards global exec graph set"};
std::unordered_set<hipUserObject*> hipUserObject::ObjectSet_;
amd::Monitor hipUserObject::UserObjectLock_{"Guards global user object"};

hipError_t hipGraphMemcpyNode1D::ValidateParams(void* dst, const void* src, size_t count,
                                                hipMemcpyKind kind) {
  hipError_t status = ihipMemcpy_validate(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  size_t sOffsetOrig = 0;
  amd::Memory* origSrcMemory = getMemoryObject(src, sOffsetOrig);
  size_t dOffsetOrig = 0;
  amd::Memory* origDstMemory = getMemoryObject(dst, dOffsetOrig);

  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
    if (origDstMemory->getContext().devices()[0] != dstMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if ((kind != hipMemcpyHostToDevice) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
    if (origSrcMemory->getContext().devices()[0] != srcMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if ((kind != hipMemcpyDeviceToHost) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
    if (origDstMemory->getContext().devices()[0] != dstMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if (origSrcMemory->getContext().devices()[0] != srcMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

hipError_t hipGraphMemcpyNode1D::SetCommandParams(void* dst, const void* src, size_t count,
                                                  hipMemcpyKind kind) {
  hipError_t status = ihipMemcpy_validate(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  size_t sOffsetOrig = 0;
  amd::Memory* origSrcMemory = getMemoryObject(src, sOffsetOrig);
  size_t dOffsetOrig = 0;
  amd::Memory* origDstMemory = getMemoryObject(dst, dOffsetOrig);

  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {
    if (origDstMemory->getContext().devices()[0] != dstMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    amd::WriteMemoryCommand* command = reinterpret_cast<amd::WriteMemoryCommand*>(commands_[0]);
    command->setParams(*dstMemory->asBuffer(), dOffset, count, src);
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {
    if (origSrcMemory->getContext().devices()[0] != srcMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    amd::ReadMemoryCommand* command = reinterpret_cast<amd::ReadMemoryCommand*>(commands_[0]);
    command->setParams(*srcMemory->asBuffer(), sOffset, count, dst);
  } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
    if (origDstMemory->getContext().devices()[0] != dstMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if (origSrcMemory->getContext().devices()[0] != srcMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    amd::CopyMemoryP2PCommand* command = reinterpret_cast<amd::CopyMemoryP2PCommand*>(commands_[0]);
    command->setParams(*srcMemory->asBuffer(), *dstMemory->asBuffer(), sOffset, dOffset, count);
    // Make sure runtime has valid memory for the command execution. P2P access
    // requires page table mapping on the current device to another GPU memory
    if (!static_cast<amd::CopyMemoryP2PCommand*>(command)->validateMemory()) {
      delete command;
      return hipErrorInvalidValue;
    }
  } else {
    amd::CopyMemoryCommand* command = reinterpret_cast<amd::CopyMemoryCommand*>(commands_[0]);
    command->setParams(*srcMemory->asBuffer(), *dstMemory->asBuffer(), sOffset, dOffset, count);
  }
  return hipSuccess;
}

hipError_t hipGraphMemcpyNode::ValidateParams(const hipMemcpy3DParms* pNodeParams) {
  hipError_t status = ihipMemcpy3D_validate(pNodeParams);
  if (status != hipSuccess) {
    return status;
  }
  const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pNodeParams);
  // If {src/dst}MemoryType is hipMemoryTypeUnified, {src/dst}Device and {src/dst}Pitch specify the
  // (unified virtual address space) base address of the source data and the bytes per row to apply.
  // {src/dst}Array is ignored.
  hipMemoryType srcMemoryType = pCopy.srcMemoryType;
  if (srcMemoryType == hipMemoryTypeUnified) {
    srcMemoryType =
        amd::MemObjMap::FindMemObj(pCopy.srcDevice) ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (srcMemoryType == hipMemoryTypeHost) {
      // {src/dst}Host may be unitialized. Copy over {src/dst}Device into it if we detect system
      // memory.
      const_cast<HIP_MEMCPY3D*>(&pCopy)->srcHost = pCopy.srcDevice;
    }
  }
  hipMemoryType dstMemoryType = pCopy.dstMemoryType;
  if (dstMemoryType == hipMemoryTypeUnified) {
    dstMemoryType =
        amd::MemObjMap::FindMemObj(pCopy.dstDevice) ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (srcMemoryType == hipMemoryTypeHost) {
      const_cast<HIP_MEMCPY3D*>(&pCopy)->dstHost = pCopy.dstDevice;
    }
  }

  // If {src/dst}MemoryType is hipMemoryTypeHost, check if the memory was prepinned.
  // In that case upgrade the copy type to hipMemoryTypeDevice to avoid extra pinning.
  if (srcMemoryType == hipMemoryTypeHost) {
    amd::Memory* mem = amd::MemObjMap::FindMemObj(pCopy.srcHost);
    srcMemoryType = mem ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (srcMemoryType == hipMemoryTypeDevice) {
      const_cast<HIP_MEMCPY3D*>(&pCopy)->srcDevice = const_cast<void*>(pCopy.srcHost);
    }
  }
  if (dstMemoryType == hipMemoryTypeHost) {
    amd::Memory* mem = amd::MemObjMap::FindMemObj(pCopy.dstHost);
    dstMemoryType = mem ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (dstMemoryType == hipMemoryTypeDevice) {
      const_cast<HIP_MEMCPY3D*>(&pCopy)->dstDevice = const_cast<void*>(pCopy.dstDevice);
    }
  }

  amd::Coord3D srcOrigin = {pCopy.srcXInBytes, pCopy.srcY, pCopy.srcZ};
  amd::Coord3D dstOrigin = {pCopy.dstXInBytes, pCopy.dstY, pCopy.dstZ};
  amd::Coord3D copyRegion = {pCopy.WidthInBytes, pCopy.Height, pCopy.Depth};

  if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Host to Device.

    amd::Memory* dstMemory;
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    status =
        ihipMemcpyHtoDValidate(pCopy.srcHost, pCopy.dstDevice, srcOrigin, dstOrigin, copyRegion,
                               pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight, pCopy.dstPitch,
                               pCopy.dstPitch * pCopy.dstHeight, dstMemory, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeHost)) {
    // Device to Host.
    amd::Memory* srcMemory;
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;
    status =
        ihipMemcpyDtoHValidate(pCopy.srcDevice, pCopy.dstHost, srcOrigin, dstOrigin, copyRegion,
                               pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight, pCopy.dstPitch,
                               pCopy.dstPitch * pCopy.dstHeight, srcMemory, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Device to Device.
    amd::Memory* srcMemory;
    amd::Memory* dstMemory;
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    status = ihipMemcpyDtoDValidate(pCopy.srcDevice, pCopy.dstDevice, srcOrigin, dstOrigin,
                                    copyRegion, pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight,
                                    pCopy.dstPitch, pCopy.dstPitch * pCopy.dstHeight, srcMemory,
                                    dstMemory, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeArray)) {
    amd::Image* dstImage;
    amd::BufferRect srcRect;

    status =
        ihipMemcpyHtoAValidate(pCopy.srcHost, pCopy.dstArray, srcOrigin, dstOrigin, copyRegion,
                               pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight, dstImage, srcRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeHost)) {
    // Image to Host.
    amd::Image* srcImage;
    amd::BufferRect dstRect;

    status =
        ihipMemcpyAtoHValidate(pCopy.srcArray, pCopy.dstHost, srcOrigin, dstOrigin, copyRegion,
                               pCopy.dstPitch, pCopy.dstPitch * pCopy.dstHeight, srcImage, dstRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeArray)) {
    // Device to Image.
    amd::Image* dstImage;
    amd::Memory* srcMemory;
    amd::BufferRect dstRect;
    amd::BufferRect srcRect;
    status = ihipMemcpyDtoAValidate(pCopy.srcDevice, pCopy.dstArray, srcOrigin, dstOrigin,
                                    copyRegion, pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight,
                                    dstImage, srcMemory, dstRect, srcRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Image to Device.
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;
    amd::Memory* dstMemory;
    amd::Image* srcImage;
    status = ihipMemcpyAtoDValidate(pCopy.srcArray, pCopy.dstDevice, srcOrigin, dstOrigin,
                                    copyRegion, pCopy.dstPitch, pCopy.dstPitch * pCopy.dstHeight,
                                    dstMemory, srcImage, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeArray)) {
    amd::Image* srcImage;
    amd::Image* dstImage;

    status = ihipMemcpyAtoAValidate(pCopy.srcArray, pCopy.dstArray, srcOrigin, dstOrigin,
                                    copyRegion, srcImage, dstImage);
    if (status != hipSuccess) {
      return status;
    }
  } else {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t hipGraphMemcpyNode::SetCommandParams(const hipMemcpy3DParms* pNodeParams) {
  hipError_t status = ihipMemcpy3D_validate(pNodeParams);
  if (status != hipSuccess) {
    return status;
  }
  const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pNodeParams);
  // If {src/dst}MemoryType is hipMemoryTypeUnified, {src/dst}Device and {src/dst}Pitch specify the
  // (unified virtual address space) base address of the source data and the bytes per row to apply.
  // {src/dst}Array is ignored.
  hipMemoryType srcMemoryType = pCopy.srcMemoryType;
  if (srcMemoryType == hipMemoryTypeUnified) {
    srcMemoryType =
        amd::MemObjMap::FindMemObj(pCopy.srcDevice) ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (srcMemoryType == hipMemoryTypeHost) {
      // {src/dst}Host may be unitialized. Copy over {src/dst}Device into it if we detect system
      // memory.
      const_cast<HIP_MEMCPY3D*>(&pCopy)->srcHost = pCopy.srcDevice;
    }
  }
  hipMemoryType dstMemoryType = pCopy.dstMemoryType;
  if (dstMemoryType == hipMemoryTypeUnified) {
    dstMemoryType =
        amd::MemObjMap::FindMemObj(pCopy.dstDevice) ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (srcMemoryType == hipMemoryTypeHost) {
      const_cast<HIP_MEMCPY3D*>(&pCopy)->dstHost = pCopy.dstDevice;
    }
  }

  // If {src/dst}MemoryType is hipMemoryTypeHost, check if the memory was prepinned.
  // In that case upgrade the copy type to hipMemoryTypeDevice to avoid extra pinning.
  if (srcMemoryType == hipMemoryTypeHost) {
    amd::Memory* mem = amd::MemObjMap::FindMemObj(pCopy.srcHost);
    srcMemoryType = mem ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (srcMemoryType == hipMemoryTypeDevice) {
      const_cast<HIP_MEMCPY3D*>(&pCopy)->srcDevice = const_cast<void*>(pCopy.srcHost);
    }
  }
  if (dstMemoryType == hipMemoryTypeHost) {
    amd::Memory* mem = amd::MemObjMap::FindMemObj(pCopy.dstHost);
    dstMemoryType = mem ? hipMemoryTypeDevice : hipMemoryTypeHost;
    if (dstMemoryType == hipMemoryTypeDevice) {
      const_cast<HIP_MEMCPY3D*>(&pCopy)->dstDevice = const_cast<void*>(pCopy.dstDevice);
    }
  }

  amd::Coord3D srcOrigin = {pCopy.srcXInBytes, pCopy.srcY, pCopy.srcZ};
  amd::Coord3D dstOrigin = {pCopy.dstXInBytes, pCopy.dstY, pCopy.dstZ};
  amd::Coord3D copyRegion = {pCopy.WidthInBytes, pCopy.Height, pCopy.Depth};

  if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Host to Device.

    amd::Memory* dstMemory;
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    status =
        ihipMemcpyHtoDValidate(pCopy.srcHost, pCopy.dstDevice, srcOrigin, dstOrigin, copyRegion,
                               pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight, pCopy.dstPitch,
                               pCopy.dstPitch * pCopy.dstHeight, dstMemory, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::WriteMemoryCommand* command = reinterpret_cast<amd::WriteMemoryCommand*>(commands_[0]);
    command->setParams(*dstMemory, {dstRect.start_, 0, 0}, copyRegion, pCopy.srcHost, dstRect,
                       srcRect);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeHost)) {
    // Device to Host.
    amd::Memory* srcMemory;
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;
    status =
        ihipMemcpyDtoHValidate(pCopy.srcDevice, pCopy.dstHost, srcOrigin, dstOrigin, copyRegion,
                               pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight, pCopy.dstPitch,
                               pCopy.dstPitch * pCopy.dstHeight, srcMemory, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::ReadMemoryCommand* command = reinterpret_cast<amd::ReadMemoryCommand*>(commands_[0]);
    command->setParams(*srcMemory, {srcRect.start_, 0, 0}, copyRegion, pCopy.dstHost, srcRect,
                       dstRect);
    command->setSource(*srcMemory);
    command->setOrigin({srcRect.start_, 0, 0});
    command->setSize(copyRegion);
    command->setDestination(pCopy.dstHost);
    command->setBufRect(srcRect);
    command->setHostRect(dstRect);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Device to Device.
    amd::Memory* srcMemory;
    amd::Memory* dstMemory;
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    status = ihipMemcpyDtoDValidate(pCopy.srcDevice, pCopy.dstDevice, srcOrigin, dstOrigin,
                                    copyRegion, pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight,
                                    pCopy.dstPitch, pCopy.dstPitch * pCopy.dstHeight, srcMemory,
                                    dstMemory, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::CopyMemoryCommand* command = reinterpret_cast<amd::CopyMemoryCommand*>(commands_[0]);
    command->setParams(*srcMemory, *dstMemory, {srcRect.start_, 0, 0}, {dstRect.start_, 0, 0},
                       copyRegion, srcRect, dstRect);
  } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeArray)) {
    amd::Image* dstImage;
    amd::BufferRect srcRect;

    status =
        ihipMemcpyHtoAValidate(pCopy.srcHost, pCopy.dstArray, srcOrigin, dstOrigin, copyRegion,
                               pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight, dstImage, srcRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::WriteMemoryCommand* command = reinterpret_cast<amd::WriteMemoryCommand*>(commands_[0]);
    command->setParams(*dstImage, dstOrigin, copyRegion,
                       static_cast<const char*>(pCopy.srcHost) + srcRect.start_, pCopy.srcPitch,
                       pCopy.srcPitch * pCopy.srcHeight);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeHost)) {
    // Image to Host.
    amd::Image* srcImage;
    amd::BufferRect dstRect;

    status =
        ihipMemcpyAtoHValidate(pCopy.srcArray, pCopy.dstHost, srcOrigin, dstOrigin, copyRegion,
                               pCopy.dstPitch, pCopy.dstPitch * pCopy.dstHeight, srcImage, dstRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::ReadMemoryCommand* command = reinterpret_cast<amd::ReadMemoryCommand*>(commands_[0]);
    command->setParams(*srcImage, srcOrigin, copyRegion,
                       static_cast<char*>(pCopy.dstHost) + dstRect.start_, pCopy.dstPitch,
                       pCopy.dstPitch * pCopy.dstHeight);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeArray)) {
    // Device to Image.
    amd::Image* dstImage;
    amd::Memory* srcMemory;
    amd::BufferRect dstRect;
    amd::BufferRect srcRect;
    status = ihipMemcpyDtoAValidate(pCopy.srcDevice, pCopy.dstArray, srcOrigin, dstOrigin,
                                    copyRegion, pCopy.srcPitch, pCopy.srcPitch * pCopy.srcHeight,
                                    dstImage, srcMemory, dstRect, srcRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::CopyMemoryCommand* command = reinterpret_cast<amd::CopyMemoryCommand*>(commands_[0]);
    command->setParams(*srcMemory, *dstImage, srcOrigin, dstOrigin, copyRegion, srcRect, dstRect);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Image to Device.
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;
    amd::Memory* dstMemory;
    amd::Image* srcImage;
    status = ihipMemcpyAtoDValidate(pCopy.srcArray, pCopy.dstDevice, srcOrigin, dstOrigin,
                                    copyRegion, pCopy.dstPitch, pCopy.dstPitch * pCopy.dstHeight,
                                    dstMemory, srcImage, srcRect, dstRect);
    if (status != hipSuccess) {
      return status;
    }
    amd::CopyMemoryCommand* command = reinterpret_cast<amd::CopyMemoryCommand*>(commands_[0]);
    command->setParams(*srcImage, *dstMemory, srcOrigin, dstOrigin, copyRegion, srcRect, dstRect);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeArray)) {
    amd::Image* srcImage;
    amd::Image* dstImage;

    status = ihipMemcpyAtoAValidate(pCopy.srcArray, pCopy.dstArray, srcOrigin, dstOrigin,
                                    copyRegion, srcImage, dstImage);
    if (status != hipSuccess) {
      return status;
    }
    amd::CopyMemoryCommand* command = reinterpret_cast<amd::CopyMemoryCommand*>(commands_[0]);
    command->setParams(*srcImage, *dstImage, srcOrigin, dstOrigin, copyRegion);
  } else {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}


bool ihipGraph::isGraphValid(ihipGraph* pGraph) {
  amd::ScopedLock lock(graphSetLock_);
  if (graphSet_.find(pGraph) == graphSet_.end()) {
    return false;
  }
  return true;
}


void ihipGraph::AddNode(const Node& node) {
  vertices_.emplace_back(node);
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Add %s(%p)\n",
          GetGraphNodeTypeString(node->GetType()), node);
  node->SetParentGraph(this);
}

void ihipGraph::RemoveNode(const Node& node) {
  vertices_.erase(std::remove(vertices_.begin(), vertices_.end(), node), vertices_.end());
  delete node;
}

// root nodes are all vertices with 0 in-degrees
std::vector<Node> ihipGraph::GetRootNodes() const {
  std::vector<Node> roots;
  for (auto entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      roots.push_back(entry);
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] root node: %s(%p)\n",
              GetGraphNodeTypeString(entry->GetType()), entry);
    }
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "\n");
  return roots;
}

// leaf nodes are all vertices with 0 out-degrees
std::vector<Node> ihipGraph::GetLeafNodes() const {
  std::vector<Node> leafNodes;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      leafNodes.push_back(entry);
    }
  }
  return leafNodes;
}

size_t ihipGraph::GetLeafNodeCount() const {
  int numLeafNodes = 0;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      numLeafNodes++;
    }
  }
  return numLeafNodes;
}

std::vector<std::pair<Node, Node>> ihipGraph::GetEdges() const {
  std::vector<std::pair<Node, Node>> edges;
  for (const auto& i : vertices_) {
    for (const auto& j : i->GetEdges()) {
      edges.push_back(std::make_pair(i, j));
    }
  }
  return edges;
}

void ihipGraph::GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                               std::vector<Node>& singleList,
                               std::vector<std::vector<Node>>& parallelLists,
                               std::unordered_map<Node, std::vector<Node>>& dependencies) {
  // Mark the current node as visited.
  visited[v] = true;
  singleList.push_back(v);
  // Recurse for all the vertices adjacent to this vertex
  for (auto& adjNode : v->GetEdges()) {
    if (!visited[adjNode]) {
      // For the parallel list nodes add parent as the dependency
      if (singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] For %s(%p)- add parent as dependency %s(%p)\n",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
      GetRunListUtil(adjNode, visited, singleList, parallelLists, dependencies);
    } else {
      for (auto& list : parallelLists) {
        // Merge singleList when adjNode matches with the first element of the list in existing
        // lists
        if (adjNode == list[0]) {
          for (auto k = singleList.rbegin(); k != singleList.rend(); ++k) {
            list.insert(list.begin(), *k);
          }
          singleList.erase(singleList.begin(), singleList.end());
        }
      }
      // If the list cannot be merged with the existing list add as dependancy
      if (!singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] For %s(%p)- add dependency %s(%p)\n",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
    }
  }
  if (!singleList.empty()) {
    parallelLists.push_back(singleList);
    singleList.erase(singleList.begin(), singleList.end());
  }
}
// The function to do Topological Sort.
// It uses recursive GetRunListUtil()
void ihipGraph::GetRunList(std::vector<std::vector<Node>>& parallelLists,
                           std::unordered_map<Node, std::vector<Node>>& dependencies) {
  std::vector<Node> singleList;

  // Mark all the vertices as not visited
  std::unordered_map<Node, bool> visited;
  for (auto node : vertices_) visited[node] = false;

  // Call the recursive helper function for all vertices one by one
  for (auto node : vertices_) {
    // If the node has embedded child graph
    node->GetRunList(parallelLists, dependencies);
    if (visited[node] == false) {
      GetRunListUtil(node, visited, singleList, parallelLists, dependencies);
    }
  }
  for (size_t i = 0; i < parallelLists.size(); i++) {
    for (size_t j = 0; j < parallelLists[i].size(); j++) {
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] list %d - %s(%p)\n", i + 1,
              GetGraphNodeTypeString(parallelLists[i][j]->GetType()), parallelLists[i][j]);
    }
  }
}

void ihipGraph::LevelOrder(std::vector<Node>& levelOrder) {
  std::vector<Node> roots = GetRootNodes();
  std::unordered_map<Node, bool> visited;
  std::queue<Node> q;
  for (auto it = roots.begin(); it != roots.end(); it++) {
    q.push(*it);
    ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] %s(%p) level:%d \n",
            GetGraphNodeTypeString((*it)->GetType()), *it, (*it)->GetLevel());
  }
  while (!q.empty()) {
    Node node = q.front();
    q.pop();
    levelOrder.push_back(node);
    for (const auto& i : node->GetEdges()) {
      if (visited.find(i) == visited.end() && i->GetLevel() == (node->GetLevel() + 1)) {
        q.push(i);
        ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] %s(%p) level:%d \n",
                GetGraphNodeTypeString(i->GetType()), i, i->GetLevel());
        visited[i] = true;
      }
    }
  }
}

ihipGraph* ihipGraph::clone(std::unordered_map<Node, Node>& clonedNodes) const {
  ihipGraph* newGraph = new ihipGraph(device_, this);
  for (auto entry : vertices_) {
    hipGraphNode* node = entry->clone();
    node->SetParentGraph(newGraph);
    newGraph->vertices_.push_back(node);
    clonedNodes[entry] = node;
  }

  std::vector<Node> clonedEdges;
  std::vector<Node> clonedDependencies;
  for (auto node : vertices_) {
    const std::vector<Node>& edges = node->GetEdges();
    clonedEdges.clear();
    for (auto edge : edges) {
      clonedEdges.push_back(clonedNodes[edge]);
    }
    clonedNodes[node]->SetEdges(clonedEdges);
  }
  for (auto node : vertices_) {
    const std::vector<Node>& dependencies = node->GetDependencies();
    clonedDependencies.clear();
    for (auto dep : dependencies) {
      clonedDependencies.push_back(clonedNodes[dep]);
    }
    clonedNodes[node]->SetDependencies(clonedDependencies);
  }
  return newGraph;
}

ihipGraph* ihipGraph::clone() const {
  std::unordered_map<Node, Node> clonedNodes;
  return clone(clonedNodes);
}

bool hipGraphExec::isGraphExecValid(hipGraphExec* pGraphExec) {
  amd::ScopedLock lock(graphExecSetLock_);
  if (graphExecSet_.find(pGraphExec) == graphExecSet_.end()) {
    return false;
  }
  return true;
}

hipError_t hipGraphExec::CreateStreams(uint32_t num_streams) {
  parallel_streams_.reserve(num_streams);
  for (uint32_t i = 0; i < num_streams; ++i) {
    auto stream = new hip::Stream(hip::getCurrentDevice(),
                                  hip::Stream::Priority::Normal, hipStreamNonBlocking);
    if (stream == nullptr || !stream->Create()) {
      delete stream;
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create parallel stream!\n");
      return hipErrorOutOfMemory;
    }
    parallel_streams_.push_back(stream);
  }
  return hipSuccess;
}

hipError_t hipGraphExec::Init() {
  hipError_t status;
  size_t min_num_streams = 1;

  for (auto& node : levelOrder_) {
    min_num_streams += node->GetNumParallelStreams();
  }
  status = CreateStreams(parallelLists_.size() - 1 + min_num_streams);
  return status;
}

hipError_t FillCommands(std::vector<std::vector<Node>>& parallelLists,
                        std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
                        std::vector<Node>& levelOrder, std::vector<amd::Command*>& rootCommands,
                        amd::Command*& endCommand, amd::HostQueue* queue) {
  hipError_t status;
  for (auto& node : levelOrder) {
    // TODO: clone commands from next launch
    status = node->CreateCommand(node->GetQueue());
    if (status != hipSuccess) return status;
    amd::Command::EventWaitList waitList;
    for (auto depNode : nodeWaitLists[node]) {
      for (auto command : depNode->GetCommands()) {
        waitList.push_back(command);
      }
    }
    node->UpdateEventWaitLists(waitList);
  }
  // rootCommand ensures graph is started (all parallel branches) after all the previous work is
  // finished
  bool first = true;
  for (auto& singleList : parallelLists) {
    if (first) {
      first = false;
      continue;
    }
    // marker from the same queue as the list
    amd::Command* rootCommand = new amd::Marker(*singleList[0]->GetQueue(), false, {});
    amd::Command::EventWaitList waitList;
    waitList.push_back(rootCommand);
    if (!singleList.empty()) {
      auto commands = singleList[0]->GetCommands();
      if (!commands.empty()) {
        commands[0]->updateEventWaitList(waitList);
        rootCommands.push_back(rootCommand);
      }
    }
  }
  // endCommand ensures next enqueued ones start after graph is finished (all parallel branches)
  amd::Command::EventWaitList graphLastCmdWaitList;
  first = true;
  for (auto& singleList : parallelLists) {
    if (first) {
      first = false;
      continue;
    }
    if (!singleList.empty()) {
      auto commands = singleList.back()->GetCommands();
      if (!commands.empty()) {
        graphLastCmdWaitList.push_back(commands.back());
      }
    }
  }
  if (!graphLastCmdWaitList.empty()) {
    endCommand = new amd::Marker(*queue, false, graphLastCmdWaitList);
    if (endCommand == nullptr) {
      return hipErrorOutOfMemory;
    }
  }
  return hipSuccess;
}

void UpdateStream(std::vector<std::vector<Node>>& parallelLists, hip::Stream* stream,
                 hipGraphExec* ptr) {
  int i = 0;
  for (const auto& list : parallelLists) {
    // first parallel list will be launched on the same queue as parent
    if (i == 0) {
      for (auto& node : list) {
        node->SetStream(stream, ptr);
      }
    } else {  // New stream for parallel branches
      hip::Stream* stream = ptr->GetAvailableStreams();
      for (auto& node : list) {
        node->SetStream(stream, ptr);
      }
    }
    i++;
  }
}

hipError_t hipGraphExec::Run(hipStream_t stream) {
  hipError_t status;
  amd::HostQueue* queue = hip::getQueue(stream);
  if (queue == nullptr) {
    return hipErrorInvalidResourceHandle;
  }
  if (flags_ == hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (!levelOrder_.empty()) {
      levelOrder_[0]->GetParentGraph()->FreeAllMemory();
    }
  }
  auto hip_stream = (stream == nullptr) ? hip::getCurrentDevice()->GetNullStream()
                                        : reinterpret_cast<hip::Stream*>(stream);
  UpdateStream(parallelLists_, hip_stream, this);
  std::vector<amd::Command*> rootCommands;
  amd::Command* endCommand = nullptr;
  status =
      FillCommands(parallelLists_, nodeWaitLists_, levelOrder_, rootCommands, endCommand, queue);
  if (status != hipSuccess) {
    return status;
  }
  for (auto& cmd : rootCommands) {
    cmd->enqueue();
    cmd->release();
  }
  for (auto& node : levelOrder_) {
    node->EnqueueCommands(stream);
  }
  if (endCommand != nullptr) {
    endCommand->enqueue();
    endCommand->release();
  }
  ResetQueueIndex();
  return status;
}
