//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "platform/agent.hpp"
#include "platform/object.hpp"
#include "os/os.hpp"
#include "amdocl/cl_common.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

namespace amd {


typedef cl_int(CL_CALLBACK* clAgent_OnLoad_fn)(cl_agent* agent);
typedef void(CL_CALLBACK* clAgent_OnUnload_fn)(cl_agent* agent);

Agent::Agent(const char* moduleName) : ready_(false) {
  ::memset(&callbacks_, '\0', sizeof(callbacks_));
  ::memset(&capabilities_, '\0', sizeof(capabilities_));

  library_ = Os::loadLibrary(moduleName);
  if (library_ == NULL) {
    return;
  }

  clAgent_OnLoad_fn onLoad =
      reinterpret_cast<clAgent_OnLoad_fn>(Os::getSymbol(library_, "clAgent_OnLoad"));
  if (onLoad == NULL) {
    return;
  }

  _cl_agent* agent = static_cast<_cl_agent*>(this);
  ::memcpy(agent, &entryPoints_, sizeof(entryPoints_));

  // Register in the agents linked-list.
  next_ = list_;
  list_ = this;

  if (onLoad(agent) != CL_SUCCESS) {
    list_ = list_->next_;
  }

  // Mark this instance as ready for use.
  ready_ = true;
}

Agent::~Agent() {
  if (library_ != NULL) {
    clAgent_OnUnload_fn onUnload =
        reinterpret_cast<clAgent_OnUnload_fn>(Os::getSymbol(library_, "clAgent_OnUnload"));

    if (onUnload != NULL) {
      onUnload(static_cast<cl_agent*>(this));
    }

    Os::unloadLibrary(library_);
  }
}

cl_int Agent::setCallbacks(const cl_agent_callbacks* callbacks, size_t size) {
  // FIXME_lmoriche: check size
  memcpy(&callbacks_, callbacks, size);
  return CL_SUCCESS;
}

cl_int Agent::getCapabilities(cl_agent_capabilities* caps) {
  if (caps == NULL) {
    return CL_INVALID_VALUE;
  }
  *caps = capabilities_;
  return CL_SUCCESS;
}

static inline cl_agent_capabilities operator~(const cl_agent_capabilities& src) {
  cl_agent_capabilities result;

  const char* a = reinterpret_cast<const char*>(&src);
  char* b = reinterpret_cast<char*>(&result);
  for (size_t i = 0; i < sizeof(cl_agent_capabilities); ++i) {
    *b++ = ~*a++;
  }

  return result;
}

static inline cl_agent_capabilities operator|(const cl_agent_capabilities& lhs,
                                              const cl_agent_capabilities& rhs) {
  cl_agent_capabilities result;

  const char* a = reinterpret_cast<const char*>(&lhs);
  const char* b = reinterpret_cast<const char*>(&rhs);
  char* c = reinterpret_cast<char*>(&result);
  for (size_t i = 0; i < sizeof(cl_agent_capabilities); ++i) {
    *c++ = *a++ | *b++;
  }

  return result;
}

static inline cl_agent_capabilities operator&(const cl_agent_capabilities& lhs,
                                              const cl_agent_capabilities& rhs) {
  cl_agent_capabilities result;

  const char* a = reinterpret_cast<const char*>(&lhs);
  const char* b = reinterpret_cast<const char*>(&rhs);
  char* c = reinterpret_cast<char*>(&result);
  for (size_t i = 0; i < sizeof(cl_agent_capabilities); ++i) {
    *c++ = *a++ & *b++;
  }

  return result;
}

static inline bool operator==(const cl_agent_capabilities& lhs, const cl_agent_capabilities& rhs) {
  const char* a = reinterpret_cast<const char*>(&lhs);
  const char* b = reinterpret_cast<const char*>(&rhs);
  for (size_t i = 0; i < sizeof(cl_agent_capabilities); ++i) {
    if (*a++ != *b++) {
      return false;
    }
  }

  return true;
}

static inline bool operator!=(const cl_agent_capabilities& lhs, const cl_agent_capabilities& rhs) {
  return !(lhs == rhs);
}

cl_int Agent::setCapabilities(const cl_agent_capabilities* caps, bool install) {
  ScopedLock sl(capabilitiesLock_);

  if (caps == NULL || *caps != (*caps & potentialCapabilities_)) {
    return CL_INVALID_VALUE;
  }

  if (install) {
    capabilities_ = capabilities_ | *caps;
  } else {
    capabilities_ = capabilities_ & ~*caps;
  }

  memset(&enabledCapabilities_, '\0', sizeof(enabledCapabilities_));
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    enabledCapabilities_ = enabledCapabilities_ | agent->capabilities_;
  }

  return CL_SUCCESS;
}

bool Agent::init() {
  ::memset(&potentialCapabilities_, '\0', sizeof(potentialCapabilities_));
  potentialCapabilities_.canGenerateContextEvents = 1;
  potentialCapabilities_.canGenerateCommandQueueEvents = 1;
  potentialCapabilities_.canGenerateEventEvents = 1;
  //    potentialCapabilities_.canGenerateMemObjectEvents    = 1;
  //    potentialCapabilities_.canGenerateSamplerEvents      = 1;
  //    potentialCapabilities_.canGenerateProgramEvents      = 1;
  //    potentialCapabilities_.canGenerateKernelEvents       = 1;

  const char* envVar = ::getenv("CL_AGENT");
  if (envVar == NULL) {
    return true;
  }

  std::string token, modules = envVar;
  std::istringstream iss(modules);

  while (getline(iss, token, ',')) {
    Agent* agent = new Agent(token.c_str());
    if (agent == NULL || !agent->isReady()) {
      delete agent;

      // Only return an error if we failed the Agent allocation. Other
      // error (the agent is not ready) can be ignored.
      return agent != NULL;
    }
  }

  return true;
}

void Agent::tearDown() {
  while (list_ != NULL) {
    Agent* agent = list_;
    list_ = list_->next_;
    delete agent;
  }
}

namespace agent {

static cl_int CL_API_CALL GetVersionNumber(cl_agent* agent, cl_int* version_ret) {
  if (version_ret == NULL) {
    return CL_INVALID_VALUE;
  }
  *version_ret = CL_AGENT_VERSION_1_0;
  return CL_SUCCESS;
}

static cl_int CL_API_CALL GetPlatform(cl_agent* agent, cl_platform_id* platform_id_ret) {
  if (platform_id_ret == NULL) {
    return CL_INVALID_VALUE;
  }
  *platform_id_ret = AMD_PLATFORM;
  return CL_SUCCESS;
}

static cl_int CL_API_CALL GetTime(cl_agent* agent, cl_long* time_nanos) {
  if (time_nanos == NULL) {
    return CL_INVALID_VALUE;
  }
  *time_nanos = Os::timeNanos() + Os::offsetToEpochNanos();
  return CL_SUCCESS;
}

static cl_int CL_API_CALL SetCallbacks(cl_agent* agent, const cl_agent_callbacks* callbacks,
                                       size_t size) {
  return Agent::get(agent)->setCallbacks(callbacks, size);
}

static cl_int CL_API_CALL GetPotentialCapabilities(cl_agent* agent,
                                                   cl_agent_capabilities* capabilities) {
  if (capabilities == NULL) {
    return CL_INVALID_VALUE;
  }

  *capabilities = Agent::potentialCapabilities();
  return CL_SUCCESS;
}

static cl_int CL_API_CALL GetCapabilities(cl_agent* agent, cl_agent_capabilities* capabilities) {
  return Agent::get(agent)->getCapabilities(capabilities);
}

static cl_int CL_API_CALL SetCapabilities(cl_agent* agent,
                                          const cl_agent_capabilities* capabilities,
                                          cl_agent_capability_action action) {
  return Agent::get(agent)->setCapabilities(capabilities, action == CL_AGENT_ADD_CAPABILITIES);
}

static cl_int CL_API_CALL GetICDDispatchTable(cl_agent* agent, cl_icd_dispatch_table* table,
                                              size_t size) {
  // FIXME_lmoriche: check size
  memcpy(table, amd::ICDDispatchedObject::icdVendorDispatch_, size);
  return CL_SUCCESS;
}

static cl_int CL_API_CALL SetICDDispatchTable(cl_agent* agent, const cl_icd_dispatch_table* table,
                                              size_t size) {
  // FIXME_lmoriche: check size
  memcpy(amd::ICDDispatchedObject::icdVendorDispatch_, table, size);
  return CL_SUCCESS;
}

}  // namespace agent

cl_agent Agent::entryPoints_ = {agent::GetVersionNumber,
                                agent::GetPlatform,
                                agent::GetTime,
                                agent::SetCallbacks,
                                agent::GetPotentialCapabilities,
                                agent::GetCapabilities,
                                agent::SetCapabilities,
                                agent::GetICDDispatchTable,
                                agent::SetICDDispatchTable};

void Agent::postContextCreate(cl_context context) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acContextCreate_fn callback = agent->callbacks_.ContextCreate;
    if (callback != NULL && agent->canGenerateContextEvents()) {
      callback(agent, context);
    }
  }
}

void Agent::postContextFree(cl_context context) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acContextFree_fn callback = agent->callbacks_.ContextFree;
    if (callback != NULL && agent->canGenerateContextEvents()) {
      callback(agent, context);
    }
  }
}

void Agent::postCommandQueueCreate(cl_command_queue queue) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acCommandQueueCreate_fn callback = agent->callbacks_.CommandQueueCreate;
    if (callback != NULL && agent->canGenerateCommandQueueEvents()) {
      callback(agent, queue);
    }
  }
}

void Agent::postCommandQueueFree(cl_command_queue queue) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acCommandQueueFree_fn callback = agent->callbacks_.CommandQueueFree;
    if (callback != NULL && agent->canGenerateCommandQueueEvents()) {
      callback(agent, queue);
    }
  }
}

void Agent::postEventCreate(cl_event event, cl_command_type type) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acEventCreate_fn callback = agent->callbacks_.EventCreate;
    if (callback != NULL && agent->canGenerateEventEvents()) {
      callback(agent, event, type);
    }
  }
}

void Agent::postEventFree(cl_event event) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acEventFree_fn callback = agent->callbacks_.EventFree;
    if (callback != NULL && agent->canGenerateEventEvents()) {
      callback(agent, event);
    }
  }
}

void Agent::postEventStatusChanged(cl_event event, cl_int status, cl_long ts) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acEventStatusChanged_fn callback = agent->callbacks_.EventStatusChanged;
    if (callback != NULL && agent->canGenerateEventEvents()) {
      callback(agent, event, status, ts);
    }
  }
}

void Agent::postMemObjectCreate(cl_mem memobj) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acMemObjectCreate_fn callback = agent->callbacks_.MemObjectCreate;
    if (callback != NULL && agent->canGenerateMemObjectEvents()) {
      callback(agent, memobj);
    }
  }
}

void Agent::postMemObjectFree(cl_mem memobj) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acMemObjectFree_fn callback = agent->callbacks_.MemObjectFree;
    if (callback != NULL && agent->canGenerateMemObjectEvents()) {
      callback(agent, memobj);
    }
  }
}

void Agent::postMemObjectAcquired(cl_mem memobj, cl_device_id device, cl_long elapsed) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acMemObjectAcquired_fn callback = agent->callbacks_.MemObjectAcquired;
    if (callback != NULL && agent->canGenerateMemObjectEvents()) {
      callback(agent, memobj, device, elapsed);
    }
  }
}

void Agent::postSamplerCreate(cl_sampler sampler) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acSamplerCreate_fn callback = agent->callbacks_.SamplerCreate;
    if (callback != NULL && agent->canGenerateSamplerEvents()) {
      callback(agent, sampler);
    }
  }
}

void Agent::postSamplerFree(cl_sampler sampler) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acSamplerFree_fn callback = agent->callbacks_.SamplerFree;
    if (callback != NULL && agent->canGenerateSamplerEvents()) {
      callback(agent, sampler);
    }
  }
}

void Agent::postProgramCreate(cl_program program) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acProgramCreate_fn callback = agent->callbacks_.ProgramCreate;
    if (callback != NULL && agent->canGenerateProgramEvents()) {
      callback(agent, program);
    }
  }
}

void Agent::postProgramFree(cl_program program) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acProgramFree_fn callback = agent->callbacks_.ProgramFree;
    if (callback != NULL && agent->canGenerateProgramEvents()) {
      callback(agent, program);
    }
  }
}

void Agent::postProgramBuild(cl_program program) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acProgramBuild_fn callback = agent->callbacks_.ProgramBuild;
    if (callback != NULL && agent->canGenerateProgramEvents()) {
      callback(agent, program);
    }
  }
}

void Agent::postKernelCreate(cl_kernel kernel) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acKernelCreate_fn callback = agent->callbacks_.KernelCreate;
    if (callback != NULL && agent->canGenerateKernelEvents()) {
      callback(agent, kernel);
    }
  }
}

void Agent::postKernelFree(cl_kernel kernel) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acKernelFree_fn callback = agent->callbacks_.KernelFree;
    if (callback != NULL && agent->canGenerateKernelEvents()) {
      callback(agent, kernel);
    }
  }
}

void Agent::postKernelSetArg(cl_kernel kernel, cl_int index, size_t size, const void* value_ptr) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acKernelSetArg_fn callback = agent->callbacks_.KernelSetArg;
    if (callback != NULL && agent->canGenerateKernelEvents()) {
      callback(agent, kernel, index, size, value_ptr);
    }
  }
}

Agent* Agent::list_ = NULL;
Monitor Agent::capabilitiesLock_;
cl_agent_capabilities Agent::enabledCapabilities_ = {0};
cl_agent_capabilities Agent::potentialCapabilities_ = {0};

}  // namespace amd
