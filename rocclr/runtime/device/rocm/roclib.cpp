#include "hsalib.h"
#include "utils/flags.hpp"

#include "hsa.h"
#include "hsa_ext_finalize.h"

namespace oclhsa {

    void* g_hsaModule = NULL;
    struct HsaLibApi g_hsaLibApi;

    //
    // g_complibModule is defined in LoadCompLib(). This macro must be used only in LoadCompLib() function.
    //
#define LOADSYMBOL(api) \
    g_hsaLibApi._##api  = (pfn_##api) amd::Os::getSymbol(g_hsaModule, #api); \
    if( g_hsaLibApi._##api == NULL ) { \
    LogError ("amd::Os::getSymbol() for exported func " #api " failed."); \
    amd::Os::unloadLibrary(g_hsaModule); \
    return false; \
    }


    bool LoadHsaLib()
    {
	
        g_hsaModule = amd::Os::loadLibrary(
			LINUX_SWITCH( "hsa_runtime_core" LP64_ONLY("64")".so" ,
				"hsa_runtime_core" LP64_ONLY("64")));
        if( g_hsaModule == NULL ) {
            return false;
        }
		LOADSYMBOL(hsa_init)
		LOADSYMBOL(hsa_shut_down)
		LOADSYMBOL(hsa_system_get_info)
		LOADSYMBOL(hsa_iterate_agents)
		LOADSYMBOL(hsa_agent_get_info)
		LOADSYMBOL(hsa_queue_create)
		LOADSYMBOL(hsa_queue_destroy)
		LOADSYMBOL(hsa_queue_inactivate)
		LOADSYMBOL(hsa_queue_load_read_index_acquire)
		LOADSYMBOL(hsa_queue_load_read_index_relaxed)
		LOADSYMBOL(hsa_queue_load_write_index_acquire)
		LOADSYMBOL(hsa_queue_load_write_index_relaxed)
		LOADSYMBOL(hsa_queue_store_write_index_relaxed)
		LOADSYMBOL(hsa_queue_store_write_index_release)
		LOADSYMBOL(hsa_queue_cas_write_index_acq_rel)
		LOADSYMBOL(hsa_queue_cas_write_index_acquire)
		LOADSYMBOL(hsa_queue_cas_write_index_relaxed)
		LOADSYMBOL(hsa_queue_cas_write_index_release)
		LOADSYMBOL(hsa_queue_add_write_index_acq_rel)
		LOADSYMBOL(hsa_queue_add_write_index_acquire)
		LOADSYMBOL(hsa_queue_add_write_index_relaxed)
		LOADSYMBOL(hsa_queue_add_write_index_release)
		LOADSYMBOL(hsa_queue_store_read_index_relaxed)
		LOADSYMBOL(hsa_queue_store_read_index_release)
		LOADSYMBOL(hsa_agent_iterate_regions)
		LOADSYMBOL(hsa_region_get_info)
		LOADSYMBOL(hsa_memory_register)
		LOADSYMBOL(hsa_memory_deregister)
		LOADSYMBOL(hsa_memory_allocate)
		LOADSYMBOL(hsa_memory_free)
		LOADSYMBOL(hsa_signal_create)
		LOADSYMBOL(hsa_signal_destroy)
		LOADSYMBOL(hsa_signal_load_relaxed)
		LOADSYMBOL(hsa_signal_load_acquire)
		LOADSYMBOL(hsa_signal_store_relaxed)
		LOADSYMBOL(hsa_signal_store_release)
		LOADSYMBOL(hsa_signal_wait_relaxed)
		LOADSYMBOL(hsa_signal_wait_acquire)
		LOADSYMBOL(hsa_signal_and_relaxed)
		LOADSYMBOL(hsa_signal_and_acquire)
		LOADSYMBOL(hsa_signal_and_release)
		LOADSYMBOL(hsa_signal_and_acq_rel)
		LOADSYMBOL(hsa_signal_or_relaxed)
		LOADSYMBOL(hsa_signal_or_acquire)
		LOADSYMBOL(hsa_signal_or_release)
		LOADSYMBOL(hsa_signal_or_acq_rel)
		LOADSYMBOL(hsa_signal_xor_relaxed)
		LOADSYMBOL(hsa_signal_xor_acquire)
		LOADSYMBOL(hsa_signal_xor_release)
		LOADSYMBOL(hsa_signal_xor_acq_rel)
		LOADSYMBOL(hsa_signal_exchange_relaxed)
		LOADSYMBOL(hsa_signal_exchange_acquire)
		LOADSYMBOL(hsa_signal_exchange_release)
		LOADSYMBOL(hsa_signal_exchange_acq_rel)
		LOADSYMBOL(hsa_signal_add_relaxed)
		LOADSYMBOL(hsa_signal_add_acquire)
		LOADSYMBOL(hsa_signal_add_release)
		LOADSYMBOL(hsa_signal_add_acq_rel)
		LOADSYMBOL(hsa_signal_subtract_relaxed)
		LOADSYMBOL(hsa_signal_subtract_acquire)
		LOADSYMBOL(hsa_signal_subtract_release)
		LOADSYMBOL(hsa_signal_subtract_acq_rel)
		LOADSYMBOL(hsa_signal_cas_relaxed)
		LOADSYMBOL(hsa_signal_cas_acquire)
		LOADSYMBOL(hsa_signal_cas_release)
		LOADSYMBOL(hsa_signal_cas_acq_rel)
		LOADSYMBOL(hsa_status_string)
		LOADSYMBOL(hsa_ext_program_create)
		LOADSYMBOL(hsa_ext_program_destroy)
		LOADSYMBOL(hsa_ext_add_module)
		LOADSYMBOL(hsa_ext_finalize_program)
		LOADSYMBOL(hsa_ext_query_program_agent_id)
		LOADSYMBOL(hsa_ext_query_program_agent_count)
		LOADSYMBOL(hsa_ext_query_program_agents)
		LOADSYMBOL(hsa_ext_query_program_module_count)
		LOADSYMBOL(hsa_ext_query_program_modules)
		LOADSYMBOL(hsa_ext_query_program_brig_module)
		LOADSYMBOL(hsa_ext_query_call_convention)
		LOADSYMBOL(hsa_ext_query_symbol_definition)
		LOADSYMBOL(hsa_ext_define_program_allocation_global_variable_address)
		LOADSYMBOL(hsa_ext_query_program_allocation_global_variable_address)
		LOADSYMBOL(hsa_ext_define_agent_allocation_global_variable_address)
		LOADSYMBOL(hsa_ext_query_agent_global_variable_address)
		LOADSYMBOL(hsa_ext_define_readonly_variable_address)
		LOADSYMBOL(hsa_ext_query_readonly_variable_address)
		LOADSYMBOL(hsa_ext_query_kernel_descriptor_address)
		LOADSYMBOL(hsa_ext_query_indirect_function_descriptor_address)
		LOADSYMBOL(hsa_ext_validate_program)
		LOADSYMBOL(hsa_ext_validate_program_module)
		LOADSYMBOL(hsa_ext_serialize_program)
		LOADSYMBOL(hsa_ext_deserialize_program)
		LOADSYMBOL(hsa_ext_get_memory_type)
		LOADSYMBOL(hsa_ext_set_memory_type)
		return true;
    }
}
