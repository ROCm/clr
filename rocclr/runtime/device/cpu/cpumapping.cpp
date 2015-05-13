//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cpukernel.hpp"
#include "platform/program.hpp"
#include "os/os.hpp"
#include "device/cpu/cpumapping.hpp"
#include <algorithm>
#include <functional>
#include <string>
#include <iostream>
#if defined(_WIN32)
#include <windows.h>
#endif
// amdrt.o
#if defined(WITH_ONLINE_COMPILER) && !defined(_LP64) && !defined(ATI_ARCH_ARM)
#include "amdrt.inc"
#endif
#include "acl.h"
using std::min;
using std::max;

namespace cpu {
    HCtoDCmap::HCtoDCmap(const clk_parameter_descriptor_t* desc, unsigned int level_alignment, unsigned int index, unsigned int init_offset)
    {
        //Initialize fields
        hc_offset = 0;
        hc_size = 0;
        dc_offset = 0;
        dc_size = 0;
        map_alignment = level_alignment;
        internal_field_map = NULL;
        next_field_map = NULL;
        return;
    }

    HCtoDCmap::~HCtoDCmap()
    {
        return;
    }

    //Helper to find sizes of each scalar type
    size_t HCtoDCmap::getHostScalarParamSize(const clk_value_type_t type) const
    {
        size_t size = 0;
        switch (type) {
        case T_CHAR:
            size = 1;
            break;
        case T_SHORT:  case T_CHAR2:
            size = 2;
            break;
        case T_FLOAT:  case T_INT:   case T_CHAR4:
        case T_SHORT2: case T_CHAR3:
            size = 4;
            break;
        case T_SAMPLER:
            size  = 4;
            break;
        case T_LONG:   case T_DOUBLE: case T_CHAR8:
        case T_SHORT4: case T_INT2:   case T_FLOAT2:
        case T_SHORT3:
            size = 8;
            break;
        case T_INT3:   case T_FLOAT3:
        case T_CHAR16: case T_SHORT8: case T_INT4:
        case T_FLOAT4: case T_LONG2:  case T_DOUBLE2:
            size = 16;
            break;
        case T_LONG3:  case T_DOUBLE3:
        case T_SHORT16: case T_INT8:  case T_FLOAT8:
        case T_LONG4:   case T_DOUBLE4:
            size = 32;
            break;
        case T_INT16: case T_FLOAT16: case T_LONG8:
        case T_DOUBLE8:
            size = 64;
            break;
        case T_LONG16: case T_DOUBLE16:
            size = 128;
            break;
        case T_POINTER: case T_VOID:
            size = sizeof(void*);
            break;
        default:
            assert(0 && "unknown scalar parameter size");
            break;
        }
        return size;
    }

    size_t HCtoDCmap::getHostScalarAlignment(const clk_value_type_t type) const
    {
        size_t align = 0;
        switch (type) {
        case T_CHAR:
            align = 1;
            break;
        case T_SHORT:  case T_CHAR2:
            align = 2;
            break;
        case T_FLOAT:  case T_INT:   case T_CHAR4:
        case T_SHORT2: case T_CHAR3:
            align = 4;
            break;
        case T_SAMPLER:
            align  = sizeof(uint32_t);
            break;
        case T_LONG:
            align  = LP64_SWITCH(4, 8);
            break;
        case T_DOUBLE:
            align = LP64_SWITCH(4, 8);
            break;
        case T_CHAR8:
        case T_SHORT4: case T_INT2:   case T_FLOAT2:
        case T_SHORT3:
            align = 4;
            break;
        case T_INT3:   case T_FLOAT3:
        case T_CHAR16: case T_SHORT8: case T_INT4:
        case T_FLOAT4: case T_LONG2:  case T_DOUBLE2:
        case T_LONG3:  case T_DOUBLE3:
        case T_SHORT16: case T_INT8:  case T_FLOAT8:
        case T_LONG4:   case T_DOUBLE4:
        case T_INT16: case T_FLOAT16: case T_LONG8:
        case T_DOUBLE8:
        case T_LONG16: case T_DOUBLE16:
            align = LP64_SWITCH(4, 8);
            break;
        case T_POINTER: case T_VOID:
            align = sizeof(void*);
            break;
        default:
            assert(0 && "unknown scalar parameter alignment");
            break;
        }
        return align;
    }

    // Align up arguments within each map, return the size of current map parameter
    // Input current alignment of the parameter, size of outer struct if it exists
    void HCtoDCmap::align_map(unsigned alignment, unsigned &outer_hc_size, unsigned &outer_dc_size, int &inStruct)
    {
        unsigned map_param_size = 0;
        if (internal_field_map != NULL) {
            hc_size = 0; //Recalculate size to account for internal offsets
            inStruct++;
            internal_field_map->align_map(map_alignment, hc_size, dc_size, inStruct); // align internal struct, might alter size of this struct
        }
        // Use map_param_size to store current parameter size after adjusting alignment
        if (alignment != 1 && hc_size % alignment != 0) {
            map_param_size = max(alignment, hc_size - (hc_size%alignment) + alignment);
        }
        else {
            map_param_size = max(alignment, hc_size);
        }
        if (next_field_map != NULL) {
            next_field_map->hc_offset = this->next_offset(hc_offset, map_param_size, inStruct);
            next_field_map->align_map(alignment, outer_hc_size, outer_dc_size, inStruct);
            // Reset parameter size for char padding
            if (next_field_map->type == T_CHAR)
                map_param_size = 1;
        }
        else
        {
            // Moving out of struct
            if (inStruct > 0)
                inStruct--;
            if (type == T_CHAR)
                map_param_size = 1;
        }
        outer_hc_size = max(outer_hc_size, hc_offset+map_param_size);
        outer_dc_size = max(outer_dc_size, dc_offset+dc_size);
        return;
    }

    // Return current size of map, calculate internal maps and process next args if in struct.
    // Alignment: alignment flag for members in case of structs, alignment of scalar otherwise.
    int HCtoDCmap::compute_map(const clk_parameter_descriptor_t* desc, unsigned int &alignment, unsigned int init_offset, int& inStruct, int& index_out)
    {
        unsigned internal_index;
        internal_index = index_out;
        unsigned int next_offset = init_offset;
        unsigned struct_size = 0;
        type = desc[internal_index].type;

        if (desc[internal_index].type == T_STRUCT) {
            //Moving into struct, go to next index
            inStruct++;
            hc_offset = init_offset;
            if (desc[index_out+1].type != T_VOID) {
                index_out++;
                internal_index = index_out;
                internal_field_map = new HCtoDCmap(desc, 0, internal_index, init_offset);
                hc_size = internal_field_map->compute_map(desc, map_alignment, next_offset, inStruct, index_out);
                map_alignment = max(map_alignment, internal_field_map->map_alignment);    // Adjust alignment to biggest member alignment
                struct_size = hc_size;
                internal_index = index_out;
                alignment = max(alignment, map_alignment);
                if (inStruct > 0) {
                    if (desc[index_out+1].type != T_VOID) {
                        //Still inside struct and not done
                        index_out++;
                        internal_index = index_out;
                        next_field_map = new HCtoDCmap(desc, 0, internal_index, next_offset);
                        struct_size = hc_size;
                        struct_size += next_field_map->compute_map(desc, alignment, next_offset, inStruct, index_out);
                        next_offset = max(next_field_map->hc_offset+next_field_map->hc_size, next_field_map->hc_offset+alignment);
                        // running count of strucdc_size = hc_size + size of next member
                        return struct_size;
                    }
                    else {
                        //Moving out of struct, go to next index
                        index_out++;
                        internal_index = index_out;
                        inStruct--;
                        return hc_size; //return last struct member size
                    }
                }
            }
        }
        else {
            //Scalar parameter
            hc_offset = init_offset;
            hc_size = getHostScalarParamSize(desc[internal_index].type);
            dc_size = hc_size;
            map_alignment = getHostScalarAlignment(desc[internal_index].type);
            alignment = max(alignment, map_alignment); //Adjust alignment of upper level struct if necessary, upper level alignment = max alignment of members
            if (desc[internal_index].type == T_LONG)
                alignment = max(alignment, (unsigned int)8);    //Set struct alignment to 8 on outside if containing struct member of long
            if (inStruct > 0) {
                if (desc[index_out+1].type != T_VOID) {
                    //Still inside struct and not done
                    index_out++;
                    next_field_map = new HCtoDCmap(desc, alignment, internal_index, next_offset);
                    struct_size = hc_size;
                    struct_size += next_field_map->compute_map(desc, alignment, next_offset, inStruct, index_out);
                    next_offset = hc_offset+alignment;
                    alignment = max(alignment, next_field_map->map_alignment);
                    // running count of strucdc_size = hc_size + size of next member
                    return struct_size;
                }
                else {
                    //Moving out of struct, go to next index
                    index_out++;
                    inStruct--;
                    return hc_size; //return last struct member size
                }
            }
        }
        return hc_size;
    }

    // Adjust offset for source and target, return next source offset
    unsigned HCtoDCmap::next_offset(unsigned current_offset, unsigned &map_param_size, int& inStruct_flag)
    {
        unsigned next_offset = current_offset;
        if (next_field_map == NULL) {
            assert(0 && "invalid next struct field map");
            return next_offset;
        }
        else {
            // Ignore alignment when a char occurs to account for padding
            if (type != T_STRUCT && next_field_map->hc_size == 1 && map_param_size > 1 && inStruct_flag > 0) {
                next_field_map->dc_offset = dc_offset + dc_size;
                next_offset = current_offset + hc_size;
            }
            //
            else {
                if (this->next_field_map->type == T_LONG) {
                    if (dc_size % 4 != 0) {
                        this->next_field_map->dc_offset = dc_offset + dc_size - (dc_size % 4) + 4;    // T_LONG aligned by 4 in target
                    }
                    else {
                        this->next_field_map->dc_offset = dc_offset + dc_size;    // T_LONG aligned by 4 in target
                    }
                    if (dc_size % 8 != 0) {
                        next_offset = current_offset + dc_size - (dc_size % 8) + 8;   //aligned by 8 in source
                    }
                    else {
                        next_offset = current_offset + dc_size;   //aligned by 8 in source
                    }
                }
                else {
                    if ((dc_offset + dc_size) % next_field_map->map_alignment != 0) {
                        this->next_field_map->dc_offset = dc_offset + dc_size - (dc_size % next_field_map->map_alignment) + next_field_map->map_alignment;
                    }
                    else {
                        this->next_field_map->dc_offset = dc_offset + max(dc_size, next_field_map->map_alignment);
                    }
                    if ((hc_offset + hc_size) % next_field_map->map_alignment != 0) {
                        next_offset = hc_offset + hc_size - (hc_size % next_field_map->map_alignment) + next_field_map->map_alignment;
                    }
                    else {
                        next_offset = hc_offset + max(next_field_map->map_alignment, map_param_size);
                    }
                }
            }
            return next_offset;
        }
    }

    // Copy memory according to mapping
    unsigned int HCtoDCmap::copy_params(void *dst, const void *src, unsigned int &arg_offset, int& error_code, int &inStruct) const
    {
        unsigned int padding = 0;
        // Pad offset to be aligned by 8 if parameter is double, not as struct field
        if ((arg_offset+dc_offset) % 8 != 0 && (type == T_DOUBLE) && inStruct == 0)
            padding = map_alignment-((arg_offset+dc_offset)%map_alignment);
        ::memcpy(reinterpret_cast<void *>(reinterpret_cast<unsigned char*>(dst)+padding), src, hc_size);
        if (internal_field_map != NULL) {
            inStruct++;
            internal_field_map->copy_params(dst, src, arg_offset, error_code, inStruct);
            inStruct--;
        }
        if (next_field_map != NULL) {
            void *next_dst = reinterpret_cast<void *>(reinterpret_cast<unsigned char*>(dst)+next_field_map->dc_offset);
            const void *next_src = reinterpret_cast<const void *>(reinterpret_cast<const unsigned char*>(src)+next_field_map->hc_offset);
            next_field_map->copy_params(next_dst, next_src, arg_offset, error_code, inStruct);
        }
        return padding;
    }
} //namespace cpu