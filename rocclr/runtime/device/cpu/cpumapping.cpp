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
#include <algorithm>

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
        hc_alignment = level_alignment;
        dc_alignment = level_alignment;
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

    size_t HCtoDCmap::getScalarAlignment(const clk_value_type_t type, bool isHost) const
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
            #if defined(_WIN32)
            align = 8;
            #else
            align = isHost? 8 : LP64_SWITCH(4, 8);
            #endif
            break;
        case T_DOUBLE:
            #if defined(_WIN32)
            align = 8;
            #else
            align = LP64_SWITCH(4, 8);
            #endif
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
    void HCtoDCmap::align_map(unsigned outer_hc_alignment, unsigned outer_dc_alignment, unsigned &outer_hc_size, unsigned &outer_dc_size, int &inStruct)
    {
        unsigned map_param_size = 0;
        if (internal_field_map != NULL) {
            hc_size = 0; //Recalculate size to account for internal offsets
            inStruct++;
            internal_field_map->align_map(hc_alignment, dc_alignment, hc_size, dc_size, inStruct); // align internal struct, might alter size of this struct
            if (hc_alignment != 1 && hc_size%hc_alignment)
                hc_size = max(hc_size, hc_size - (hc_size%hc_alignment) + hc_alignment);
            if (dc_alignment != 1 && dc_size%dc_alignment)
                dc_size = max(dc_size, dc_size - (dc_size%dc_alignment) + dc_alignment);
        }
        // Use map_param_size to store current parameter size after adjusting alignment
        if (hc_alignment != 1 && hc_size % hc_alignment != 0) {
            map_param_size = max(hc_alignment, hc_size - (hc_size%hc_alignment) + hc_alignment);
        }
        else {
            map_param_size = max(hc_alignment, hc_size);
        }
        if (next_field_map != NULL) {
            next_field_map->hc_offset = this->next_offset(hc_offset, map_param_size, inStruct);
            next_field_map->align_map(outer_hc_alignment, outer_dc_alignment, outer_hc_size, outer_dc_size, inStruct);
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
    int HCtoDCmap::compute_map(const clk_parameter_descriptor_t* desc, unsigned int &outer_hc_alignment, unsigned int &outer_dc_alignment, unsigned int init_offset, int& inStruct, int& index_out)
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
                hc_size = internal_field_map->compute_map(desc, hc_alignment, dc_alignment, next_offset, inStruct, index_out);
                hc_alignment = max(hc_alignment, internal_field_map->hc_alignment);    // Adjust alignment to biggest member alignment
                struct_size = hc_size;
                internal_index = index_out;
                outer_hc_alignment = max(outer_hc_alignment, hc_alignment);
                if (inStruct > 0) {
                    if (desc[index_out+1].type != T_VOID) {
                        //Still inside struct and not done
                        index_out++;
                        internal_index = index_out;
                        next_field_map = new HCtoDCmap(desc, 0, internal_index, next_offset);
                        struct_size = hc_size;
                        struct_size += next_field_map->compute_map(desc, outer_hc_alignment, outer_dc_alignment, next_offset, inStruct, index_out);
                        next_offset = max(next_field_map->hc_offset+next_field_map->hc_size, next_field_map->hc_offset+hc_alignment);
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
        else if (desc[internal_index].type == T_PAD) {
            //Struct has padding
            hc_offset = init_offset;
            if (desc[index_out+1].type != T_VOID) {
                index_out++;
                internal_index = index_out;
                internal_field_map = new HCtoDCmap(desc, 0, internal_index, init_offset);
                hc_size = internal_field_map->compute_map(desc, hc_alignment, dc_alignment, next_offset, inStruct, index_out);
                // Adjust alignment to biggest member alignment
                hc_alignment = 1;
                dc_alignment = 1;
                unsigned pad_size = hc_size;
                internal_index = index_out;
                if (desc[index_out+1].type != T_VOID) {
                    //Still inside padding and not done
                    index_out++;
                    internal_index = index_out;
                    next_field_map = new HCtoDCmap(desc, 0, internal_index, next_offset);
                    pad_size = hc_size;
                    pad_size += next_field_map->compute_map(desc, outer_hc_alignment, outer_dc_alignment, next_offset, inStruct, index_out);
                    next_offset = max(next_field_map->hc_offset+next_field_map->hc_size, next_field_map->hc_offset+hc_alignment);
                    // running count of padding dc_size = hc_size + size of next member
                    return pad_size;
                }
                else {
                    //Moving out of struct, go to next index
                    index_out++;
                    internal_index = index_out;
                    return hc_size; //return last padding member size
                }
            }
        }
        else {
            //Scalar parameter
            hc_offset = init_offset;
            hc_size = getHostScalarParamSize(desc[internal_index].type);
            dc_size = hc_size;
            hc_alignment = getScalarAlignment(desc[internal_index].type, true);
            dc_alignment = getScalarAlignment(desc[internal_index].type, false);
            outer_hc_alignment = max(outer_hc_alignment, hc_alignment); //Adjust alignment of upper level struct if necessary, upper level alignment = max alignment of members
            outer_dc_alignment = max(outer_dc_alignment, dc_alignment); //Adjust alignment of upper level struct if necessary, upper level alignment = max alignment of members
            if (inStruct > 0) {
                if (desc[index_out+1].type != T_VOID) {
                    //Still inside struct and not done
                    index_out++;
                    next_field_map = new HCtoDCmap(desc, outer_hc_alignment, internal_index, next_offset);
                    struct_size = hc_size;
                    struct_size += next_field_map->compute_map(desc, outer_hc_alignment, outer_dc_alignment, next_offset, inStruct, index_out);
                    next_offset = hc_offset+hc_alignment;
                    outer_hc_alignment = max(outer_hc_alignment, next_field_map->hc_alignment);
                    outer_dc_alignment = max(outer_dc_alignment, next_field_map->dc_alignment);
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
            if (type == T_PAD) {
                next_field_map->dc_offset = dc_offset + dc_size;
                next_offset = current_offset + hc_size;
            }
            else {
                if ((dc_offset + dc_size) % next_field_map->dc_alignment != 0) {
                    this->next_field_map->dc_offset = dc_offset + dc_size - (dc_size % next_field_map->dc_alignment) + next_field_map->dc_alignment;
                }
                else {
                    this->next_field_map->dc_offset = dc_offset + max(dc_size, next_field_map->dc_alignment);
                }
                if ((hc_offset + hc_size) % next_field_map->hc_alignment != 0) {
                    next_offset = hc_offset + hc_size - (hc_size % next_field_map->hc_alignment) + next_field_map->hc_alignment;
                }
                else {
                    next_offset = hc_offset + max(next_field_map->hc_alignment, map_param_size);
                }
            }
            return next_offset;
        }
    }

    // Copy memory according to mapping
    unsigned int HCtoDCmap::copy_params(void *dst, const void *src, unsigned int arg_offset, int& error_code, int &inStruct) const
    {
        unsigned int padding = 0;
        // Pad offset to be aligned by 8 if parameter is double, not as struct field
        if ((arg_offset) % 8 != 0 && (type == T_DOUBLE) && inStruct == 0)
            padding = hc_alignment-((arg_offset+dc_offset)%hc_alignment);
        #if defined(_WIN32)
        // In windows, double is aligned by 8, add padding to struct if it contains double
        if ((arg_offset+dc_offset) % 8 != 0 && hc_alignment == 8)
            padding = hc_alignment-((arg_offset+dc_offset)%hc_alignment);
        #endif
        ::memcpy(reinterpret_cast<void *>(reinterpret_cast<unsigned char*>(dst)+padding), src, hc_size);
        #if defined(_WIN32)
        if (internal_field_map != NULL) {
            inStruct++;
            void *internal_dst = reinterpret_cast<void *>(reinterpret_cast<unsigned char*>(dst)+padding);
            internal_field_map->copy_params(internal_dst, src, arg_offset+padding, error_code, inStruct);
            inStruct--;
        }
        if (next_field_map != NULL) {
            void *next_dst = reinterpret_cast<void *>(reinterpret_cast<unsigned char*>(dst)+next_field_map->dc_offset); // Next field starts with padding
            const void *next_src = reinterpret_cast<const void *>(reinterpret_cast<const unsigned char*>(src)+next_field_map->hc_offset);
            next_field_map->copy_params(next_dst, next_src, arg_offset+next_field_map->dc_offset, error_code, inStruct);
        }
        #else
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
        #endif
        return padding;
    }

} //namespace cpu