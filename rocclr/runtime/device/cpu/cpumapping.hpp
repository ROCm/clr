//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
// HCtoDCmap provides a mapping of parameters from host compiler to device compiler
// The mapping can be used to copy parameters from host to device where field alignment
// is different in compilers
#ifndef CPUMAPPING_HPP_
#define CPUMAPPING_HPP_

using std::min;
using std::max;

namespace cpu {

class HCtoDCmap
{

public:
    unsigned int hc_offset, hc_size;  // Offset and size of this parameter in host compiler
    unsigned int dc_offset, dc_size;  // Offset and size of this parameter in device compiler
    unsigned int hc_alignment;     // Alignment of parameter in host compiler
    unsigned int dc_alignment;     // Alignment of parameter in device compiler
    clk_value_type_t type;          // Type of parameter
    HCtoDCmap *internal_field_map;           // Pointer to internal mapping when current parameter is of type T_STRUCT
    HCtoDCmap *next_field_map;               // Pointer to next struct field when current parameter is a struct member

    HCtoDCmap(const clk_parameter_descriptor_t*, unsigned int, unsigned int, unsigned int);
    virtual ~HCtoDCmap();
    int compute_map(const clk_parameter_descriptor_t*, unsigned int &, unsigned int &, unsigned int, int&, int&);
    unsigned next_offset(unsigned, unsigned &, int &);
    size_t getHostScalarParamSize(const clk_value_type_t) const;
    size_t getScalarAlignment(const clk_value_type_t, bool) const;
    void align_map(unsigned, unsigned, unsigned&, unsigned&, int&);
    unsigned int copy_params(void *, const void *, unsigned int, int&, int&) const;

private:
};


}   // namespace cpu

#endif // CPUMAPPING_HPP_
// Mapping rule
// Long types are treated with 8 byte alignment in runtime when passed in as arguments
// but they are treated with 4 byte alignment in compiler
// Double members have 8 byte alignment when passed as scalar argument
// but have 4 byte alignment as a field inside a struct