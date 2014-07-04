//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef HSABINARY_HPP_
#define HSABINARY_HPP_

#include "top.hpp"
#include "hsadevice.hpp"

#ifndef WITHOUT_FSA_BACKEND

namespace oclhsa {


typedef std::map<std::string, device::Kernel*> NameKernelMap;

class FSAILProgram;

class ClBinary : public device::ClBinary
{
public:
    ClBinary(const Device& dev, BinaryImageFormat bifVer = BIF_VERSION3)
        : device::ClBinary(dev, bifVer)
        {}

    //! Destructor
    ~ClBinary() {}


protected:
    bool setElfTarget() {
        uint32_t target = static_cast<uint32_t>(21);//dev().calTarget());
        assert (((0xFFFF8000 & target) == 0) && "ASIC target ID >= 2^15");
        uint16_t elf_target = (uint16_t)(0x7FFF & target);
        return elfOut()->setTarget(elf_target, amd::OclElf::CAL_PLATFORM);
        return true;
    }
 
private:
    //! Disable default copy constructor
    ClBinary(const ClBinary&);

    //! Disable default operator=
    ClBinary& operator=(const ClBinary&);

    //! Returns the HSA device for this object
    const Device& dev() const { return static_cast<const Device&>(dev_); }

};

} // namespace oclhsa

#endif // WITHOUT_FSA_BACKEND

#endif // HSABINARY_HPP_

