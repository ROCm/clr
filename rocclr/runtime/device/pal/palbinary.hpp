//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef PALBINARY_HPP_
#define PALBINARY_HPP_

#include "top.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palkernel.hpp"

namespace pal {

class ClBinaryHsa : public device::ClBinary
{
public:
    ClBinaryHsa(const Device& dev, BinaryImageFormat bifVer = BIF_VERSION3)
        : device::ClBinary(dev, bifVer)
        {}

    //! Destructor
    ~ClBinaryHsa() {}


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
    ClBinaryHsa(const ClBinaryHsa&);

    //! Disable default operator=
    ClBinaryHsa& operator=(const ClBinaryHsa&);

    //! Returns the HSA device for this object
    const Device& dev() const { return static_cast<const Device&>(dev_); }

};

} // namespace pal

#endif // PALBINARY_HPP_

