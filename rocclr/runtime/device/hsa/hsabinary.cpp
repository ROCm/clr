//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_FSA_BACKEND


#include "hsabinary.hpp"
#include "hsaprogram.hpp"
#include "hsakernel.hpp"
#include "utils/options.hpp"
#include "os/os.hpp"
#include <string>
#include <sstream>



namespace oclhsa {
    /*
bool
ClBinary::loadKernels(FSAILProgram& program, NameKernelMap &kernels)
{
    return true;

 
    const char _kernel[] = "_kernel";
    const char __FSA_[] = "__FSA_";
    const char _header[] = "_header";
    const char _fsail[] = "_fsail";
    bool hasKernels = false;

    // TODO : jugu
    // Target should be 15 bit maximum. Should check this somewhere.
    uint32_t    target = static_cast<uint32_t>(21);//dev().calTarget());
    uint16_t elf_target;
    amd::OclElf::oclElfPlatform platform;
    if (!elfIn()->getTarget(elf_target, platform) ||
        (platform != amd::OclElf::CAL_PLATFORM)     ||
        ((uint32_t)target != elf_target)) {
        // warning !
        // LogError("The OCL binary image loading failed: different target");

        // LHOWES TODO: target in kannan's elf is wrong so skip this for now
        // We may want a special HSA target or a similar more substantial change.
       // return false;
    }

    for (amd::Sym_Handle sym = elfIn()->nextSymbol(NULL);
         sym != NULL;
         sym = elfIn()->nextSymbol(sym)) {
        amd::OclElf::SymbolInfo symInfo;
        if (!elfIn()->getSymbolInfo(sym, &symInfo)) {
            LogError("LoadKernelFromElf: getSymbolInfo() fails");
            return false;
        }

        std::string elfSymName(symInfo.sym_name);

        const size_t offset = sizeof(__FSA_) - 1;
        if (elfSymName.compare(0, offset, __FSA_) != 0) {
            continue;
        }

        // Assume this elfSymName is associated with a kernel name. The folloiwng code will adjust
        // If it isn't.
        const size_t suffixPos = elfSymName.rfind('_');
        bool isKernel = true;  // assume it is a kernel
        std::string functionName = elfSymName.substr(sizeof(__FSA_)-1, suffixPos-(sizeof(__FSA_)-1));
        //"__OpenCL_";
        //functionName.append(elfSymName.substr(sizeof(__FSA_)-1, suffixPos-(sizeof(__FSA_)-1)));
        //functionName.append("_kernel");  // make the kernel's linkage name

        // Find kernel in map and get its kernel representation
        NameKernelMap::iterator searchIterator = kernels.find(functionName);
        Kernel *currentKernel = 0;
        if( searchIterator == kernels.end() ) {
            // TODO: note, this will need to be decided on based on the the device type. As we have no CPU yet...
            //currentKernel = new Kernel(functionName);
            //kernels[functionName] = currentKernel;
        } else {
            currentKernel = static_cast<oclhsa::Kernel*>(searchIterator->second);
        }


        // Add info for this elf symbol into tempobj's functionNameMap[]
        if (elfSymName.compare(suffixPos, sizeof(_fsail) - 1, _fsail) == 0) {
            
            assert (currentKernel->hasFSAIL() &&
                    "More than one fsail symbol for a kernel");
            // LHOWES TODO: Currently this is using the section address and size because 
            // we only have a single kernel and there is a bug in the current AMP compiler.
            // Kannan is working on fixing this and once we have the symbol address and size 
            // correct in the metadata then we can change this and it'll work properly for
            // multiple kernels.
            std::string options("");
            std::string fsailString(symInfo.sec_addr, symInfo.sec_addr + symInfo.sec_size);
            currentKernel->setFSAIL(fsailString);
            //currentKernel->compile(options);
            
        }


        // LHOWES
        // Hack to assume that this is the AMP path for now
        // until we have kernel metadata we need a way to generate the parameter list.
        {
            device::Kernel::parameters_t parameterList;
            // Is AMP code

            amd::KernelParameterDescriptor  desc;
            desc.name_      = "Functor";
            desc.type_      = T_POINTER;

            desc.size_      = sizeof(void*);
            desc.offset_    = 0;

            // BKENDALL HACK
            desc.typeName_   = "";
            desc.typeQualifier_ = 0;
            desc.accessQualifier_ = 0;
            desc.addressQualifier_ = 0;
            // !BKENDALL HACK

            parameterList.push_back(desc);
           // oclhsa OpenCL integration	
      }

        hasKernels = true;
    }
    

    return hasKernels;

}
    */
/*
bool
ClBinary::clearElfOut()
{
    // Recreate libelf elf object
    if (!elfOut()->Clear()) {
        return false;
    }

    // Need to re-setup target
    return setElfTarget();
}
*/
} // namespace oclhsa

#endif // WITHOUT_FSA_BACKEND
