#
# Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
#

        .text
.globl _WorkGroup_callKernel
#if defined(ATI_ARCH_X86)
        .type _WorkGroup_callKernel, @function
_WorkGroup_callKernel:
#if defined(_LP64)
        pushq %rbp
        movq %rsp, %rbp
        movq %rdx, %rsp // stackPtr
        call *%rsi
        movq %rbp, %rsp
        popq %rbp
#else // _LP64
        pushl %ebp
        movl %esp, %ebp
        movl 0x10(%ebp), %esp // stackPtr
        movl 0x0C(%ebp), %edx // entryPoint
        movl 0x08(%ebp), %ecx // params
        movl %ecx, (%esp)
        call *%edx
        movl %ebp, %esp
        popl %ebp
#endif // _LP64
        ret
#elif defined(ATI_ARCH_ARM)
	.type	_WorkGroup_callKernel, %function
_WorkGroup_callKernel:
	bx lr	
#endif

.globl _WorkGroup_callKernelProtectedReturn
#if defined(ATI_ARCH_X86)
        .type _WorkGroup_callKernelProtectedReturn, @function
_WorkGroup_callKernelProtectedReturn:
#if defined(_LP64)
        movq %rbp, %rax
        movq %rsp, %rbp
        movq %rdx, %rsp // stackPtr
        subq $CPUKERNEL_STACK_ALIGN, %rsp
        movq %rax, 0x08(%rsp) // save rbp
        movq %rbx, 0x00(%rsp) // save rbx
        movq (%rbp), %rbx // return address

        call *%rsi

        movq %rbx, %rdx
        movq %rbp, %rcx
        movq 0x00(%rsp), %rbx // load rbx
        movq 0x08(%rsp), %rbp // load rbp
        movq %rcx, %rsp
        addq $0x08, %rsp // skip return address
        jmp *%rdx
#else // !_LP64
        movl %ebp, %eax
        movl %esp, %ebp
        movl 0x0C(%ebp), %esp // stackPtr
        subl $CPUKERNEL_STACK_ALIGN, %esp
        movl 0x04(%ebp), %ecx // params
        movl %eax, 0x08(%esp) // save ebp
        movl %ebx, 0x04(%esp) // save ebx
        movl %ecx, 0x00(%esp) // pass params
        movl 0x00(%ebp), %ebx // return address
        movl 0x08(%ebp), %edx // entryPoint

        call *%edx

        movl %ebx, %edx
        movl %ebp, %ecx
        movl 0x04(%esp), %ebx // load ebx
        movl 0x08(%esp), %ebp // load ebp
        movl %ecx, %esp
        addl $0x4, %esp // skip return address
        jmp *%edx
#endif // !_LP64
#elif defined(ATI_ARCH_ARM)
        .type   _WorkGroup_callKernelProtectedReturn, %function
_WorkGroup_callKernelProtectedReturn:
        bx lr   
#endif


.section .note.GNU-stack,"",%progbits

