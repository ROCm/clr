;
; Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
;

ifndef _WIN64
    .386
    .model flat, c
endif ; !_WIN64

OPTION PROLOGUE:NONE
OPTION EPILOGUE:NONE
.code

ifndef _WIN64

_WorkGroup_callKernel proc
    push ebp
    mov ebp, esp
    mov esp, 10h[ebp] ; stackPtr
    mov edx, 0Ch[ebp] ; entryPoint
    push 08h[ebp] ; params
    call edx
    mov esp, ebp
    pop ebp
    ret
_WorkGroup_callKernel endp

_WorkGroup_callKernelProtectedReturn proc
    mov eax, ebp
    mov ebp, esp
    mov esp, 0Ch[ebp] ; stackPtr
    sub esp, CPUKERNEL_STACK_ALIGN
    mov 04h[esp], eax ; save ebp
    mov 00h[esp], ebx ; save ebx
    mov ebx, 00h[ebp] ; return address
    mov edx, 08h[ebp] ; entryPoint

    push 04h[ebp] ; params
    call edx

    mov edx, ebx
    mov ecx, ebp
    mov ebx, 04h[esp] ; load ebx
    mov ebp, 08h[esp] ; load ebp
    mov esp, ecx
    add esp, 04h ; skip return address
    jmp edx
_WorkGroup_callKernelProtectedReturn endp

else ; _WIN64

_WorkGroup_callKernel proc
    push rbp
    mov rbp, rsp
    mov rsp, r8 ; stackPtr
    call rdx
    mov rsp, rbp
    pop rbp
    ret
_WorkGroup_callKernel endp

_WorkGroup_callKernelProtectedReturn proc
    mov rax, rbp
    mov rbp, rsp
    mov rsp, r8 ; stackPtr
    sub rsp, CPUKERNEL_STACK_ALIGN
    mov 08h[rsp], rax ; save rbp
    mov 00h[rsp], rbx ; save rbx
    mov rbx, [rbp] ; return address

    call rdx

    mov rdx, rbx
    mov rcx, rbp
    mov rbx, 00h[rsp] ; load rbx
    mov rbp, 08h[rsp] ; load rbp
    mov rsp, rcx
    add rsp, 08h ; skip return address
    jmp rdx
_WorkGroup_callKernelProtectedReturn endp

endif ; _WIN64

end
