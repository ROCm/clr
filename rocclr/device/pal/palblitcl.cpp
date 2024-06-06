/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

namespace amd::pal {

#define RUNTIME_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = RUNTIME_KERNEL(
\n
extern void __amd_scheduler(__global void*, __global void*, uint);
\n
__kernel void __amd_rocclr_scheduler(__global void* queue, __global void* params, uint paramIdx) {
  __amd_scheduler(queue, params, paramIdx);
}
\n);

const char* SchedulerSourceCode20 = RUNTIME_KERNEL(
\n
extern void __amd_scheduler_pal(__global void*, __global void*, uint);
\n
 __kernel void __amd_rocclr_scheduler(__global void* queue, __global void* params,
                                         uint paramIdx) {
  __amd_scheduler_pal(queue, params, paramIdx);
}
\n);

const char* TrapHandlerCode = RUNTIME_KERNEL(
\n.if .amdgcn.gfx_generation_number < 12
\n.set SQ_WAVE_PC_HI_ADDRESS_MASK              , 0xFFFF
\n.set SQ_WAVE_PC_HI_HT_SHIFT                  , 24
\n.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT             , 16
\n.set SQ_WAVE_PC_HI_TRAP_ID_SIZE              , 8
\n.set SQ_WAVE_PC_HI_TRAP_ID_BFE, (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
\n.set SQ_WAVE_STATUS_HALT_SHIFT, 13
\n.set SQ_WAVE_STATUS_HALT_BFE, (SQ_WAVE_STATUS_HALT_SHIFT | (1 << 16))
\n.set SQ_WAVE_TRAPSTS_MEM_VIOL_SHIFT, 8
\n.set SQ_WAVE_TRAPSTS_ILLEGAL_INST_SHIFT      , 11
\n.set SQ_WAVE_TRAPSTS_XNACK_ERROR_SHIFT       , 28
\n.set SQ_WAVE_TRAPSTS_MATH_EXCP               , 0x7F
\n.set SQ_WAVE_MODE_EXCP_EN_SHIFT              , 12
\n.set TRAP_ID_ABORT                           , 2
\n.set TRAP_ID_DEBUGTRAP                       , 3
\n.set DOORBELL_ID_SIZE                        , 10
\n.set DOORBELL_ID_MASK                        , ((1 << DOORBELL_ID_SIZE) - 1)
\n.set EC_QUEUE_WAVE_ABORT_M0                  , (1 << (DOORBELL_ID_SIZE + 0))
\n.set EC_QUEUE_WAVE_TRAP_M0                   , (1 << (DOORBELL_ID_SIZE + 1))
\n.set EC_QUEUE_WAVE_MATH_ERROR_M0             , (1 << (DOORBELL_ID_SIZE + 2))
\n.set EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0    , (1 << (DOORBELL_ID_SIZE + 3))
\n.set EC_QUEUE_WAVE_MEMORY_VIOLATION_M0       , (1 << (DOORBELL_ID_SIZE + 4))
\n.set EC_QUEUE_WAVE_APERTURE_VIOLATION_M0     , (1 << (DOORBELL_ID_SIZE + 5))
\n.set TTMP6_WAVE_STOPPED_SHIFT                , 30
\n.set TTMP6_SAVED_STATUS_HALT_SHIFT           , 29
\n.set TTMP6_SAVED_STATUS_HALT_MASK, (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
\n.set TTMP6_SAVED_TRAP_ID_SHIFT, 25
\n.set TTMP6_SAVED_TRAP_ID_SIZE, 4
\n.set TTMP6_SAVED_TRAP_ID_MASK, (((1 << TTMP6_SAVED_TRAP_ID_SIZE) - 1) << TTMP6_SAVED_TRAP_ID_SHIFT)
\n.set TTMP6_SAVED_TRAP_ID_BFE, (TTMP6_SAVED_TRAP_ID_SHIFT | (TTMP6_SAVED_TRAP_ID_SIZE << 16))
\n.set TTMP_PC_HI_SHIFT, 7
\n.set TTMP_DEBUG_ENABLED_SHIFT, 23
\n.if .amdgcn.gfx_generation_number == 9
\n.set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT, 26
\n.set SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT, 15
\n.set SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK, 0x1F8000
\n.elseif .amdgcn.gfx_generation_number == 10 &&.amdgcn.gfx_generation_minor < 3
\n.set TTMP_SAVE_REPLAY_W64H_SHIFT, 31
\n.set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT, 24
\n.set SQ_WAVE_IB_STS_REPLAY_W64H_SHIFT, 25
\n.set SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT, 15
\n.set SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK, 0x3F8000
\n.set SQ_WAVE_IB_STS_REPLAY_W64H_MASK, 0x2000000
\n.endif
\n.if .amdgcn.gfx_generation_number == 9 &&.amdgcn.gfx_generation_minor >= 4
\n.set TTMP11_TTMPS_SETUP_SHIFT,    31
\n.endif
// ABI between first and second level trap handler:
//   ttmp0 = PC[31:0]
//   ttmp12 = SQ_WAVE_STATUS
//   ttmp14 = TMA[31:0]
//   ttmp15 = TMA[63:32]
// gfx9:
//   ttmp1 = 0[2:0], PCRewind[3:0], HostTrap[0], TrapId[7:0], PC[47:32]
// gfx906/gfx908/gfx90a:
//   ttmp11 = SQ_WAVE_IB_STS[20:15], 0[1:0], DebugEnabled[0], 0[15:0], NoScratch[0], WaveIdInWG[5:0]
// gfx940/gfx941/gfx942:
//   ttmp13 = SQ_WAVE_IB_STS[20:15], 0[1:0], DebugEnabled[0], 0[22:0]
// gfx10:
//   ttmp1 = 0[0], PCRewind[5:0], HostTrap[0], TrapId[7:0], PC[47:32]
// gfx1010:
//   ttmp11 = SQ_WAVE_IB_STS[25], SQ_WAVE_IB_STS[21:15], DebugEnabled[0], 0[15:0], NoScratch[0], WaveIdInWG[5:0]
// gfx1030/gfx1100:
//   ttmp11 = 0[7:0], DebugEnabled[0], 0[15:0], NoScratch[0], WaveIdInWG[5:0]
\n .globl	trap_entry
\n .type	trap_entry,@function
\ntrap_entry:
  // Branch if not a trap (an exception instead).
\n  s_bfe_u32            ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
\n  s_cbranch_scc0       .no_skip_debugtrap
  // If caused by s_trap then advance PC.
\n  s_bitcmp1_b32        ttmp1, SQ_WAVE_PC_HI_HT_SHIFT
\n  s_cbranch_scc1       .not_s_trap
\n  s_add_u32            ttmp0, ttmp0, 0x4
\n  s_addc_u32           ttmp1, ttmp1, 0x0
\n.not_s_trap:
  // If llvm.debugtrap and debugger is not attached.
\n  s_cmp_eq_u32         ttmp2, TRAP_ID_DEBUGTRAP
\n  s_cbranch_scc0       .no_skip_debugtrap
\n.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor < 4) || .amdgcn.gfx_generation_number == 10
\n  s_bitcmp0_b32        ttmp11, TTMP_DEBUG_ENABLED_SHIFT
\n.else
\n  s_bitcmp0_b32        ttmp13, TTMP_DEBUG_ENABLED_SHIFT
\n.endif
\n  s_cbranch_scc0       .no_skip_debugtrap
  // Ignore llvm.debugtrap.
\n  s_branch             .exit_trap
\n.no_skip_debugtrap:
  // Save trap id and halt status in ttmp6.
\n  s_andn2_b32          ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
\n  s_min_u32            ttmp2, ttmp2, 0xF
\n  s_lshl_b32           ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
\n  s_or_b32             ttmp6, ttmp6, ttmp2
\n  s_bfe_u32            ttmp2, ttmp12, SQ_WAVE_STATUS_HALT_BFE
\n  s_lshl_b32           ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
\n  s_or_b32             ttmp6, ttmp6, ttmp2
  // Fetch doorbell id for our queue.
\n.if .amdgcn.gfx_generation_number < 11
\n  s_mov_b32            ttmp2, exec_lo
\n  s_mov_b32            ttmp3, exec_hi
\n  s_mov_b32            exec_lo, 0x80000000
\n  s_sendmsg            sendmsg(MSG_GET_DOORBELL)
\n.wait_sendmsg:
\n  s_nop                0x7
\n  s_bitcmp0_b32        exec_lo, 0x1F
\n  s_cbranch_scc0       .wait_sendmsg
\n  s_mov_b32            exec_hi, ttmp3
  // Restore exec_lo, move the doorbell_id into ttmp3
\n  s_and_b32            ttmp3, exec_lo, DOORBELL_ID_MASK
\n  s_mov_b32            exec_lo, ttmp2
\n.else
\n  s_sendmsg_rtn_b32    ttmp3, sendmsg(MSG_RTN_GET_DOORBELL)
\n  s_waitcnt            lgkmcnt(0)
\n  s_and_b32            ttmp3, ttmp3, DOORBELL_ID_MASK
\n.endif
 // Map trap reason to an exception code.
\n  s_getreg_b32         ttmp2, hwreg(HW_REG_TRAPSTS)
\n
\n  s_bitcmp1_b32        ttmp2, SQ_WAVE_TRAPSTS_XNACK_ERROR_SHIFT
\n  s_cbranch_scc0       .not_memory_violation
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_MEMORY_VIOLATION_M0
  // Aperture violation requires XNACK_ERROR == 0.
\n  s_branch             .not_aperture_violation
\n.not_memory_violation:
\n  s_bitcmp1_b32        ttmp2, SQ_WAVE_TRAPSTS_MEM_VIOL_SHIFT
\n  s_cbranch_scc0       .not_aperture_violation
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_APERTURE_VIOLATION_M0
\n.not_aperture_violation:
\n  s_bitcmp1_b32        ttmp2, SQ_WAVE_TRAPSTS_ILLEGAL_INST_SHIFT
\n  s_cbranch_scc0       .not_illegal_instruction
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0
\n.not_illegal_instruction:
\n  s_and_b32            ttmp2, ttmp2, SQ_WAVE_TRAPSTS_MATH_EXCP
\n  s_cbranch_scc0       .not_math_exception
\n  s_getreg_b32         ttmp7, hwreg(HW_REG_MODE)
\n  s_lshl_b32           ttmp2, ttmp2, SQ_WAVE_MODE_EXCP_EN_SHIFT
\n  s_and_b32            ttmp2, ttmp2, ttmp7
\n  s_cbranch_scc0       .not_math_exception
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0
\n.not_math_exception:
\n  s_bfe_u32            ttmp2, ttmp6, TTMP6_SAVED_TRAP_ID_BFE
\n  s_cmp_eq_u32         ttmp2, TRAP_ID_ABORT
\n  s_cbranch_scc0       .not_abort_trap
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_ABORT_M0
\n.not_abort_trap:
  // If no other exception was flagged then report a generic error.
\n  s_andn2_b32          ttmp2, ttmp3, DOORBELL_ID_MASK
\n  s_cbranch_scc1       .send_interrupt
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0
\n.send_interrupt:
  // m0 = interrupt data = (exception_code << DOORBELL_ID_SIZE) | doorbell_id
\n  s_mov_b32            ttmp2, m0
\n  s_mov_b32            m0, ttmp3
\n  s_nop                0x0 // Manually inserted wait states
\n  s_sendmsg            sendmsg(MSG_INTERRUPT)
\n  s_waitcnt            lgkmcnt(0) // Wait for the message to go out.
\n  s_mov_b32            m0, ttmp2
  // Parking the wave requires saving the original pc in the preserved ttmps.
  // Register layout before parking the wave:
  //
  // ttmp7: 0[31:0]
  // ttmp11: 1st_level_ttmp11[31:23] 0[15:0] 1st_level_ttmp11[6:0]
  //
  // After parking the wave:
  //
  // ttmp7:  pc_lo[31:0]
  // ttmp11: 1st_level_ttmp11[31:23] pc_hi[15:0] 1st_level_ttmp11[6:0]
\n.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor < 4) || (.amdgcn.gfx_generation_number == 10 && .amdgcn.gfx_generation_minor < 3)
\n  // Save the PC
\n  s_mov_b32            ttmp7, ttmp0
\n  s_and_b32            ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK
\n  s_lshl_b32           ttmp1, ttmp1, TTMP_PC_HI_SHIFT
\n  s_andn2_b32          ttmp11, ttmp11, (SQ_WAVE_PC_HI_ADDRESS_MASK << TTMP_PC_HI_SHIFT)
\n  s_or_b32             ttmp11, ttmp11, ttmp1
  // Park the wave
\n  s_getpc_b64          [ttmp0, ttmp1]
\n  s_add_u32            ttmp0, ttmp0, .parked - .
\n  s_addc_u32           ttmp1, ttmp1, 0x0
\n.endif
\n.halt_wave:
  // Halt the wavefront upon restoring STATUS below.
\n  s_bitset1_b32        ttmp6, TTMP6_WAVE_STOPPED_SHIFT
\n  s_bitset1_b32        ttmp12, SQ_WAVE_STATUS_HALT_SHIFT
\n.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
\n  s_bitcmp1_b32        ttmp11, TTMP11_TTMPS_SETUP_SHIFT
\n  s_cbranch_scc1       .ttmps_initialized
\n  s_mov_b32            ttmp4, 0
\n  s_mov_b32            ttmp5, 0
\n  s_bitset1_b32        ttmp11, TTMP11_TTMPS_SETUP_SHIFT
\n.ttmps_initialized:
\n.endif
\n.exit_trap:
  // Restore SQ_WAVE_IB_STS.
\n.if .amdgcn.gfx_generation_number == 9
\n.if .amdgcn.gfx_generation_minor < 4
\n  s_lshr_b32           ttmp2, ttmp11, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT)
\n.else
\n  s_lshr_b32           ttmp2, ttmp13, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT)
\n.endif
\n  s_and_b32            ttmp2, ttmp2, SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK
\n  s_setreg_b32         hwreg(HW_REG_IB_STS), ttmp2
\n.elseif .amdgcn.gfx_generation_number == 10 && .amdgcn.gfx_generation_minor < 3
\n  s_lshr_b32           ttmp2, ttmp11, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT)
\n  s_and_b32            ttmp3, ttmp2, SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK
\n  s_lshr_b32           ttmp2, ttmp11, (TTMP_SAVE_REPLAY_W64H_SHIFT - SQ_WAVE_IB_STS_REPLAY_W64H_SHIFT)
\n  s_and_b32            ttmp2, ttmp2, SQ_WAVE_IB_STS_REPLAY_W64H_MASK
\n  s_or_b32             ttmp2, ttmp2, ttmp3
\n  s_setreg_b32         hwreg(HW_REG_IB_STS), ttmp2
\n.endif
  // Restore SQ_WAVE_STATUS.
\n  s_and_b64            exec, exec, exec // Restore STATUS.EXECZ, not writable by s_setreg_b32
\n  s_and_b64            vcc, vcc, vcc    // Restore STATUS.VCCZ, not writable by s_setreg_b32
\n  s_setreg_b32         hwreg(HW_REG_STATUS), ttmp12
  // Return to original (possibly modified) PC.
\n  s_rfe_b64            [ttmp0, ttmp1]
\n.parked:
\n  s_trap               0x2
\n  s_branch             .parked
\n.else
\n.set DOORBELL_ID_SIZE                          , 10
\n.set DOORBELL_ID_MASK                          , ((1 << DOORBELL_ID_SIZE) - 1)
\n.set EC_QUEUE_WAVE_ABORT_M0                    , (1 << (DOORBELL_ID_SIZE + 0))
\n.set EC_QUEUE_WAVE_TRAP_M0                     , (1 << (DOORBELL_ID_SIZE + 1))
\n.set EC_QUEUE_WAVE_MATH_ERROR_M0               , (1 << (DOORBELL_ID_SIZE + 2))
\n.set EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0      , (1 << (DOORBELL_ID_SIZE + 3))
\n.set EC_QUEUE_WAVE_MEMORY_VIOLATION_M0         , (1 << (DOORBELL_ID_SIZE + 4))
\n.set EC_QUEUE_WAVE_APERTURE_VIOLATION_M0       , (1 << (DOORBELL_ID_SIZE + 5))
\n.set SQ_WAVE_EXCP_FLAG_PRIV_MEMVIOL_SHIFT      , 4
\n.set SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT           , 7
\n.set SQ_WAVE_EXCP_FLAG_PRIV_ILLEGAL_INST_SHIFT , 6
\n.set SQ_WAVE_EXCP_FLAG_PRIV_XNACK_ERROR_SHIFT  , 8
\n.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT    , 0
\n.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE     , 6
\n.set SQ_WAVE_TRAP_CTRL_MATH_EXCP_SHIFT         , 0
\n.set SQ_WAVE_TRAP_CTRL_MATH_EXCP_SIZE          , 6
\n.set SQ_WAVE_PC_HI_ADDRESS_MASK                , 0xFFFF
\n.set SQ_WAVE_PC_HI_TRAP_ID_BFE                 , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
\n.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT               , 28
\n.set SQ_WAVE_PC_HI_TRAP_ID_SIZE                , 4
\n.set SQ_WAVE_STATE_PRIV_HALT_BFE               , (SQ_WAVE_STATE_PRIV_HALT_SHIFT | (1 << 16))
\n.set SQ_WAVE_STATE_PRIV_HALT_SHIFT             , 14
\n.set TRAP_ID_ABORT                             , 2
\n.set TRAP_ID_DEBUGTRAP                         , 3
\n.set TTMP6_SAVED_STATUS_HALT_MASK              , (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
\n.set TTMP6_SAVED_STATUS_HALT_SHIFT             , 29
\n.set TTMP6_SAVED_TRAP_ID_BFE                   , (TTMP6_SAVED_TRAP_ID_SHIFT | (TTMP6_SAVED_TRAP_ID_SIZE << 16))
\n.set TTMP6_SAVED_TRAP_ID_MASK                  , (((1 << TTMP6_SAVED_TRAP_ID_SIZE) - 1) << TTMP6_SAVED_TRAP_ID_SHIFT)
\n.set TTMP6_SAVED_TRAP_ID_SHIFT                 , 25
\n.set TTMP6_SAVED_TRAP_ID_SIZE                  , 4
\n.set TTMP6_WAVE_STOPPED_SHIFT                  , 30
\n.set TTMP8_DEBUG_FLAG_SHIFT                    , 31
\n.set TTMP11_DEBUG_ENABLED_SHIFT                , 23
\n.set TTMP_PC_HI_SHIFT                          , 7
\n
\n// ABI between first and second level trap handler:
\n//   { ttmp1, ttmp0 } = TrapID[3:0], zeros, PC[47:0]
\n//   ttmp11 = 0[7:0], DebugEnabled[0], 0[15:0], NoScratch[0], 0[5:0]
\n//   ttmp12 = SQ_WAVE_STATE_PRIV
\n//   ttmp14 = TMA[31:0]
\n//   ttmp15 = TMA[63:32]
\n
\ntrap_entry:
\n  // Branch if not a trap (an exception instead).
\n  s_bfe_u32            ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
\n  s_cbranch_scc0       .no_skip_debugtrap
\n
\n  // If caused by s_trap then advance PC.
\n  s_add_u32            ttmp0, ttmp0, 0x4
\n  s_addc_u32           ttmp1, ttmp1, 0x0
\n
\n.not_s_trap:
\n  // If llvm.debugtrap and debugger is not attached.
\n  s_cmp_eq_u32         ttmp2, TRAP_ID_DEBUGTRAP
\n  s_cbranch_scc0       .no_skip_debugtrap
\n
\n  s_bitcmp0_b32        ttmp11, TTMP11_DEBUG_ENABLED_SHIFT
\n  s_cbranch_scc0       .no_skip_debugtrap
\n
\n  // Ignore llvm.debugtrap.
\n  s_branch             .exit_trap
\n
\n.no_skip_debugtrap:
\n  // Save trap id and halt status in ttmp6.
\n  s_andn2_b32          ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
\n  s_min_u32            ttmp2, ttmp2, 0xF
\n  s_lshl_b32           ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
\n  s_or_b32             ttmp6, ttmp6, ttmp2
\n  s_bfe_u32            ttmp2, ttmp12, SQ_WAVE_STATE_PRIV_HALT_BFE
\n  s_lshl_b32           ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
\n  s_or_b32             ttmp6, ttmp6, ttmp2
\n
\n  // Fetch doorbell id for our queue.
\n  s_sendmsg_rtn_b32    ttmp3, sendmsg(MSG_RTN_GET_DOORBELL)
\n  s_wait_kmcnt         0
\n  s_and_b32            ttmp3, ttmp3, DOORBELL_ID_MASK
\n
\n  s_getreg_b32         ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)
\n
\n  s_bitcmp1_b32        ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_XNACK_ERROR_SHIFT
\n  s_cbranch_scc0       .not_memory_violation
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_MEMORY_VIOLATION_M0
\n
\n  // Aperture violation requires XNACK_ERROR == 0.
\n  s_branch             .not_aperture_violation
\n
\n.not_memory_violation:
\n  s_bitcmp1_b32        ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_MEMVIOL_SHIFT
\n  s_cbranch_scc0       .not_aperture_violation
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_APERTURE_VIOLATION_M0
\n
\n.not_aperture_violation:
\n  s_bitcmp1_b32        ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_ILLEGAL_INST_SHIFT
\n  s_cbranch_scc0       .not_illegal_instruction
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0
\n
\n.not_illegal_instruction:
\n  s_getreg_b32         ttmp2, hwreg(HW_REG_EXCP_FLAG_USER, SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT, SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE)
\n  s_cbranch_scc0       .not_math_exception
\n  s_getreg_b32         ttmp10, hwreg(HW_REG_TRAP_CTRL, SQ_WAVE_TRAP_CTRL_MATH_EXCP_SHIFT, SQ_WAVE_TRAP_CTRL_MATH_EXCP_SIZE)
\n  s_and_b32            ttmp2, ttmp2, ttmp10
\n
\n  s_cbranch_scc0       .not_math_exception
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0
\n
\n.not_math_exception:
\n  s_bfe_u32            ttmp2, ttmp6, TTMP6_SAVED_TRAP_ID_BFE
\n  s_cmp_eq_u32         ttmp2, TRAP_ID_ABORT
\n  s_cbranch_scc0       .not_abort_trap
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_ABORT_M0
\n
\n.not_abort_trap:
\n  // If no other exception was flagged then report a generic error.
\n  s_andn2_b32          ttmp2, ttmp3, DOORBELL_ID_MASK
\n  s_cbranch_scc1       .send_interrupt
\n  s_or_b32             ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0
\n
\n.send_interrupt:
\n  // m0 = interrupt data = (exception_code << DOORBELL_ID_SIZE) | doorbell_id
\n  s_mov_b32            ttmp2, m0
\n  s_mov_b32            m0, ttmp3
\n  s_nop                0x0 // Manually inserted wait states
\n  s_sendmsg            sendmsg(MSG_INTERRUPT)
\n  // Wait for the message to go out.
\n  s_wait_kmcnt         0
\n  s_mov_b32            m0, ttmp2
\n
\n  // Parking the wave requires saving the original pc in the preserved ttmps.
\n  // Register layout before parking the wave:
\n  //
\n  // ttmp10: ?[31:0]
\n  // ttmp11: 1st_level_ttmp11[31:23] 0[15:0] 1st_level_ttmp11[6:0]
\n  //
\n  // After parking the wave:
\n  //
\n  // ttmp10: pc_lo[31:0]
\n  // ttmp11: 1st_level_ttmp11[31:23] pc_hi[15:0] 1st_level_ttmp11[6:0]
\n  //
\n  // Save the PC
\n  s_mov_b32            ttmp10, ttmp0
\n  s_and_b32            ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK
\n  s_lshl_b32           ttmp1, ttmp1, TTMP_PC_HI_SHIFT
\n  s_andn2_b32          ttmp11, ttmp11, (SQ_WAVE_PC_HI_ADDRESS_MASK << TTMP_PC_HI_SHIFT)
\n  s_or_b32             ttmp11, ttmp11, ttmp1
\n
\n  // Park the wave
\n  s_getpc_b64          [ttmp0, ttmp1]
\n  s_add_u32            ttmp0, ttmp0, .parked - .
\n  s_addc_u32           ttmp1, ttmp1, 0x0
\n
\n.halt_wave:
\n  // Halt the wavefront upon restoring STATUS below.
\n  s_bitset1_b32        ttmp6, TTMP6_WAVE_STOPPED_SHIFT
\n  s_bitset1_b32        ttmp12, SQ_WAVE_STATE_PRIV_HALT_SHIFT
\n
\n  // Initialize TTMP registers
\n  s_bitcmp1_b32        ttmp8, TTMP8_DEBUG_FLAG_SHIFT
\n  s_cbranch_scc1       .ttmps_initialized
\n  s_mov_b32            ttmp4, 0
\n  s_mov_b32            ttmp5, 0
\n  s_bitset1_b32        ttmp8, TTMP8_DEBUG_FLAG_SHIFT
\n.ttmps_initialized:
\n
\n.exit_trap:
\n  // Restore SQ_WAVE_STATUS.
\n  s_and_b64            exec, exec, exec // Restore STATUS.EXECZ, not writable by s_setreg_b32
\n  s_and_b64            vcc, vcc, vcc    // Restore STATUS.VCCZ, not writable by s_setreg_b32
\n  s_setreg_b32         hwreg(HW_REG_STATE_PRIV), ttmp12
\n
\n  // Return to original (possibly modified) PC.
\n  s_rfe_b64            [ttmp0, ttmp1]
\n
\n.parked:
\n  s_trap               0x2
\n  s_branch             .parked
\n
\n// Add s_code_end padding so instruction prefetch always has something to read.
\n//.rept (256 - ((. - trap_entry) % 64)) / 4
\n 64 s_code_end
\n//.endr
\n.endif
\n);
}  // namespace amd::pal
