/*******************************************************************************
 *    The source of the runtime trap handler, "runtimetraphandler.sp3".
 *    The binary is created by the SP3 tool with the following command:
 *
 *       sp3.exe runtimetraphandler.sp3  -hex runtimeTrapCode.hex
 *
 *******************************************************************************

shader main
  asic(TAHITI)  // for SI/CI or asic(VI) for VI
  type(CS)

  // clear wave exception state
  v_clrexcp
  s_waitcnt 0
  //==========================================================================
  // Handle the workaround for HW bug that causes the incorrect TMA value.
  //   Retrieve the TMA values, which are stored at TBA buffer at location
  //   256 (0x100).

  // Construct the memory descriptor with TBA as the start address
  // we are using the registers ttmp[8:11] for that.
  s_mov_b32     ttmp8, tba_lo
  s_and_b32     ttmp9, tba_hi, 0xffff

  // 0x100=256 bytes, which is the size of the buffer to
  // store all the level 2 trap handler info
  s_or_b32      ttmp9,  ttmp9, 0x01000000
  s_mov_b32     ttmp10, 0x00002000
  s_mov_b32     ttmp11, 0x00024fac

  // TMA is stored 256 (0x100) bytes before the TBA value
  s_sub_u32     ttmp8, ttmp8, 0x100

  // Backup the s0 since ttmp registers cannot be target of
  // buffer read instruction
  s_mov_b32      ttmp7, s0
  s_buffer_load_dword  s0, ttmp8, 0x0   // VI: offset=0x0 (bytes)
  s_waitcnt      0
  s_mov_b32      tma_lo, s0
  s_buffer_load_dword  s0, ttmp8, 0x1   // VI: offset=0x4 (bytes)
  s_waitcnt      0
  s_mov_b32      tma_hi, s0
  s_mov_b32      s0, ttmp7

  //===================================================
  //         setup the mmeory descriptor for TMA
  s_mov_b32     ttmp6, 0x18
  s_add_u32     ttmp8, tma_lo, ttmp6
  s_and_b32     ttmp9, tma_hi, 0xffff
  //0x68=104 bytes, which is the size of the buffer to
  //store all the level2 trap handler info
  s_or_b32      ttmp9,  ttmp9, 0x00680000
  s_mov_b32     ttmp10, 0x00002000
  s_mov_b32     ttmp11, 0x00024fac

  //===================================================
  //    backup the TMA values to be restored later
  //      level-one TMA saved in the ttmp6,ttmp7
  s_mov_b32     ttmp6, tma_lo
  s_mov_b32     ttmp7, tma_hi

  //===================================================
  //   setup the TMA for the level-two trap handler
  //      level-two TMA saved in tma_hi, tma_lo
  s_mov_b32     ttmp3, s0
  s_buffer_load_dword  s0, ttmp8, 0x2   // VI: offset=0x8 (bytes)
  s_waitcnt     0x0000
  s_mov_b32     tma_lo, s0

  s_buffer_load_dword  s0, ttmp8, 0x3   // VI: offset=0xc (bytes)
  s_waitcnt     0x0000
  s_mov_b32     tma_hi, s0

  //===================================================
  //   setup the TBA for the level-two trap handler
  //       level-two TBA saved in ttmp9, ttmp8
  s_buffer_load_dword  s0, ttmp8, 0x0   // VI: offset=0x0 (bytes)
  s_waitcnt     0x0000
  s_mov_b32     ttmp2, s0

  s_buffer_load_dword  s0, ttmp8, 0x1   // VI: offset=0x4 (bytes)
  s_waitcnt     0x0000

  //swap the values of s0 and ttmp3 without using other registers
  s_xor_b32     ttmp3, s0, ttmp3
  s_xor_b32     s0,    s0, ttmp3
  s_xor_b32     ttmp3, s0, ttmp3

  //store the debug trap handler start address in ttmp8,9
  s_mov_b32    ttmp8, ttmp2
  s_mov_b32    ttmp9, ttmp3

  //===================================================
  //         get the pc value to resume execution
  s_getpc_b64  [ttmp2, ttmp3]
  s_add_u32    ttmp2, ttmp2, 0x8

  //===================================================
  //set the pc value to jump to the debug trap handler
  s_setpc_b64   [ttmp8, ttmp9]

  //===================================================
  //              restore the tamp values
  s_mov_b32    tma_hi, ttmp7
  s_mov_b32    tma_lo, ttmp6

  label_return:
  //===================================================
  //   return from the trap handler to the saved PC
  s_and_b32     ttmp1, ttmp1, 0xffff
  s_rfe_b64     [ttmp0,ttmp1]

end

*******************************************************************************/

///  shader codes with "asic(TAHITI)" instruction
static const uint32_t RuntimeTrapCode[] = {
    0x7e008200, 0xbf8c0000, 0xbef8036c, 0x8779ff6d, 0x0000ffff, 0x8879ff79, 0x01000000, 0xbefa03ff,
    0x00002000, 0xbefb03ff, 0x00024fac, 0x80f8ff78, 0x00000100, 0xbef70300, 0xc2007900, 0xbf8c0000,
    0xbeee0300, 0xc2007901, 0xbf8c0000, 0xbeef0300, 0xbe800377, 0xbef60398, 0x8078766e, 0x8779ff6f,
    0x0000ffff, 0x8879ff79, 0x00680000, 0xbefa03ff, 0x00002000, 0xbefb03ff, 0x00024fac, 0xbef6036e,
    0xbef7036f, 0xbef30300, 0xc2007902, 0xbf8c0000, 0xbeee0300, 0xc2007903, 0xbf8c0000, 0xbeef0300,
    0xc2007900, 0xbf8c0000, 0xbef20300, 0xc2007901, 0xbf8c0000, 0x89737300, 0x89007300, 0x89737300,
    0xbef80372, 0xbef90373, 0xbef21f00, 0x80728872, 0xbe802078, 0xbeef0377, 0xbeee0376, 0x8771ff71,
    0x0000ffff, 0xbe802270};


///  shader codes with "asic(VI)" instruction
static const uint32_t RuntimeTrapCodeVi[] = {
    0x7e006a00, 0xbf8c0000, 0xbef8006c, 0x8679ff6d, 0x0000ffff, 0x8779ff79, 0x01000000, 0xbefa00ff,
    0x00002000, 0xbefb00ff, 0x00024fac, 0x80f8ff78, 0x00000100, 0xbef70000, 0xc022003c, 0x00000000,
    0xbf8c0000, 0xbeee0000, 0xc022003c, 0x00000004, 0xbf8c0000, 0xbeef0000, 0xbe800077, 0xbef60098,
    0x8078766e, 0x8679ff6f, 0x0000ffff, 0x8779ff79, 0x00680000, 0xbefa00ff, 0x00002000, 0xbefb00ff,
    0x00024fac, 0xbef6006e, 0xbef7006f, 0xbef30000, 0xc022003c, 0x00000008, 0xbf8c0000, 0xbeee0000,
    0xc022003c, 0x0000000c, 0xbf8c0000, 0xbeef0000, 0xc022003c, 0x00000000, 0xbf8c0000, 0xbef20000,
    0xc022003c, 0x00000004, 0xbf8c0000, 0x88737300, 0x88007300, 0x88737300, 0xbef80072, 0xbef90073,
    0xbef21c00, 0x80728872, 0xbe801d78, 0xbeef0077, 0xbeee0076, 0x8671ff71, 0x0000ffff, 0xbe801f70};
