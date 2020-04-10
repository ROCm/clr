 /* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#ifndef __INI_VALUES_H__
#define __INI_VALUES_H__

#include "cm_string.h"


const cmString section("CAL");
const cmString INI_FILE("cal.ini");

/* VSYNC COMMENTS
0 - always off
1 - app preference (default off)
2 - app preference (default on)
3 - always on
*/
const cmString CAL_OGLWAITVERTICALSYNC("VSyncControl");

// Private panel setting for V-sync control
const cmString CAL_ENABLETEARFREESWAP("VSyncControl");

// Public panel setting to set max anisotropy: 0=app pref, 2=2x, 4=4x, 8=8x, 16=16x
const cmString CAL_OGLMAXANISOTROPY("MaxAnisotropy");

// Public panel setting to select performance Aniso
const cmString CAL_OGLANISOPERF("AnisoPerf");


// Public panel setting to select quality mode
const cmString CAL_OGLANISOQUAL("AnisoQuality");

// Public panel setting
const cmString CAL_OGLANISOTYPE("AnisoType");

// Private panel
const cmString CAL_ENABLEANISOTROPICFILTERING("AnisoFiltering");

// Public panel setting
const cmString CAL_OGLALIASSLIDER("AnisoDegree");

// Public panel setting for LOD bias; ranges from 0(high quality) to 3(high performance);
const cmString CAL_OGLLODBIAS("TextureLod");

// Public panel setting to force Z buffer depth
const cmString CAL_OGLFORCEZBUFFERDEPTH("ForceZBufferDepth");

// Public panel setting to select alpha dither method
const cmString CAL_OGLALPHADITHERMETHOD("DitherAlpha");

// Private Panel Setting for setting multisample value for FSAA
const cmString CAL_MULTISAMPLE("Multisample");

// Public Panel setting for forcing AA
const cmString CAL_ACE_OGLENABLEFSAA("AntiAlias");

// Public panel setting to Enable fast full scene anti-aliasing
const cmString CAL_OGLENABLEFASTFULLSCENEAA("FSAAPerfMode");

//Private Panel setting to force FSAA on
const cmString CAL_ENABLEFASTFULLSCENEAA("FastFullSceneAntiAlias");


// Public panel setting to set full scene anti-aliasing scale.
// Acceptable values are 0, 2-6
const cmString CAL_OGLFULLSCENEAASCALE("AntiAliasSamples");

// Private panel setting to force FSAA, Acceptable values are 0, 2-6.
const cmString CAL_FULLSCENEAASCALE("FullSceneAntiAliasScale");

// Public panel setting to enable triple-buffering
const cmString CAL_OGLENABLETRIPLEBUFFERING("EnableTripleBuffering");

// Public panel setting to set texture optimization
const cmString CAL_OGLTEXTUREOPT("TextureOpt");

// Public panel settings to set postprocessing shaders
const cmString CAL_OGLSELECTEDSWAPEFFECT("SwapEffect");

// Public panel settings to control CatalystAI settings
const cmString CAL_OGLCATALYSTAI("CatalystAI");


// Public panel settings to set postprocessing shaders
const cmString CAL_OGLSUPPORTEDSWAPEFFECTS("SupportedSwapEffects");

const cmString CAL_OGLCUSTOMSWAPSOURCEFILE("CustomSwapSourceFile");

//Public panel setting for allowing special pixel shaders to be applied at swap time.
const cmString CAL_SPECIALSWAP("SpecialSwap");

//Public panel setting for special swap file
const cmString CAL_SPECIALSWAPFILE("SpecialSwapFile");



// Private Panel specific Defines
//

// Private panel setting to force SW path
const cmString CAL_PICKSOFTWARE("PickSoftware");

// Private panel setting to force Microsoft path
const cmString CAL_PICKSOFTWAREMICROSOFT("PickSoftwareMicrosoft");

// Private Panel setting to enable TCL (versus forcing SW TCL)
const cmString CAL_ENABLETCL("EnableTCL");

// Private Panel setting to control HW Flips
const cmString CAL_ALLOWHWFLIP("AllowHWFlip");

// Private Panel setting to allow Z compression
const cmString CAL_ENABLEZCOMPRESSION("ZCompression");

// Private Panel setting to use fast z clears
const cmString CAL_ENABLEFASTZMASKCLEAR("FastZMaskClear");

// Private Panel setting to enable hierarchical Z
const cmString CAL_ENABLEHIERARCHICALZ("HierachicalZ");

// Private Panel setting to enable/disable cmask clears
const cmString CAL_ENABLECMASKCLEARS("MaskClears");

// Private Panel setting to force cmask clear after swap
const cmString CAL_CLEARCMASKAFTERSWAP("ClearCMaskAfterSwap");

// Private Panel setting to enable cmask compression
const cmString CAL_ENABLECMASKCOMPRESSION("CMaskCompression");

// Private Panel setting to force LOD Bias
const cmString CAL_LODBIAS("LODBias");

// Private Panel setting enable fast trilinear
const cmString CAL_FASTTRILINEAR("FastTrilinear");

// Private Panel setting to force clears to be skipped
const cmString CAL_DISABLECLEAR("DisableClear");

// Private Panel setting to control swapping
const cmString CAL_DISABLESWAP("DisableSwap");

// Private Panel setting to force HW idle after submit
const cmString CAL_WAITFORIDLEAFTERSUBMIT("WaitForIdleAfterSubmit");

// Private Panel setting to force single buffered rendering
const cmString CAL_FORCESINGLEBUFFER("ForceSingleBuffer");

// Private Panel setting to force buffer config for single buffered
// configs
const cmString CAL_SINGLE_BUF_CONFIG("SingleBufferConfig");

// Private Panel setting to force buffer config for double buffered
const cmString CAL_DOUBLE_BUF_CONFIG("DoubleBufferConfig");

// Private Panel setting to cause driver to breka on load
const cmString CAL_BREAK_ON_LOAD("BreakOnLoad");

// Private Panel setting for asserting when we set an error
const cmString CAL_ASSERTONERROR("AssertOnError");

// Private Panel setting to turn on shader dumping
const cmString CAL_ENABLESHADERDUMP("EnableShaderDump");

// Private Panel setting to turn on packet dumping
const cmString CAL_ENABLEPACKETDUMP("EnablePacketDump");

// Private Panel setting to set location of packet dump
const cmString CAL_PACKETDUMPLOCATION("PacketDumpLocation");

// Private Panel setting to set what type of file to be written
const cmString CAL_PACKETDUMPTYPE("PacketDumpType");

// Private Panel setting to select file overwrite
const cmString CAL_ONLYSAVELASTPACKET("OnlySaveLastPacket");

// Private Panel setting to turn on vcop patchlist dumping
const cmString CAL_ENABLEPATCHDUMP("EnablePatchDump");

// Private Panel setting to set dump file name
const cmString CAL_DUMPFILENAME("DumpFilename");

// Private Panel setting to control level of HW detail dumped
const cmString CAL_DUMPADDITIONALHWINFO("DumpAdditionalHWInfo");


// Private Panel setting to select frames to dump
const cmString CAL_FRAMESTORECORD("FrameStoreCord");

// Private Panel setting to drop all PM4 packets
const cmString CAL_DROPFLUSH("DropFlush");

// Private Panel setting to furce use of dummy QS
const cmString CAL_ENABLEDUMMYQS("DummyQS");

// Private Panel setting to stub post setup
const cmString CAL_STUBPOSTSETUP("StubPostSetup");

// Private Panel setting to stub post TCL
const cmString CAL_STUBPOSTTCL("StubPostTCL");

// Private Panel setting to disable RB3D
const cmString CAL_DISABLERB3D("DisableR3D");

// Private Panel setting to disable alpha blend
const cmString CAL_DISABLEALPHABLEND("DisableAlphaBlend");

// Private Panel setting to force use of tiny textures
const cmString CAL_FORCETINYTEXTURES("ForceTinyTextures");

// Private Panel setting to prevent object allocation in AGP
const cmString CAL_OBJBUFINAGP("OBJBufferInAGP");

// Private Panel setting to prevent object allcoation in local
const cmString CAL_OBJBUFINLOCAL("OBJBufferInLocal");

// Private Panel setting to set the length of the swap queue
const cmString CAL_SWAPQUEUELENGTH("SwapQueueLength");

// Private Panel setting to enable macro tiling for textures
const cmString CAL_ENABLEMACROTILE("MacroTile");

// Private Panel setting to enable micro tiling for textures
const cmString CAL_ENABLEMICROTILE("MicroTile");

// Private Panel setting for allowing early z
const cmString CAL_ALLOWEARLYZ("AllowEarlyZ");

//Private Panel setting to allow for window to be broken into
// multiple pieces(allows full use of C and Z mask on R300 at high res);
const cmString CAL_ALLOWSPLITSCREEN("AllowSplitScreen");

//Private Panel setting for aniso threshold
const cmString CAL_ANISOTHRESHOLD("AnisoThreshold");

//Private Panel setting for aniso bias
const cmString CAL_ANISOLOD("AnisoLod");

//Private Panel setting fpr aniso bias
const cmString CAL_ANISOBIAS("AnisoBias");

//Private Panel setting to control ainos theshold mode
const cmString CAL_ANISOTHRESHMODE("AnisoThreshmode");

// Private Panel Setting for turnning off multi vpu mode(ie render everything to both) for the rest of a frame after a glCopyTexImage or glCopyTexSubImage happen.
const cmString CAL_DISABLEMVPUONCOPYTEX("DisableMVPUOnCopyTexture");

// Private Panel Setting for forcing swap to happen on slave vpu(useful for debugging);
const cmString CAL_FORCEMVPUSWAPONSLAVE("ForceMVPUSwapOnSlave");

// Private Panel Setting for skipping multi-vpu synchronization
const cmString CAL_SKIPMVPUSYNCH("SkipMVPUSynch");

// Private Panel Setting for controlling the percent of screen rendered on the master vpu
const cmString CAL_PERCENTONMASTERMVPU("PercentOnMasterMVPU");

// Private Panel Setting for controlling the mode of mvpu operation
const cmString CAL_MODEMVPU("ModeMVPU");

// Private Panel Setting for drawing a line where the scissored split happened in mvpu mode
const cmString CAL_DRAWSPLITLINEMVPU("DrawSplitLineMVPU");

// Private Panel Setting for controlling whether or not to unroll loops in the GLSL parser
const cmString CAL_UNROLL_LOOPS("UnrollLoops");

// Private Panel Spare setting 1
const cmString CAL_SPARE1("Spare1");

// Private Panel Spare setting 2
const cmString CAL_SPARE2("Spare2");

// Private Panel Spare setting 3
const cmString CAL_SPARE3("Spare3");

// Private Panel Spare setting 4
const cmString CAL_SPARE4("Spare4");

// Private Panel Spare setting 5
const cmString CAL_SPARE5("Spare5");

// Private Panel Spare setting 6
const cmString CAL_SPARE6("Spare6");

// Private Panel Spare setting 7
const cmString CAL_SPARE7("Spare7");

// Private Panel Spare setting 8
const cmString CAL_SPARE8("Spare8");

// Private Panel Spare setting 9
const cmString CAL_SPARE9("Spare9");

// Private Panel Spare setting 10
const cmString CAL_SPARE10("Spare10");

// Private Panel Spare setting 11 - accepts numbers, not just 0 and 1
const cmString CAL_SPARE11("Spare11");

// Private Panel Spare setting 12 - accepts numbers, not just 0 and 1
const cmString CAL_SPARE12("Spare12");

// Private Panel Spare setting 12 - accepts numbers, not just 0 and 1
const cmString CAL_PS3ENABLE("PS3Enable");

// Private Panel setting for asserting when we punt to SW
const cmString CAL_ASSERTONSWPUNT("OrcaAssertOnSWPunt");

// Private Panel setting for logging when we punt to SW
const cmString CAL_LOGSWPUNTCASES("OrcaLogSWPuntCases");

// Private Panel setting to set punt log file name
const cmString CAL_PUNTLOGFILENAME("OrcaPuntLogFileName");

// softVAP mode
const cmString CAL_SOFTVAP("SoftVAP");

// softVAP il compile mode
const cmString CAL_SVPOFFLINECOMPILE("SvpOfflineCompile");

const cmString CAL_EMULATOR("Emulator");

const cmString CAL_ENABLE_FORCE_ASIC_ID("EnableForceAsicID");
const cmString CAL_FORCE_ASIC_ID("ForceAsicID");
const cmString CAL_FORCE_REMOTE_MEMORY("ForceRemoteMemory");
const cmString CAL_DISABLE_ASYNC_DMA("DisableAsyncDma");
const cmString CAL_ENABLE_DUMP_IL("DumpIL");
const cmString CAL_ENABLE_DUMP_ISA("DumpISA");

// TDR
const cmString CAL_ENABLE_FLUSH_AFTER_RENDER("FlushAfterRender");

// VM Disabling
const cmString CAL_DISABLE_VM("DisableVM");

#endif


