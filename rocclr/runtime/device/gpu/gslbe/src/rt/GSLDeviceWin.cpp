#include "gsl_ctx.h"
#include "GSLDevice.h"
#include <windows.h>

void CALGSLDevice::closeNativeDisplayHandle()
{
    DeleteDC((HDC)m_nativeDisplayHandle);
    m_nativeDisplayHandle = NULL;
}
