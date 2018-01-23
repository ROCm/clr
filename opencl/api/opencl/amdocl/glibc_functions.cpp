#if defined(__linux__)

#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus

asm (".symver memcpy, memcpy@GLIBC_2.2.5");
void *__wrap_memcpy(void *dest, const void *src, size_t n)
{
    return memcpy(dest, src, n);
}

#if defined(__cplusplus)
}
#endif // __cplusplus)

#endif // __linux__
