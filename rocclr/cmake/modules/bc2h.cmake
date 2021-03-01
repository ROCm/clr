# Copyright (C) 2017-2021 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bc2h.c
"#include <stdio.h>\n"
"int main(int argc, char **argv){\n"
"    FILE *ifp, *ofp;\n"
"    int c, i, l;\n"
"    if (argc != 4) return 1;\n"
"    ifp = fopen(argv[1], \"rb\");\n"
"    if (!ifp) return 1;\n"
"    i = fseek(ifp, 0, SEEK_END);\n"
"    if (i < 0) return 1;\n"
"    l = ftell(ifp);\n"
"    if (l < 0) return 1;\n"
"    i = fseek(ifp, 0, SEEK_SET);\n"
"    if (i < 0) return 1;\n"
"    ofp = fopen(argv[2], \"wb+\");\n"
"    if (!ofp) return 1;\n"
"    fprintf(ofp, \"#define %s_size %d\\n\\n\"\n"
"                 \"#if defined __GNUC__\\n\"\n"
"                 \"__attribute__((aligned (4096)))\\n\"\n"
"                 \"#elif defined _MSC_VER\\n\"\n"
"                 \"__declspec(align(4096))\\n\"\n"
"                 \"#endif\\n\"\n"
"                 \"static const unsigned char %s[%s_size+1] = {\",\n"
"                 argv[3], l,\n"
"                 argv[3], argv[3]);\n"
"    i = 0;\n"
"    while ((c = getc(ifp)) != EOF) {\n"
"        if (0 == (i&7)) fprintf(ofp, \"\\n   \");\n"
"        fprintf(ofp, \" 0x%02x,\", c);\n"
"        ++i;\n"
"    }\n"
"    fprintf(ofp, \" 0x00\\n};\\n\\n\");\n"
"    fclose(ifp);\n"
"    fclose(ofp);\n"
"    return 0;\n"
"}\n"
)

add_executable(bc2h ${CMAKE_CURRENT_BINARY_DIR}/bc2h.c)
