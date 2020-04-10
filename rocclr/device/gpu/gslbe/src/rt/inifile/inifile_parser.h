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

#ifndef INIFILE_PARSER_H
#define INIFILE_PARSER_H
//
//  Trade secret of ATI Technologies, Inc.
//  Copyright 2005, ATI Technologies, Inc., (unpublished)
//
//  All rights reserved.  This notice is intended as a precaution against
//  inadvertent publication and does not imply publication or any waiver
//  of confidentiality.  The year included in the foregoing notice is the
//  year of creation of the work.
//

/// @file  inifile_parser.h
/// @brief INI File Parser Implementation

// if compiled from OGTST, add the following, normally defined in atitypes.h

#include "inifile.h"
#include "cm_string.h"

#include <istream>
#include <iostream>
#include <string>

class IniFileParser
{
public:
    static void Parse(std::istream& in, IniFile& iniFile);

private:
    static void parseLine( std::string line, IniSection* section, CALuint count );
    static bool parseSectionName(std::string line, std::string& section );
    static IniValue* parseValue(std::string value );
    static void cleanup( std::string& line );
    static std::string trim(std::string const& source, char const* delims = " \t\r\n");
};

#endif



