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
#include "cal.h"

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



