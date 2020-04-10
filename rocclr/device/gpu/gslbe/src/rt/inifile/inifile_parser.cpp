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

//
//  Trade secret of ATI Technologies, Inc.
//  Copyright 2005, ATI Technologies, Inc., (unpublished)
//
//  All rights reserved.  This notice is intended as a precaution against
//  inadvertent publication and does not imply publication or any waiver
//  of confidentiality.  The year included in the foregoing notice is the
//  year of creation of the work.
//

/// @file  inifile_parser.cpp
/// @brief INI File Parser Implementation

#include "inifile.h"
#include "inifile_parser.h"

#include "cm_string.h"


#include <cctype>
#include <string>
#include <istream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>


void IniFileParser::Parse(std::istream& in, IniFile& iniFile)
{
    CALuint count = 0;
    std::string line;
    bool inSection = false;

    std::string sectionName;

    IniSection* section = NULL;
    while(std::getline(in, line)) {
        count++;
        cleanup(line);
        if(line.empty())
        {
            continue;
        }
        if(parseSectionName(line, sectionName))
        {
            section = new IniSection(cmString(sectionName.c_str()));
            iniFile.addSection(section);
            inSection = true;
        }
        else if(inSection)
        {
            parseLine(line, section, count);
        }
    }
}

void IniFileParser::parseLine( std::string line, IniSection* section, CALuint count ) {
    std::string::size_type equals = line.find( '=' );
    if ( equals == std::string::npos ) {
#ifdef DEBUG
        std::cerr << "IniFileParser: Could not parse line " << count << ", ignoring.\n";
#endif
        return;
    }

    std::string name( line, 0, equals );
    IniValue* value = parseValue( std::string( line, equals + 1, std::string::npos));

    section->addEntry(cmString(trim(name).c_str()), value);
}

void IniFileParser::cleanup( std::string& line ) {
    std::string copy = line;
    unsigned int begin = 0;
    while ( begin != line.size() && isspace(line[begin]))
    {
        ++begin;
    }

    bool inQuote = false;
    unsigned int end;
    for(end = begin; end != line.size(); ++end)
    {
        if ( line[end] == '\"' )
        {
            inQuote = !inQuote;
        }
        // comments starts with # or ;
        else if ( (line[end] == '#' || line[end] == ';') && !inQuote )
        {
            break;
        }
        else if ( line[ end ] == '\\' )
        {
            ++end; // ignore next character
            if ( end == line.size() ) {
#ifdef DEBUG
                std::cerr << "INIFileParser: Error parsing file: \\ character "
                    "at the end of line (sorry, not supported)\n";
#endif
                break;
            }
        }
    }
    while ( end > begin && isspace( line[ end - 1 ] ) ) --end;
    // This is used over assign so that we don't have memcpy overrun
    // errors in valgrind.
    line = line.substr(begin, end - begin);
}

class isint
{
public:
    isint()
    {
        is_int = true;
    }
    void operator() (char c)
    {
        is_int = is_int && isdigit(c);
    }
    bool is_int;
};

class isfloat
{
public:
    isfloat()
    {
        is_float = true;
    }
    void operator() (char c)
    {
        is_float = is_float && (isdigit(c) || c == '.');
    }
    bool is_float;
};

int cmp_nocase(const std::string s1, const std::string s2)
{
    std::string::const_iterator p1 = s1.begin();
    std::string::const_iterator p2 = s2.begin();

    while( p1 != s1.end() && p2 != s2.end())
    {
        if(toupper(*p1) != toupper(*p2))
        {
            return (toupper(*p1) < toupper(*p2)) ? -1 : 1;
        }
        ++p1;
        ++p2;
    }
    return static_cast<int>(s2.size()-s1.size());
}

IniValue* IniFileParser::parseValue(std::string value ) {
    std::string trimmed = trim(value);

    std::stringstream ss(trimmed);

    // look for a boolean
    static const std::string strTrue("true");
    static const std::string strFalse("false");
    if(cmp_nocase(trimmed, strTrue) == 0)
    {
        return new IniValueBool(CAL_TRUE);
    }
    if(cmp_nocase(trimmed, strFalse) == 0)
    {
        return new IniValueBool(CAL_FALSE);
    }

    // try now to get an int
    isint ii;
    ii = std::for_each(trimmed.begin(),trimmed.end(), ii);
    if(ii.is_int)
    {
        CALint intValue = 0;
        ss >> intValue;
        return new IniValueInt(intValue);
    }

    // if not an int, try to get a float
    isfloat isf;
    isf = std::for_each(trimmed.begin(),trimmed.end(), isf);
    if(isf.is_float)
    {
        CALfloat floatValue;
        // mbeuchat: Remove STL conversion of string to float. When compiled
        // on Linux, DK g++ with optimization requires linking against
        // libstdc++-6.0.9 which is not available on all Linux systems.
        // ss >> floatValue;
        floatValue = (float)atof(ss.str().c_str());
        return new IniValueFloat(floatValue);
    }

    // finally, default to a string
    return new IniValueString(cmString(trimmed.c_str()));
}

bool IniFileParser::parseSectionName(std::string line, std::string& section )
{
    if ( line[ 0 ] != '[' ) return false;
    if ( line[ line.size() - 1 ] != ']' ) return false;

    section.assign( line, 1, line.size() - 2 );
    return true;
}


std::string IniFileParser::trim(std::string const& source, char const* delims) {
  std::string result(source);
  std::string::size_type index = result.find_last_not_of(delims);
  if(index != std::string::npos)
    result.erase(++index);

  index = result.find_first_not_of(delims);
  if(index != std::string::npos)
    result.erase(0, index);
  else
    result.erase();
  return result;
}
