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

/// @file  inifile.cpp
/// @brief INI File Parser

#include "inifile.h"
#include "cm_string.h"
#include "inifile_parser.h"
#include "assert.h"

#include <iostream>
#include <istream>
#include <fstream>

#ifdef DEBUG
 #include <sstream>
 #include <string>
#endif

/**
 * IniValueString members
 */

IniValueString::IniValueString()
{
    value = cmString("");
}

IniValueString::IniValueString(const IniValueString& val)
{
    value = val.value;
}

IniValueString::IniValueString(cmString val)
{
    value = val;
}

IniValueString& IniValueString::operator=(IniValueString& v)
{
    value = v.value;
    return *this;
}

CALboolean IniValueString::getValue(cmString* value)
{
    *value = this->value;
    return CAL_TRUE;
}


/**
 * IniValueBool members
 */

IniValueBool::IniValueBool()
{
    value = CAL_FALSE;
}

IniValueBool::IniValueBool(const IniValueBool& val)
{
    value = val.value;
}

IniValueBool::IniValueBool(CALboolean val)
{
    value = val;
}

IniValueBool& IniValueBool::operator=(IniValueBool& v)
{
    value = v.value;
    return *this;
}

CALboolean IniValueBool::getValue(CALboolean* value)
{
    *value = this->value;
    return CAL_TRUE;
}


/**
 * IniValueInt members
 */

IniValueInt::IniValueInt()
{
    value = 0;
}

IniValueInt::IniValueInt(const IniValueInt& val)
{
    value = val.value;
}

IniValueInt::IniValueInt(CALint val)
{
    value = val;
}

IniValueInt& IniValueInt::operator=(IniValueInt& v)
{
    value = v.value;
    return *this;
}

CALboolean IniValueInt::getValue(CALint* value)
{
    *value = this->value;
    return CAL_TRUE;
}

/**
 * IniValueFloat members
 */

IniValueFloat::IniValueFloat()
{
    value = 0;
}

IniValueFloat::IniValueFloat(const IniValueFloat& val)
{
    value = val.value;
}

IniValueFloat::IniValueFloat(CALfloat val)
{
    value = val;
}

IniValueFloat& IniValueFloat::operator=(IniValueFloat& v)
{
    value = v.value;
    return *this;
}

CALboolean IniValueFloat::getValue(CALfloat* value)
{
    *value = this->value;
    return CAL_TRUE;
}

/**
 * IniSection Members
 */
IniSection::IniSection()
{
    name = cmString("");
}

IniSection::IniSection(const IniSection& s)
{
    name = s.name;
    for(EntryDBIterator iter = s.entryDB.begin() ; iter != s.entryDB.end(); iter++)
    {
        entryDB[iter->first] = iter->second;
    }

}

IniSection::IniSection(cmString n)
{
    name = n;
}

IniSection::~IniSection()
{
    for(EntryDBIterator iter = entryDB.begin() ; iter != entryDB.end(); iter++)
    {
        delete iter->second;
    }
    entryDB.clear();
}

IniSection& IniSection::operator=(IniSection& s)
{
    name = s.name;;
    entryDB.clear();
    for(EntryDBIterator iter = s.entryDB.begin() ; iter != s.entryDB.end(); iter++)
    {
        entryDB[iter->first] = iter->second;
    }
    return *this;
}


void IniSection::addEntry(cmString name, IniValue* value)
{
    IniValue* v = findEntry(name);
    if (v)
    {
        delete v;
    }
    entryDB[name] = value;
}

IniValue* IniSection::findEntry(cmString name)
{
    EntryDBIterator iter = entryDB.find(name);
    if(iter != entryDB.end())
    {
        return iter->second;
    }
    else
    {
        return NULL;
    }

}

/**
 * IniFile members
 */

IniFile::IniFile(cmString filename)
{
    #ifdef DEBUG
    SanityTest();
    #endif

    std::ifstream in(filename.c_str());
    IniFileParser::Parse(in, *this);
}

IniFile::IniFile(std::istream& in)
{
    IniFileParser::Parse(in, *this);
}

IniFile::~IniFile()
{
    for(SectionDBIterator iter = sectionDB.begin() ; iter != sectionDB.end(); iter++)
    {
        delete iter->second;
    }
    sectionDB.clear();
}

const cmString IniSection::getName()
{
    return name;
}

void IniFile::addSection(IniSection* section)
{
    IniSection* v = findSection(section->getName());
    if (v)
    {
        delete v;
    }
    sectionDB[section->getName()] = section;
}

IniSection* IniFile::findSection(cmString section)
{
    SectionDBIterator iter = sectionDB.find(section);
    if (iter != sectionDB.end())
    {
        return iter->second;
    }
    else
    {
        return NULL;
    }
}

IniValue* IniFile::getValue(cmString section, cmString entry)
{
    IniSection* s = findSection(section);
    if(s == NULL)
    {
        return NULL;
    }

    return s->findEntry(entry);
}


CALboolean IniFile::getValue(cmString section, cmString entry, CALboolean* value)
{
    IniValue* v = getValue(section, entry);
    if (v != NULL)
    {
        return v->getValue(value);
    }
    return CAL_FALSE;
}

CALboolean IniFile::getValue(cmString section, cmString entry, CALint* value)
{
    IniValue* v = getValue(section, entry);
    if (v != NULL)
    {
        return v->getValue(value);
    }
    return CAL_FALSE;
}

CALboolean IniFile::getValue(cmString section, cmString entry, CALfloat* value)
{
    IniValue* v = getValue(section, entry);
    if (v != NULL)
    {
        return v->getValue(value);
    }
    return CAL_FALSE;
}

CALboolean IniFile::getValue(cmString section, cmString entry, cmString* value)
{
    IniValue* v = getValue(section, entry);
    if (v != NULL)
    {
        return v->getValue(value);
    }
    return CAL_FALSE;
}

/**
 * Debug only methods
 *
 */
#ifdef DEBUG

void IniValueString::printAST()
{
    std::cerr << value.c_str() << " [string]\n";
}

void IniValueBool::printAST()
{
    std::cerr << value << " [bool]\n";
}


void IniValueInt::printAST()
{
    std::cerr << value << " [int]\n";
}

void IniValueFloat::printAST()
{
    std::cerr << value << " [float]\n";
}

void IniSection::printAST()
{
    for(EntryDBIterator iter = entryDB.begin() ; iter != entryDB.end(); iter++)
    {
        cmString name = iter->first;
        IniValue *v = iter->second;

        std::cerr << name.c_str() << " = ";
        v->printAST();
    }
}


void IniFile::printAST()
{
    for(SectionDBIterator iter = sectionDB.begin() ; iter != sectionDB.end(); iter++)
    {
        IniSection* s = iter->second;
        std::cerr << "[" << s->getName().c_str() << "]\n";
        s->printAST();
    }
    std::cerr << "\n";
}

void IniFile::SanityTest()
{
    //std::cerr << "Running IniFile Sanity...\n";

    static const cmString section("section");


    static const std::string file1(
"[section]\n\
bool1=true\n\
bool2=false\n\
int=3\n\
float=1.1111\n\
string=abc def\n");

    std::istringstream s1(file1);
    IniFile* iniFile = new IniFile(s1);
    //iniFile->printAST();

    CALboolean b;

    assert(iniFile->getValue(section, cmString("bool1"), &b) == CAL_TRUE);
    assert(b == CAL_TRUE);

    assert(iniFile->getValue(section, cmString("bool2"), &b) == CAL_TRUE);
    assert(b == CAL_FALSE);

    CALint i;
    assert(iniFile->getValue(section, cmString("int"), &i) == CAL_TRUE);
    assert(i == 3);


    CALfloat f;
    assert(iniFile->getValue(section, cmString("float"), &f) == CAL_TRUE);
    assert(f == 1.1111f);

    cmString s;
    assert(iniFile->getValue(section, cmString("string"), &s) == CAL_TRUE);
    assert(s == cmString("abc def"));

    i = -1;
    // Wrong section
    assert(iniFile->getValue(cmString("dummy"), cmString("int"), &i) == CAL_FALSE);
    assert(i == -1);

    // Wrong entry
    assert(iniFile->getValue(section, cmString("dummy"), &i) == CAL_FALSE);
    assert(i == -1);


    static const std::string file2(
"[section]\n\
bool1=1true\n\
bool2=false2\n\
int=3a\n\
float=1.1111b\n\
string=1\n");

    delete iniFile;

    std::istringstream s2(file2);
    iniFile = new IniFile(s2);
    //iniFile->printAST();

    cmString str;

    b = CAL_FALSE;
    // try to get a bool, then a string
    assert(iniFile->getValue(section, cmString("bool1"), &b) == CAL_FALSE);
    assert(b == CAL_FALSE);
    assert(iniFile->getValue(section, cmString("bool1"), &str) == CAL_TRUE);
    assert(str == cmString("1true"));

    // try to get a bool, then a string
    assert(iniFile->getValue(section, cmString("bool2"), &b) == CAL_FALSE);
    assert(b == CAL_FALSE);
    assert(iniFile->getValue(section, cmString("bool2"), &str) == CAL_TRUE);
    assert(str == cmString("false2"));

    i = -1;
    // try to get an int, then a string
    assert(iniFile->getValue(section, cmString("int"), &i) == CAL_FALSE);
    assert(i == -1);
    assert(iniFile->getValue(section, cmString("int"), &str) == CAL_TRUE);
    assert(str == cmString("3a"));


    f = -1.1f;
    // try to get a float, then a string
    assert(iniFile->getValue(section, cmString("float"), &f) == CAL_FALSE);
    assert(f == -1.1f);
    assert(iniFile->getValue(section, cmString("float"), &str) == CAL_TRUE);
    assert(str == cmString("1.1111b"));

    // try to get a string, value is an int
    assert(iniFile->getValue(section, cmString("string"), &str) == CAL_FALSE);
    assert(str == cmString("1.1111b"));
    assert(iniFile->getValue(section, cmString("string"), &i) == CAL_TRUE);
    assert(i == 1);

    static const cmString section1("section1");
    static const cmString section2("section2");
    static const cmString section3("section3");

    static const std::string file3(
"[section1\n\
bool1=false\n\
bool2=false\n\
int=1\n\
float=1.1\n\
string=abc\n\
[section2]\n\
bool1=true\n\
bool2=true\n\
int=2\n\
float=1.2\n\
string=def\n\
[section3]\n\
int=3\n\
[section2]\n\
float=1.3\n");

    delete iniFile;

    std::istringstream s3(file3);
    iniFile = new IniFile(s3);
    //iniFile->printAST();

    // section1 should not exist (syntax error)
    assert(iniFile->getValue(section1, cmString("bool1"), &str) == CAL_FALSE);
    assert(iniFile->getValue(section1, cmString("bool2"), &str) == CAL_FALSE);
    assert(iniFile->getValue(section1, cmString("int"), &str) == CAL_FALSE);
    assert(iniFile->getValue(section1, cmString("float"), &str) == CAL_FALSE);
    assert(iniFile->getValue(section1, cmString("string"), &str) == CAL_FALSE);

    // section2 should exist, only with the float
    assert(iniFile->getValue(section2, cmString("bool1"), &b) == CAL_FALSE);
    assert(iniFile->getValue(section2, cmString("bool2"), &b) == CAL_FALSE);
    assert(iniFile->getValue(section2, cmString("int"), &i) == CAL_FALSE);

    // overridden
    assert(iniFile->getValue(section2, cmString("float"), &f) == CAL_TRUE);
    assert(f == 1.3f);

    assert(iniFile->getValue(section2, cmString("string"), &str) == CAL_FALSE);

    // section3 had a differant int
    assert(iniFile->getValue(section3, cmString("int"), &i) == CAL_TRUE);
    assert(i == 3);

    delete iniFile;

    //std::cerr << "Done!";
}

#endif
