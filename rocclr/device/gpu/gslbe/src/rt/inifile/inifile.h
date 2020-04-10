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

#ifndef INIFILE_H
#define INIFILE_H
//
//  Trade secret of ATI Technologies, Inc.
//  Copyright 2005, ATI Technologies, Inc., (unpublished)
//
//  All rights reserved.  This notice is intended as a precaution against
//  inadvertent publication and does not imply publication or any waiver
//  of confidentiality.  The year included in the foregoing notice is the
//  year of creation of the work.
//

/// @file  inifile.h
/// @brief INI File Parser

#include "cm_string.h"
#include "backend.h"

#include <map>
#include <istream>

class IniValue
{
public:
    virtual ~IniValue() {}
    virtual CALboolean getValue(CALboolean* value) { return CAL_FALSE; };
    virtual CALboolean getValue(CALint*     value) { return CAL_FALSE; };
    virtual CALboolean getValue(CALfloat*   value) { return CAL_FALSE; };
    virtual CALboolean getValue(cmString*  value) { return CAL_FALSE; };

#ifdef DEBUG
    virtual void printAST() {};
#endif
private:

};

class IniValueBool : public IniValue
{
public:
    IniValueBool();
    IniValueBool(const IniValueBool& val);
    IniValueBool(CALboolean val);
    IniValueBool& operator=(IniValueBool& v);

    CALboolean getValue(CALboolean* value);

#ifdef DEBUG
    void printAST();
#endif
private:
    CALboolean value;
};

class IniValueString : public IniValue
{
public:
    IniValueString();
    IniValueString(const IniValueString& val);
    IniValueString(cmString val);
    IniValueString& operator=(IniValueString& v);

    CALboolean getValue(cmString* value);

#ifdef DEBUG
    void printAST();
#endif
private:
    cmString value;
};

class IniValueInt : public IniValue
{
public:
    IniValueInt();
    IniValueInt(const IniValueInt& val);
    IniValueInt(CALint val);
    IniValueInt& operator=(IniValueInt& v);

    CALboolean getValue(CALint* value);
    void printAST();
private:
    CALint value;
};

class IniValueFloat : public IniValue
{
public:
    IniValueFloat();
    IniValueFloat(const IniValueFloat& val);
    IniValueFloat(CALfloat val);
    IniValueFloat& operator=(IniValueFloat& v);

    CALboolean getValue(CALfloat* value);

#ifdef DEBUG
    void printAST();
#endif
private:
    CALfloat value;
};



class IniSection
{
public:
    IniSection();
    IniSection(const IniSection& s);
    IniSection(cmString n);
    ~IniSection();

    IniSection& operator=(IniSection& s);

    void addEntry(cmString name, IniValue* value);
    IniValue* findEntry(cmString name);
    const cmString getName();
#ifdef DEBUG
    void printAST();
#endif
private:
    typedef std::map<cmString, IniValue*>  EntryDB;
    typedef EntryDB::const_iterator              EntryDBIterator;
    typedef std::pair<cmString, IniValue*> EntryDBPair;

    cmString name;
    EntryDB entryDB;
};



class IniFile
{
public:
    IniFile(cmString filename);
    IniFile(std::istream& in);
    ~IniFile();

    CALboolean getValue(cmString section, cmString entry, CALboolean* value);
    CALboolean getValue(cmString section, cmString entry, CALint* value);
    CALboolean getValue(cmString section, cmString entry, CALfloat* value);
    CALboolean getValue(cmString section, cmString entry, cmString* value);

    // should be protected
    void addSection(IniSection* section);
    IniSection* findSection(cmString section);
#ifdef DEBUG
    void printAST();
    static void SanityTest();
#endif
private:
    typedef std::map<cmString, IniSection*>  SectionDB;
    typedef SectionDB::const_iterator              SectionDBIterator;
    typedef std::pair<cmString, IniSection*> SectionDBPair;

    IniValue* getValue(cmString section, cmString entry);


    SectionDB sectionDB;
};



#endif
