#!/usr/bin/python

# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#This script updates the copyright end year to the current year for all files in the repository.

import re
from datetime import date
import fileinput
import sys
import os
import argparse

#====================================================================

#modifies the end year of the copyright line
def modify_line(line, new_end_yr):
    try:
        constructed_years = ""
        matched = re.findall(r'(.*)Copyright(.*)Advanced', line) #returns list of one string
        years = matched[0][1].replace(" ", "") #removes spaces for easier regex
        if len(years) == 0: #if no year or text between Copyright and Advanced then if portion is selected
            constructed_years = "Copyright (c) "+ "2015" + " - " + new_end_yr + " "
            formatted_line = line.replace("Copyright", constructed_years)
        else:
            if any(char.isdigit() for char in years):
                years = re.findall('([0-9]+)(-*)(.*)', years) # 3 strings - start year, - , end year
                start_yr = years[0][0]  #using the same start year
            else:
                start_yr = "2015"    #if no start year mentioned. then use 2015
            constructed_years = " (c) "+ start_yr + " - " + new_end_yr + " "
            formatted_line = line.replace(matched[0][1], constructed_years)
    except:
        formatted_line = line #return the same line incase of exceptions
    return formatted_line

#updates the copyrights year for the given file
def update_copyright_year(file_name, new_end_yr):
    for line in fileinput.input(file_name,inplace = True):
        if "Copyright" in line and "Advanced Micro Devices" in line:
            line = modify_line(line, new_end_yr)
        sys.stdout.write(line)

#gets all the file names recursively for the given folder name
def get_files_names(folder_name):
    folders = ["build", "docs", ".git"] #folders to exclude
    exc_folders = [folder_name + '/' + x for x in folders]  #folders to exclude full path
    exc_files = [".out",".md"] #files to exclude
    filenames = []
    for dirpath, dirs, files in os.walk(folder_name):
        for filename in files:
            #excluding files from some folders and excluding some built files
            if not any(exc in dirpath for exc in exc_folders) and not any(exc_file in filename for exc_file in exc_files):
                filenames.append(os.path.join(dirpath,filename))
    return filenames

#====================================================================


#Main function
parser=argparse.ArgumentParser(
    description=''' copyright_year.py script updates the end year of the copyrights

                    python copyright_year.py $HIP_DIR 2021
                    First argument takes the repo name,
                    Second argument takes the end year
                ''')
parser.add_argument('repo_name', help='Repo name, example: $OPENCL_DIR')
parser.add_argument('end_year',  help='end year , example: 2021')
args=parser.parse_args()
year = args.end_year
repo = args.repo_name
print "Selected Repo: ", repo
print "Selected Year: ", year

if len(year) != 4 or not year.isdigit():
    print "Invalid year passed: ", year
    print "Please enter 4 digit year. exiting...."
    sys.exit()

files =  get_files_names(repo)
for file_name in files:
    print "filename:", file_name
    update_copyright_year(file_name, year)
