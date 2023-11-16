import getopt, sys, os

def write_new_header():
    arg_list = sys.argv[1:] # Files to read is dictated by arguments
    optlist, files_to_read = getopt.getopt(arg_list, "od", ["output=", "deprecated="])

    write_header = ''
    deprecated_functions = ''
    for arg, value in optlist:
        if arg in ["-o", "--output"]:
            write_header = value
        elif arg in ["-p", "--deprecated"]:
            deprecated_functions = value

    print(optlist)
    if len(write_header) == 0:
        print("hip_find_defs.py Command Line argument parsing incorrectly!")
        return

    new_header_file = open(write_header, 'w')

    version_define_map = {}

    struct_mode = False
    struct_string = ''
    struct_name = ''
    struct_depth = 0

    for the_file in files_to_read:
        header = open(the_file, 'r')
        header_lines = header.readlines()
        for line in header_lines:
            #reading a struct
            if struct_mode:
                struct_string += line
                if '{' in line:
                    struct_depth += 1
                elif '}' in line:
                    struct_depth -= 1

                if struct_depth == 0:
                    struct_mode = False
                    if struct_name in version_define_map:
                        #new_header_file.write('\n')
                        #new_header_file.write(getOlderStruct(struct_name, version_define_map[struct_name], hip_device))
                        new_header_file.write(struct_string.replace(struct_name, version_define_map[struct_name]))
                    else:
                        new_header_file.write(struct_string)
                continue

            #finding defines used for versioning
            if "#define" in line:
                line_split = line.split()
                if len(line_split) == 3 and line_split[1] in line_split[2] and line_split[2][-1].isnumeric():
                    version_define_map[line_split[1]] = line_split[2]
                    continue

            #Looking for struct
            if "typedef struct" in line and '{' in line:
                struct_mode = True
                struct_string = line
                struct_depth = 1
                struct_name = line.replace('{', '').split()[-1]
                continue

            #Looking for a typical function signature
            if '(' in line and ')' in line and len(line.split('(')[0].split(' ')) == 2:
                function_name = line.split('(')[0].split(' ')[1]
                #If this function is one of the version functions, write the versioned function too
                if function_name in version_define_map:
                    duplicate_line = line.replace(function_name, version_define_map[function_name])
                    new_header_file.write(duplicate_line)
                    continue
            new_header_file.write(line)
        header.close()

    if os.path.exists(deprecated_functions):
        deprecated_file = open(deprecated_functions, 'r')
        new_header_file.write(deprecated_file.read())

    new_header_file.close()

write_new_header()
