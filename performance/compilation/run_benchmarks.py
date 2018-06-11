#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
import shutil

cxx = os.environ.get('CXX')
if not cxx or not os.path.isfile(cxx) :
    print("set CXX environment variable to a valid compiler")
    quit(1)

pwd = os.path.dirname(os.path.realpath(__file__))
tmp_dir = pwd + "/tmp"
boost_incl = "-I" + os.path.normpath(pwd + "/../../../../")
fmt_incl = "-I" + os.path.normpath(pwd + "/../fmt-4.0.0")
stringify_cpp = os.path.normpath(pwd + "/../../build/stringify.cpp")
lib_boost_stringify_release = tmp_dir + "/boost_stringify.a"
lib_boost_stringify_debug = tmp_dir + "/boost_stringify.g"
libfmt_release = "libfmt.a"
libfmt_debug  = "libfmt.g"


files_per_program = [1, 21, 41]

def empty_row() :
    part1 = '[[{:^24}][{:^24}][{:^9}]'.format(' ', ' ', ' ', ' ')
    part2 = '[{:9}]'.format('') * (1 + len(files_per_program))
    return part1 + part2 + ']'

def table_header() :
    hdr = '[[{:^24}][{:^24}][{:^9}]'.format(
        'source file',
        'compilation times(w,u,s)',
        'obj. file')        

    for n in files_per_program:
        hdr = hdr + '[{:>3} files]'.format(n)
    return hdr + '[diff per file]]'    

def benchmark_release(basename, main_src, flags, libs):
    benchmark(True, flags, basename, main_src, libs)

def benchmark_debug(basename, main_src, flags, libs):
    benchmark(False, flags, basename, main_src, libs)

def benchmark(release, flags, basename, main_src, libs):
    num_sourcefiles = max(files_per_program)
    compile_stats = create_obj_files(release, flags, basename, num_sourcefiles)
    programs_size = build_programs(release, main_src, basename,
                                   libs, files_per_program)
    result = '[[{:<24}][{:>4.2f} `,  `{:>4.2f} `,  `{:>4.2f}][{:>9.1f}]'.format(
        basename,
        compile_stats['wall time'],
        compile_stats['user time'],
        compile_stats['system time'],
        float(compile_stats['output size']) / 1000.0,
    )
    for s in programs_size:
        result = result + '[{:>9.1f}]'.format(float(s) / 1000.0)
    avr_increment =  float(programs_size[-1] - programs_size[0]) \
        / float(files_per_program[-1] - files_per_program[-0]) \
        / 1000.0
    print(result + '[{:>9.3f}]]'.format(avr_increment))
    
#def benchmark(release, flags, basename, main_src, libs):
#    num_sourcefiles = max(files_per_program)
#    compile_stats = create_obj_files(release, flags, basename, num_sourcefiles)
#    programs_size = build_programs(release, main_src, basename, libs, files_per_program)
#
#    with open('sizes.txt', 'a') as f: 
#        f.write("{:<24}:".format(basename))
#        for size in programs_size :
#            f.write("{:>8}; ".format(size))
#        f.write('\n')
#
#    with open('diffs.txt', 'a') as f: 
#        f.write("{:<24}:".format(basename))
#        for i in range(1, len(programs_size)) :
#            f.write("{:>8}; ".format(programs_size[i] - programs_size[i - 1]))
#        f.write('\n')
#
#    with open('ddiffs.txt', 'a') as f: 
#        f.write("{:<24}:".format(basename))
#        for i in range(2, len(programs_size)) :
#            curr_diff = programs_size[i] - programs_size[i - 1]
#            prev_diff = programs_size[i - 1] - programs_size[i - 2]
#            f.write("{:>8}; ".format(curr_diff - prev_diff))
#        f.write('\n')

    
def create_obj_files(release, flags, basename, num_objs) :
    wtime = 0.0
    utime = 0.0
    stime = 0.0
    osize = 0L
    for i in range(num_objs) :
        obj_id = str(i)
        stat = compile_unit(release, flags, basename, obj_id)
        wtime = wtime + stat['wall time']
        utime = utime + stat['user time']
        stime = stime + stat['system time']
        osize = osize + stat['output size'] 
    return { 'wall time'   : wtime / float(num_objs),
             'user time'   : utime / float(num_objs),
             'system time' : stime / float(num_objs),
             'output size' : osize / num_objs}

def compile_unit(release, flags, basename, obj_id):
    compile_cmd = compile_unit_cmd(release, flags, basename, obj_id)
    cmd = ["time",  "-f", "%e %U %S"] + compile_cmd
    compilation_times = []
    with tempfile.SpooledTemporaryFile() as tmpfile:
        rc = subprocess.call(cmd, stderr=tmpfile)
        tmpfile.seek(0)
        compilation_times = tmpfile.read().strip(' \n').split()
    if rc != 0:
        print("failed to compile " + basename + ".cpp")
        print(" ".join(compile_cmd))
        exit(1)
    obj_size = os.path.getsize(obj_filename(release, basename, obj_id))    
    return { 'wall time'   : float(compilation_times[0]),
             'user time'   : float(compilation_times[1]),
             'system time' : float(compilation_times[2]),
             'output size' : obj_size }

def compile_unit_cmd(release, flags, basename, obj_id) :
    cmd = [ cxx,
            "-O3" if release else "-g",
            "-std=c++14",
            "-DSRC_ID=" + obj_id,
            "-DFUNCTION_NAME=function" + obj_id]
    cmd.extend(flags)
    cmd.extend(["-c",  basename + ".cpp",
                "-o",  obj_filename(release, basename, obj_id)])
    return cmd

def obj_filename(release, basename, obj_id) :
    ext = ".rel.o" if release else ".debug.o"
    return tmp_dir + "/" + basename + "." + obj_id + ext

def build_programs(release, main_src, basename, libs, num_objs_list) :
    programs_size = []
    for num_objs in num_objs_list :
        size = build_program(release, main_src, basename, libs, num_objs)
        programs_size.append(size)
    return programs_size

def build_program(release, main_src, basename, libs, num_objs) :
    write_auxiliary_src_files(num_objs)
    cmd = build_program_command(release, main_src, basename, num_objs, libs)
    exe_name = program_name(basename, num_objs)
    if 0 != subprocess.call(cmd):
        print("failed to build " + exe_name)
        print(" ".join(cmd))
        exit(1)
    return os.path.getsize(exe_name)


def write_auxiliary_src_files(num_objs) :
    header_file = open(tmp_dir + '/functions_declations.hpp', 'w')
    sub_cpp_file = open(tmp_dir + '/functions_calls.cpp', 'w')
    for i in range(num_objs) :
        header_file.write('void function' + str(i) + "(output_type);\n")
        sub_cpp_file.write('function' + str(i) + "(destination);\n")
    header_file.close()
    sub_cpp_file.close()
   
def build_program_command(release, main_src, basename, num_objs, libs) :
    cmd = [cxx]
    cmd.append("-O3" if release else "-g")
    cmd.append("-std=c++14")
    cmd.append(main_src)
#   cmd.append(boost_incl)
    cmd.extend(obj_files(release, basename, num_objs))
    cmd.extend(libs + ["-o", program_name(basename, num_objs)])
    return cmd

def program_name(basename, num_objs):
    return tmp_dir + "/" + basename + '.' + str(num_objs) + '.exe'
              
def obj_files(release, basename, num_objs):
    filenames = []
    for i in range(num_objs) :
        filenames.append(obj_filename(release, basename, str(i)))
    return filenames

def build_lib_stringify(flag, libname):
    print("building " + libname + " ...")
    cmd = [cxx, flag, "-std=c++14", boost_incl, '-c', stringify_cpp,
           '-o', libname]
    if 0 != subprocess.call(cmd):
        print("error building" + libname)
        exit(1)


only_boost_stringify = False

shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir)

build_lib_stringify('-O3', lib_boost_stringify_release)
build_lib_stringify('-g', lib_boost_stringify_debug)

print('\n[table Release mode / linked libraries \n')
print(table_header())

benchmark_release(basename = 'to_string_stringify',
                  main_src = 'to_string_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_release])

if not only_boost_stringify :
    benchmark_release(basename = 'to_string_fmtlib',
                      main_src = 'to_string_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [libfmt_release])
print(empty_row())

benchmark_release(basename = 'to_charptr_stringify',
                  main_src = 'to_charptr_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_release])

if not only_boost_stringify :
    benchmark_release(basename = 'to_charptr_fmtlib',
                      main_src = 'to_charptr_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [libfmt_release])
    benchmark_release(basename = 'to_charptr_sprintf',
                      main_src = 'to_charptr_main.cpp',
                      flags    = [],
                      libs     = [])
print(empty_row())

benchmark_release(basename = 'to_FILE_stringify',
                  main_src = 'to_FILE_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_release])

if not only_boost_stringify :
    benchmark_release(basename = 'to_FILE_fmtlib',
                      main_src = 'to_FILE_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [libfmt_release])
    benchmark_release(basename = 'to_FILE_fprintf',
                      main_src = 'to_FILE_main.cpp',
                      flags    = [],
                      libs     = [])
print(empty_row())

benchmark_release(basename = 'to_ostream_stringify',
                  main_src = 'to_ostream_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_release])
if not only_boost_stringify :
    benchmark_release(basename = 'to_ostream_fmtlib',
                      main_src = 'to_ostream_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [libfmt_release])
    benchmark_release(basename = 'to_ostream_itself',
                      main_src = 'to_ostream_main.cpp',
                      flags    = [],
                      libs     = [])
print(']\n')

print('\n[table Release mode / header-only libraries \n')
print(table_header())
benchmark_release(basename = 'to_string_stringify_ho',
                  main_src = 'to_string_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])
if not only_boost_stringify :
    benchmark_release(basename = 'to_string_fmtlib_ho',
                      main_src = 'to_string_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [])
print(empty_row())

benchmark_release(basename = 'to_charptr_stringify_ho',
                  main_src = 'to_charptr_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])
if not only_boost_stringify :
    benchmark_release(basename = 'to_charptr_fmtlib_ho',
                      main_src = 'to_charptr_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [])
print(empty_row())


benchmark_release(basename = 'to_FILE_stringify_ho',
                  main_src = 'to_FILE_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])

if not only_boost_stringify :
    benchmark_release(basename = 'to_FILE_fmtlib_ho',
                      main_src = 'to_FILE_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [])
print(empty_row())

benchmark_release(basename = 'to_ostream_stringify_ho',
                  main_src = 'to_ostream_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])
if not only_boost_stringify :
    benchmark_release(basename = 'to_ostream_fmtlib_ho',
                      main_src = 'to_ostream_main.cpp',
                      flags    = [fmt_incl],
                      libs     = [])
    benchmark_release(basename = 'to_ostream_BoostFormat',
                      main_src = 'to_ostream_main.cpp',
                      flags    = [boost_incl],
                      libs     = [])

print(']\n')

print('\n[table Debug mode / linked libraries \n')
print(table_header())
benchmark_debug(basename = 'to_string_stringify',
                  main_src = 'to_string_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_debug])
if not only_boost_stringify :
    benchmark_debug(basename = 'to_string_fmtlib',
                    main_src = 'to_string_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [libfmt_debug])
print(empty_row())

benchmark_debug(basename = 'to_charptr_stringify',
                  main_src = 'to_charptr_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_debug])
if not only_boost_stringify :
    benchmark_debug(basename = 'to_charptr_fmtlib',
                    main_src = 'to_charptr_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [libfmt_debug])
    benchmark_debug(basename = 'to_charptr_sprintf',
                    main_src = 'to_charptr_main.cpp',
                    flags    = [],
                    libs     = [])
print(empty_row())

benchmark_debug(basename = 'to_FILE_stringify',
                  main_src = 'to_FILE_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_debug])
if not only_boost_stringify :
    benchmark_debug(basename = 'to_FILE_fmtlib',
                    main_src = 'to_FILE_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [libfmt_debug])
    benchmark_debug(basename = 'to_FILE_fprintf',
                    main_src = 'to_FILE_main.cpp',
                    flags    = [],
                    libs     = [])
print(empty_row())

benchmark_debug(basename = 'to_ostream_stringify',
                  main_src = 'to_ostream_main.cpp',
                  flags    = [boost_incl],
                  libs     = [lib_boost_stringify_debug])

if not only_boost_stringify :
    benchmark_debug(basename = 'to_ostream_fmtlib',
                    main_src = 'to_ostream_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [libfmt_debug])
    benchmark_debug(basename = 'to_ostream_itself',
                    main_src = 'to_ostream_main.cpp',
                    flags    = [],
                    libs     = [])
print(']\n')


print('\n[table Debug mode / header-only libraries \n')
print(table_header())
benchmark_debug(basename = 'to_string_stringify_ho',
                  main_src = 'to_string_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])

if not only_boost_stringify :
    benchmark_debug(basename = 'to_string_fmtlib_ho',
                    main_src = 'to_string_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [])
print(empty_row())

benchmark_debug(basename = 'to_charptr_stringify_ho',
                  main_src = 'to_charptr_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])
if not only_boost_stringify :
    benchmark_debug(basename = 'to_charptr_fmtlib_ho',
                    main_src = 'to_charptr_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [])
print(empty_row())

benchmark_debug(basename = 'to_FILE_stringify_ho',
                  main_src = 'to_FILE_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])

if not only_boost_stringify :
    benchmark_debug(basename = 'to_FILE_fmtlib_ho',
                    main_src = 'to_FILE_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [])
print(empty_row())

benchmark_debug(basename = 'to_ostream_stringify_ho',
                  main_src = 'to_ostream_main.cpp',
                  flags    = [boost_incl],
                  libs     = [])

if not only_boost_stringify :
    benchmark_debug(basename = 'to_ostream_fmtlib_ho',
                    main_src = 'to_ostream_main.cpp',
                    flags    = [fmt_incl],
                    libs     = [])
    benchmark_debug(basename = 'to_ostream_BoostFormat',
                    main_src = 'to_ostream_main.cpp',
                    flags    = [boost_incl],
                    libs     = [])
print(']\n')

shutil.rmtree(tmp_dir, ignore_errors=True)
