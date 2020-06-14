#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
import shutil
import string

cxx = os.environ.get('CXX')
if cxx is None or not os.path.isfile(cxx) :
    print("set CXX environment variable to a valid compiler")
    quit(1)

cxx_standard='17'

cxxflags = ['-std=c++' + cxx_standard]
cxxflags_str = os.environ.get('CXXFLAGS')
if cxxflags_str is not None:
    cxxflags = cxxflags_str.split()
    idx = string.find(cxxflags_str, '-std=')
    if idx == -1:
        cxxflags.append('-std=c++' + cxx_standard)
    else:
        cxx_standard = cxxflags_str[idx + 8: idx + 10]
del cxxflags_str


ldflags = []
ldflags_str = os.environ.get('LDFLAGS')
if ldflags_str is not None:
    ldflags = ldflags_str.split()
del ldflags_str


pwd = os.path.dirname(os.path.realpath(__file__))
tmp_dir = pwd + "/tmp"
strf_incl = "-I" + os.path.normpath(pwd + "/../../include")
fmt_dir = os.path.normpath(pwd + "/../../externals/fmt")
fmt_incl = "-I" + fmt_dir + "/include"
strf_cpp = os.path.normpath(pwd + "/../../src/strf.cpp")

files_per_program = [1, 21, 31, 41]

#def clean_abort(msg):
#    print(msg)
#    shutil.rmtree(tmp_dir, ignore_errors=True)
#    return(1)

def empty_row() :
    part1 = '|{:^21}|{:^24} |{:^9} '.format(' ', ' ', ' ', ' ')
    part2 = '|{:9} '.format('') * (1 + len(files_per_program))
    return part1 + part2

def table_header() :
    hdr = '|{:^21}|{:^18} '.format(
        'source file',
        'comp. times(w,u,s)' )

    for n in files_per_program:
        hdr = hdr + '|{:>3} files '.format(n)
    return hdr + '| diff'

def print_table_header(title):
    print('\n[caption=]')
    print(title)
#    print('[cols=\"<22m,^18m,>10m,>10m,>10m,>15m\"]\n|===')
    print('|===')
#    print(table_header())

def benchmark(build_type, flags, basename, main_src, libs):
    num_sourcefiles = max(files_per_program)
    compile_stats = create_obj_files(build_type, flags, basename, num_sourcefiles)
    programs_size = build_programs(build_type, main_src, basename,
                                   libs, files_per_program)
    result = '|{:<21}|{:>4.2f} , {:>4.2f} , {:>4.2f} '.format(
        '{' + basename + '}',
        compile_stats['wall time'],
        compile_stats['user time'],
        compile_stats['system time'],
        #float(compile_stats['output size']) / 1000.0,
    )
    for s in programs_size:
        result = result + '|{:>9.1f} '.format(float(s) / 1000.0)
    diff =  float(programs_size[-1] - programs_size[-2]) / 1000.0
    print(result + '|{:>9.1f}'.format(diff))

def create_obj_files(build_type, flags, basename, num_objs) :
    wtime = 0.0
    utime = 0.0
    stime = 0.0
    osize = 0L
    for i in range(num_objs) :
        obj_id = str(i)
#       print(compile_unit_cmd(build_type, flags, basename, obj_id))
        stat = compile_unit(build_type, flags, basename, obj_id)
        wtime = wtime + stat['wall time']
        utime = utime + stat['user time']
        stime = stime + stat['system time']
        osize = osize + stat['output size']
    return { 'wall time'   : wtime / float(num_objs),
             'user time'   : utime / float(num_objs),
             'system time' : stime / float(num_objs),
             'output size' : osize / num_objs}

def compile_unit(build_type, flags, basename, obj_id):
    compile_cmd = compile_unit_cmd(build_type, flags, basename, obj_id)
    cmd = ["time",  "-f", "%e %U %S"] + compile_cmd
    compilation_times = []
    with tempfile.SpooledTemporaryFile() as tmpfile:
        rc = subprocess.call(cmd, stderr=tmpfile)
        tmpfile.seek(0)
        compilation_times = tmpfile.read().strip(' \n').split()
    if rc != 0:
        print("failed to compile " + basename + ".cpp")
        print(" ".join(compile_cmd))
        return(1)
    obj_size = os.path.getsize(obj_filename(build_type, basename, obj_id))
    return { 'wall time'   : float(compilation_times[0]),
             'user time'   : float(compilation_times[1]),
             'system time' : float(compilation_times[2]),
             'output size' : obj_size }

def compile_unit_cmd(build_type, flags, basename, obj_id) :
    cmd = [ cxx,
            build_type_get_cflag(build_type),
            "-DSRC_ID=" + obj_id,
            "-DFUNCTION_NAME=function" + obj_id]
    cmd.extend(flags)
    cmd.extend(cxxflags)
    cmd.extend(["-c",  basename + ".cpp",
                "-o",  obj_filename(build_type, basename, obj_id)])
    return cmd

def obj_filename(build_type, basename, obj_id) :
    ext = "." + build_type_get_strid(build_type) + ".o"
    return tmp_dir + "/" + basename + "." + obj_id + ext


def build_programs(build_type, main_src, basename, libs, num_objs_list) :
    programs_size = []
    for num_objs in num_objs_list :
        size = build_program(build_type, main_src, basename, libs, num_objs)
        programs_size.append(size)
    return programs_size

def build_program(build_type, main_src, basename, libs, num_objs) :
    write_auxiliary_src_files(num_objs)
    cmd = build_program_command(build_type, main_src, basename, num_objs, libs)
    exe_name = program_name(basename, num_objs)
    if 0 != subprocess.call(cmd):
        print("failed to build " + exe_name)
        print(" ".join(cmd))
        return(1)
    return os.path.getsize(exe_name)


def write_auxiliary_src_files(num_objs) :
    header_file = open(tmp_dir + '/functions_declations.hpp', 'w')
    sub_cpp_file = open(tmp_dir + '/functions_calls.cpp', 'w')
    for i in range(num_objs) :
        header_file.write('void function' + str(i) + "(output_type);\n")
        sub_cpp_file.write('function' + str(i) + "(destination);\n")
    header_file.close()
    sub_cpp_file.close()

def build_program_command(build_type, main_src, basename, num_objs, libs) :
    cmd = [cxx]
    cmd.append(build_type_get_cflag(build_type))
    cmd.append(main_src)
    cmd.extend(cxxflags)
    cmd.extend(ldflags)
    cmd.extend(obj_files(build_type, basename, num_objs))
    cmd.extend(libs + ["-o", program_name(basename, num_objs)])
    return cmd

def program_name(basename, num_objs):
    return tmp_dir + "/" + basename + '.' + str(num_objs) + '.exe'

def obj_files(build_type, basename, num_objs):
    filenames = []
    for i in range(num_objs) :
        filenames.append(obj_filename(build_type, basename, str(i)))
    return filenames

def cmake_generate_strf(buildtype):
    build_dir = tmp_dir + "/strf-" + buildtype
    src_dir = os.path.normpath(pwd + "/../..")
    os.makedirs(build_dir)
    gen_args = ["cmake", "-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=" + buildtype,
                "-DCMAKE_CXX_STANDARD=" + cxx_standard, src_dir]
    gen_p = subprocess.Popen(gen_args, cwd=build_dir)
    gen_p.wait()
    if gen_p.returncode != 0:
        print("Failed generation CMake build project for Strf")
        return 1
    return build_dir

def cmake_generate_fmt(buildtype):
    build_dir = tmp_dir + "/fmt-" + buildtype
    os.makedirs(build_dir)
    gen_args = ["cmake", "-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=" + buildtype,
            "-DFMT_DOC=OFF", "-DFMT_TEST=OFF", "-DCMAKE_CXX_STANDARD=" + cxx_standard, fmt_dir]
    gen_p = subprocess.Popen(gen_args, cwd=build_dir)
    gen_p.wait()
    if gen_p.returncode != 0:
        print("Failed generation CMake build project for {fmt}")
        return 1
    return build_dir

def cmake_build(build_dir):
    build_args = ["cmake", "--build", "."]
    build_p = subprocess.Popen(build_args, cwd=build_dir)
    build_p.wait()
    if build_p.returncode != 0:
         print("Failed to build")
         return 1

def build_libstrf(buildtype):
    print("building Strf " + buildtype)
    build_dir = cmake_generate_strf(buildtype)
    cmake_build(build_dir)
    return build_dir + "/" + "libstrf.a"

def build_libfmt(buildtype):
    print("building {fmt} " + buildtype)
    build_dir = cmake_generate_fmt(buildtype)
    cmake_build(build_dir)
    #libname = "libfmtd.a" if buildtype == build_type_debug() else "libfmt.a"
    return build_dir + "/" + "libfmt.a"

def build_type_O3():
    return "O3"
def build_type_Os():
    return "Os"
def build_type_debug():
    return "g"

def build_type_get_cflag(build_type):
    return "-" + build_type

def build_type_get_strid(build_type):
    return build_type

def benchmark_list(build_type, samples):
    for s in samples:
        prefix = s[0]
        main_src = prefix + "_main.cpp"
        for sub_s in s[1]:
            lib_list = sub_s[0]
            flags = sub_s[1]
            basename = prefix + sub_s[2];
            benchmark(build_type, flags, basename, main_src, lib_list)

shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir)

lib_strf_Os = build_libstrf(build_type_Os())
lib_strf_O3 = build_libstrf(build_type_O3())
lib_strf_db = build_libstrf(build_type_debug())
#
libfmt_Os = build_libfmt(build_type_Os())
libfmt_O3 = build_libfmt(build_type_O3())
libfmt_db = build_libfmt(build_type_debug())

strf_static_lib = [strf_incl, "-DSTRF_SEPARATE_COMPILATION"]
strf_ho         = [strf_incl]
fmt_static_lib  = [fmt_incl]
fmt_ho          = [fmt_incl, "-DFMT_HEADER_ONLY=1"]

samples_O3 = \
[ ('to_charptr',        [ ([lib_strf_O3], strf_static_lib, '_strf'   )
                        , ([lib_strf_O3], strf_static_lib, '_strf_tr')
                        , ([libfmt_O3],   fmt_static_lib,  '_fmtlib' )
                        , ([],            [],              '_sprintf')
] )
, ('to_string',         [ ([lib_strf_O3], strf_static_lib, '_strf'   )
                        , ([lib_strf_O3], strf_static_lib, '_strf_tr')
                        , ([libfmt_O3],   fmt_static_lib,  '_fmtlib' )
] )
, ('to_FILE',           [ ([lib_strf_O3], strf_static_lib, '_strf'   )
                        , ([lib_strf_O3], strf_static_lib, '_strf_tr')
                        , ([libfmt_O3],   fmt_static_lib,  '_fmtlib' )
                        , ([],            [],              '_fprintf')
] )
, ('to_ostream',        [ ([lib_strf_O3], strf_static_lib, '_strf'   )
                        , ([lib_strf_O3], strf_static_lib, '_strf_tr')
                        , ([libfmt_O3],   fmt_static_lib,  '_fmtlib' )
] )
]

samples_Os = \
[ ('to_charptr',        [ ([lib_strf_Os], strf_static_lib, '_strf'   )
                        , ([lib_strf_Os], strf_static_lib, '_strf_tr')
                        , ([libfmt_Os],   fmt_static_lib,  '_fmtlib' )
                        , ([],            [],              '_sprintf')
] )
, ('to_string',         [ ([lib_strf_Os], strf_static_lib, '_strf'   )
                        , ([lib_strf_Os], strf_static_lib, '_strf_tr')
                        , ([libfmt_Os],   fmt_static_lib,  '_fmtlib' )
] )
, ('to_FILE',           [ ([lib_strf_Os], strf_static_lib, '_strf'   )
                        , ([lib_strf_Os], strf_static_lib, '_strf_tr')
                        , ([libfmt_Os],   fmt_static_lib,  '_fmtlib' )
                        , ([],            [],              '_fprintf')
] )
, ('to_ostream',        [ ([lib_strf_Os], strf_static_lib, '_strf'   )
                        , ([lib_strf_Os], strf_static_lib, '_strf_tr')
                        , ([libfmt_Os],   fmt_static_lib,  '_fmtlib' )
] )
]

samples_debug = \
[ ('to_charptr',        [ ([lib_strf_db], strf_static_lib, '_strf'   )
                        , ([lib_strf_db], strf_static_lib, '_strf_tr')
                        , ([libfmt_db],   fmt_static_lib,  '_fmtlib' )
                        , ([],            [],              '_sprintf')
] )
, ('to_string',         [ ([lib_strf_db], strf_static_lib, '_strf'   )
                        , ([lib_strf_db], strf_static_lib, '_strf_tr')
                        , ([libfmt_db],   fmt_static_lib,  '_fmtlib')
] )
, ('to_FILE',           [ ([lib_strf_db], strf_static_lib, '_strf'   )
                        , ([lib_strf_db], strf_static_lib, '_strf_tr')
                        , ([libfmt_db],   fmt_static_lib,  '_fmtlib' )
                        , ([],            [],              '_fprintf')
] )
, ('to_ostream',        [ ([lib_strf_db], strf_static_lib, '_strf'   )
                        , ([lib_strf_db], strf_static_lib, '_strf_tr')
                        , ([libfmt_db],   fmt_static_lib,  '_fmtlib' )
] ) ]

samples_O3_header_only = \
[ ('to_charptr',        [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_string',         [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_FILE',           [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_ostream',        [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] ) ]

samples_Os_header_only = \
[ ('to_charptr',        [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib' )
] )
, ('to_string',         [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib' )
] )
, ('to_FILE',           [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_ostream',        [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
]
samples_debug_header_only = \
[ ('to_charptr',        [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_string',         [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_FILE',           [ ([], strf_ho, '_strf'   )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib')
] )
, ('to_ostream',        [ ([], strf_ho, '_strf'    )
                        , ([], strf_ho, '_strf_tr')
                        , ([], fmt_ho,  '_fmtlib' )
] )
]

print(table_header())

print_table_header('.Release mode with -Os flag / linked libraries')
benchmark_list(build_type_Os(), samples_Os)
print('|===\n')

print_table_header('.Release mode with -Os flag / header only libraries')
benchmark_list(build_type_Os(), samples_Os_header_only)
print('|===\n')

print_table_header('.Release mode with -O3 flag / linked libraries')
benchmark_list(build_type_O3(), samples_O3)
print('|===\n')

print_table_header('.Release mode with -O3 flag / header only libraries')
benchmark_list(build_type_O3(), samples_O3_header_only)
print('|===\n')

print_table_header('.Debug mode / linked libraries')
benchmark_list(build_type_debug(), samples_debug)
print('|===\n')

print_table_header('.Debug mode / header only libraries')
benchmark_list(build_type_debug(), samples_debug_header_only)
print('|===\n')

shutil.rmtree(tmp_dir, ignore_errors=True)
