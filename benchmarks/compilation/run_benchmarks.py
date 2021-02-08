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


cxxflags_str = os.environ.get('CXXFLAGS')
cxx_standard='20'
cxxflags = ['-std=c++2a']
if cxxflags_str is not None:
    cxxflags = cxxflags_str.split()
    idx = string.find(cxxflags_str, '-std=')
    if idx == -1:
        cxxflags.append('-std=c++2a')
    else:
        cxx_standard = cxxflags_str[idx + 8: idx + 10]
        if cxx_standard == '2a':
            cxx_standard == '20'
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

def empty_row() :
    part1 = '|{:^24}|{:^24} |{:^9} '.format(' ', ' ', ' ', ' ')
    part2 = '|{:9} '.format('') * (1 + len(files_per_program))
    return part1 + part2

def table_header() :
    hdr = '|{:^24}|{:^18} '.format(
        'source file',
        'comp. times(w|u|s)' )

    for n in files_per_program:
        hdr = hdr + '|{:>3} files '.format(n)
    return hdr + '| diff'

def print_table_header(title):
    print(title)
    print('|===')
    print(table_header())

def benchmark(build_type, flags, basename, main_src, libs):
    num_sourcefiles = max(files_per_program)
    compile_stats = create_obj_files(build_type, flags, basename, num_sourcefiles)
    programs_size = build_programs(build_type, main_src, basename,
                                   libs, files_per_program)
    result = '|{:<24}|{:>4.2f} | {:>4.2f} | {:>4.2f} '.format(
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

def build_type_get_cflag(build_type):
    return "-" + build_type

def build_type_get_strid(build_type):
    return build_type

shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir)

lib_strf_Os = build_libstrf("O3")
lib_strf_O3 = build_libstrf("Os")
lib_strf_g  = build_libstrf("g")
#
libfmt_Os = build_libfmt("O3")
libfmt_O3 = build_libfmt("Os")
libfmt_g  = build_libfmt("g")

strf_static_lib = [strf_incl, "-DSTRF_SEPARATE_COMPILATION"]
strf_ho         = [strf_incl]
fmt_static_lib  = [fmt_incl]
fmt_ho          = [fmt_incl, "-DFMT_HEADER_ONLY=1"]

def benchmark_Os_header_only_strf(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [strf_incl]
    benchmark("Os", [strf_incl], basename, main_src, [])

def benchmark_Os_static_link_strf(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [strf_incl,  "-DSTRF_SEPARATE_COMPILATION"]
    benchmark("Os", flags, basename, main_src, [lib_strf_Os])

def benchmark_Os_header_only_fmtlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [fmt_incl, "-DFMT_HEADER_ONLY=1"]
    benchmark("Os", flags, basename, main_src, [])

def benchmark_Os_static_link_fmtlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    benchmark("Os", [fmt_incl], basename, main_src, [libfmt_Os])

def benchmark_Os_stdlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    benchmark("Os", [], basename, main_src, [])

def benchmark_O3_header_only_strf(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [strf_incl]
    benchmark("O3", [strf_incl], basename, main_src, [])

def benchmark_O3_static_link_strf(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [strf_incl,  "-DSTRF_SEPARATE_COMPILATION"]
    benchmark("O3", flags, basename, main_src, [lib_strf_O3])

def benchmark_O3_header_only_fmtlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [fmt_incl, "-DFMT_HEADER_ONLY=1"]
    benchmark("O3", flags, basename, main_src, [])

def benchmark_O3_static_link_fmtlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    benchmark("O3", [fmt_incl], basename, main_src, [libfmt_O3])

def benchmark_O3_stdlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    benchmark("O3", [], basename, main_src, [])

def benchmark_g_header_only_strf(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [strf_incl]
    benchmark("g", [strf_incl], basename, main_src, [])

def benchmark_g_static_link_strf(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [strf_incl,  "-DSTRF_SEPARATE_COMPILATION"]
    benchmark("g", flags, basename, main_src, [lib_strf_g])

def benchmark_g_header_only_fmtlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    flags = [fmt_incl, "-DFMT_HEADER_ONLY=1"]
    benchmark("g", flags, basename, main_src, [])

def benchmark_g_static_link_fmtlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    benchmark("g", [fmt_incl], basename, main_src, [libfmt_g])

def benchmark_g_stdlib(prefix, suffix):
    main_src = prefix + "_main.cpp"
    basename = prefix + suffix
    benchmark("g", [], basename, main_src, [])

print_table_header('==== Static libs, with `-Os`')
benchmark_Os_static_link_strf   ('to_charptr', '_strf')
benchmark_Os_static_link_strf   ('to_charptr', '_strf_tr')
benchmark_Os_static_link_fmtlib ('to_charptr', '_fmtlib_n_c')
benchmark_Os_static_link_fmtlib ('to_charptr', '_fmtlib_n')
benchmark_Os_static_link_fmtlib ('to_charptr', '_fmtlib_c')
benchmark_Os_static_link_fmtlib ('to_charptr', '_fmtlib')
benchmark_Os_stdlib             ('to_charptr', '_sprintf')

benchmark_Os_static_link_strf   ('to_string', '_strf')
benchmark_Os_static_link_strf   ('to_string', '_strf_tr')
benchmark_Os_static_link_fmtlib ('to_string', '_fmtlib_c')
benchmark_Os_static_link_fmtlib ('to_string', '_fmtlib')

benchmark_Os_static_link_strf   ('to_FILE', '_strf')
benchmark_Os_static_link_strf   ('to_FILE', '_strf_tr')
benchmark_Os_static_link_fmtlib ('to_FILE', '_fmtlib')
benchmark_Os_stdlib             ('to_FILE', '_fprintf')

benchmark_Os_static_link_strf   ('to_ostream', '_strf')
benchmark_Os_static_link_strf   ('to_ostream', '_strf_tr')
benchmark_Os_static_link_fmtlib ('to_ostream', '_fmtlib')

print('|===\n')

print_table_header('==== Header-only libs, with `-Os`')
benchmark_Os_header_only_strf  ('to_charptr', '_strf')
benchmark_Os_header_only_strf  ('to_charptr', '_strf_tr')
benchmark_Os_header_only_fmtlib('to_charptr', '_fmtlib_n_c')
benchmark_Os_header_only_fmtlib('to_charptr', '_fmtlib_n')
benchmark_Os_header_only_fmtlib('to_charptr', '_fmtlib_c')
benchmark_Os_header_only_fmtlib('to_charptr', '_fmtlib')

benchmark_Os_header_only_strf   ('to_string', '_strf')
benchmark_Os_header_only_strf   ('to_string', '_strf_tr')
benchmark_Os_header_only_fmtlib ('to_string', '_fmtlib_c')
benchmark_Os_header_only_fmtlib ('to_string', '_fmtlib')

benchmark_Os_header_only_strf   ('to_FILE', '_strf')
benchmark_Os_header_only_strf   ('to_FILE', '_strf_tr')
benchmark_Os_header_only_fmtlib ('to_FILE', '_fmtlib')

benchmark_Os_header_only_strf   ('to_ostream', '_strf')
benchmark_Os_header_only_strf   ('to_ostream', '_strf_tr')
benchmark_Os_header_only_fmtlib ('to_ostream', '_fmtlib')

print('|===\n')

print_table_header('==== Static libs, with `-O3`')
benchmark_O3_static_link_strf   ('to_charptr', '_strf')
benchmark_O3_static_link_strf   ('to_charptr', '_strf_tr')
benchmark_O3_static_link_fmtlib ('to_charptr', '_fmtlib_n_c')
benchmark_O3_static_link_fmtlib ('to_charptr', '_fmtlib_n')
benchmark_O3_static_link_fmtlib ('to_charptr', '_fmtlib_c')
benchmark_O3_static_link_fmtlib ('to_charptr', '_fmtlib')
benchmark_O3_stdlib             ('to_charptr', '_sprintf')

benchmark_O3_static_link_strf   ('to_string', '_strf')
benchmark_O3_static_link_strf   ('to_string', '_strf_tr')
benchmark_O3_static_link_fmtlib ('to_string', '_fmtlib_c')
benchmark_O3_static_link_fmtlib ('to_string', '_fmtlib')

benchmark_O3_static_link_strf   ('to_FILE', '_strf')
benchmark_O3_static_link_strf   ('to_FILE', '_strf_tr')
benchmark_O3_static_link_fmtlib ('to_FILE', '_fmtlib')
benchmark_O3_stdlib             ('to_FILE', '_fprintf')

benchmark_O3_static_link_strf   ('to_ostream', '_strf')
benchmark_O3_static_link_strf   ('to_ostream', '_strf_tr')
benchmark_O3_static_link_fmtlib ('to_ostream', '_fmtlib')

print('|===\n')

print_table_header('==== Header-only libs, with `-O3`')
benchmark_O3_header_only_strf  ('to_charptr', '_strf')
benchmark_O3_header_only_strf  ('to_charptr', '_strf_tr')
benchmark_O3_header_only_fmtlib('to_charptr', '_fmtlib_n_c')
benchmark_O3_header_only_fmtlib('to_charptr', '_fmtlib_n')
benchmark_O3_header_only_fmtlib('to_charptr', '_fmtlib_c')
benchmark_O3_header_only_fmtlib('to_charptr', '_fmtlib')

benchmark_O3_header_only_strf   ('to_string', '_strf')
benchmark_O3_header_only_strf   ('to_string', '_strf_tr')
benchmark_O3_header_only_fmtlib ('to_string', '_fmtlib_c')
benchmark_O3_header_only_fmtlib ('to_string', '_fmtlib')

benchmark_O3_header_only_strf   ('to_FILE', '_strf')
benchmark_O3_header_only_strf   ('to_FILE', '_strf_tr')
benchmark_O3_header_only_fmtlib ('to_FILE', '_fmtlib')

benchmark_O3_header_only_strf   ('to_ostream', '_strf')
benchmark_O3_header_only_strf   ('to_ostream', '_strf_tr')
benchmark_O3_header_only_fmtlib ('to_ostream', '_fmtlib')

print('|===\n')

print_table_header('==== Static libs, with `-g`')
benchmark_g_static_link_strf   ('to_charptr', '_strf')
benchmark_g_static_link_strf   ('to_charptr', '_strf_tr')
benchmark_g_static_link_fmtlib ('to_charptr', '_fmtlib_n_c')
benchmark_g_static_link_fmtlib ('to_charptr', '_fmtlib_n')
benchmark_g_static_link_fmtlib ('to_charptr', '_fmtlib_c')
benchmark_g_static_link_fmtlib ('to_charptr', '_fmtlib')
benchmark_g_stdlib             ('to_charptr', '_sprintf')

benchmark_g_static_link_strf   ('to_string', '_strf')
benchmark_g_static_link_strf   ('to_string', '_strf_tr')
benchmark_g_static_link_fmtlib ('to_string', '_fmtlib_c')
benchmark_g_static_link_fmtlib ('to_string', '_fmtlib')

benchmark_g_static_link_strf   ('to_FILE', '_strf')
benchmark_g_static_link_strf   ('to_FILE', '_strf_tr')
benchmark_g_static_link_fmtlib ('to_FILE', '_fmtlib')
benchmark_g_stdlib             ('to_FILE', '_fprintf')

benchmark_g_static_link_strf   ('to_ostream', '_strf')
benchmark_g_static_link_strf   ('to_ostream', '_strf_tr')
benchmark_g_static_link_fmtlib ('to_ostream', '_fmtlib')

print('|===\n')

print_table_header('==== Header-only libs, with `-g`')
benchmark_g_header_only_strf  ('to_charptr', '_strf')
benchmark_g_header_only_strf  ('to_charptr', '_strf_tr')
benchmark_g_header_only_fmtlib('to_charptr', '_fmtlib_n_c')
benchmark_g_header_only_fmtlib('to_charptr', '_fmtlib_n')
benchmark_g_header_only_fmtlib('to_charptr', '_fmtlib_c')
benchmark_g_header_only_fmtlib('to_charptr', '_fmtlib')

benchmark_g_header_only_strf   ('to_string', '_strf')
benchmark_g_header_only_strf   ('to_string', '_strf_tr')
benchmark_g_header_only_fmtlib ('to_string', '_fmtlib_c')
benchmark_g_header_only_fmtlib ('to_string', '_fmtlib')

benchmark_g_header_only_strf   ('to_FILE', '_strf')
benchmark_g_header_only_strf   ('to_FILE', '_strf_tr')
benchmark_g_header_only_fmtlib ('to_FILE', '_fmtlib')

benchmark_g_header_only_strf   ('to_ostream', '_strf')
benchmark_g_header_only_strf   ('to_ostream', '_strf_tr')
benchmark_g_header_only_fmtlib ('to_ostream', '_fmtlib')

print('|===\n')

shutil.rmtree(tmp_dir, ignore_errors=True)
