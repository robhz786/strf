#!/bin/sh

if [ -z $CXX ]
then
    echo "Set the CXX envionroment variable with the compiler executable"
    exit 1
fi
   
rm -rf tmp/
mkdir tmp/

BOOST_DIR=$PWD/../../../../
FMT_DIR=$PWD/../fmt-4.0.0

run_benchmark () {
    local flag=$2
    local src=$1.cpp
    local obj=tmp/$1.$flag.obj

    timming=$(\time -f "[ %e s ][ %U s ][ %S s ]" $CXX -std=c++14 -I$BOOST_DIR -I$FMT_DIR $flag -c $src -o $obj 2>&1)
    objsize=$(du -ks $obj | sed 's/\([0-9][0-9]*\).*/\1 K/g' -)
    printf  "[[ %-40s  ][ %3s ]%s[%8s]]\n" "$src" "$flag"  "$timming"   "$objsize"
}

echo
echo -n "[[  source file                              ]"
echo "[flags][real time][usr time][sys time][obj file size]]"
echo

for src in \
    sample1_BoostStringify_header_only \
    sample1_BoostStringify_linked_lib \
    sample1_BoostFormat \
    sample1_fmtlib_header_only \
    sample1_fmtlib \
    sample1_ostream \
    sample1_fprintf \
    sample2_BoostStringify_header_only \
    sample2_BoostStringify_linked_lib \
    sample2_fmtlib_header_only \
    sample2_fmtlib
do
    run_benchmark $src -g
    run_benchmark $src -O0
    run_benchmark $src -O3
done
