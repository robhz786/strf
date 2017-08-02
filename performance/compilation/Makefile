
BOOST_DIR = ../../../../
FMT_DIR = ../fmt-4.0.0
INCLUDES = -I$(BOOST_DIR) -I$(FMT_DIR)
BENCHMARK_FILE = out/benchmark.tmp
MEASURE_TIMES = \time -o $(BENCHMARK_FILE) -f "[ %e s ][ %U s ][ %S s ]"
GET_COMPILATION_TIMES = $$(cat $(BENCHMARK_FILE))
PRINT_OBJ_SIZE_CMD = du -ks $@ | sed 's/\([0-9][0-9]*\).*/\1 K/g' -
GET_OBJ_FILE_SIZE = $$($(PRINT_OBJ_SIZE_CMD))
PRINTF_CMD = printf "[[ %-40s  ][ %4s ]%s[%8s]]\n"

out/debug/%.o: %.cpp phony_target
	@mkdir -p out/debug
	@$(MEASURE_TIMES) $(CXX) -std=c++14 -g $(INCLUDES) -c $< -o $@
	@$(PRINTF_CMD)   "$<"  "-g"  "$(GET_COMPILATION_TIMES)"  "$(GET_OBJ_FILE_SIZE)"


out/O0/%.o: %.cpp phony_target
	@mkdir -p $(@D)
	@$(MEASURE_TIMES) $(CXX) -std=c++14 -O0 $(INCLUDES) -c $< -o $@
	@$(PRINTF_CMD)   "$<"  "-O0"  "$(GET_COMPILATION_TIMES)"  "$(GET_OBJ_FILE_SIZE)"


out/O3/%.o: %.cpp phony_target
	@mkdir -p $(@D)
	@$(MEASURE_TIMES) $(CXX) -std=c++14 -O3 $(INCLUDES) -c $< -o $@
	@$(PRINTF_CMD)   "$<"  "-O3"  "$(GET_COMPILATION_TIMES)"  "$(GET_OBJ_FILE_SIZE)"


sources := \
sample1_BoostStringify.cpp \
sample1_BoostFormat.cpp \
sample1_fmtlib_header_only.cpp \
sample1_fmtlib.cpp \
sample1_ostream.cpp \
sample1_fprintf.cpp \
\
sample2_BoostStringify.cpp \
sample2_fmtlib_header_only.cpp \
sample2_fmtlib.cpp 


default_target: print_table_header $(foreach bsrc,$(basename $(sources)),out/debug/$(bsrc).o out/O0/$(bsrc).o out/O3/$(bsrc).o)

clean:
	rm -rf out

.PHONY: phony_target print_table_header

phony_target:
	@echo

print_table_header:
	@echo
	@echo -n "[[  source file                              ]"
	@echo "[flags][real time][usr time][sys time][obj file size]]"


