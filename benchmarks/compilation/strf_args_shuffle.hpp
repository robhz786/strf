#ifndef STRF_ARGS_SHUFFLE_HPP
#define STRF_ARGS_SHUFFLE_HPP

#include <strf.hpp>

template <int>
struct arg_idx{};

inline auto get_arg(arg_idx<0>){ return "abc"; }
inline auto get_arg(arg_idx<1>){ return ' '; }
inline auto get_arg(arg_idx<2>){ return (const void*)0; }
inline auto get_arg(arg_idx<3>){ return 1234; }
inline auto get_arg(arg_idx<4>){ return 1234u; }
inline auto get_arg(arg_idx<5>){ return 123.4; }
inline auto get_arg(arg_idx<6>){ return strf::sci(123.4); }
inline auto get_arg(arg_idx<7>){ return strf::fixed(123.4); }
inline auto get_arg(arg_idx<8>){ return strf::gen(123.4); }
inline auto get_arg(arg_idx<9>){ return strf::hex(1234); }
inline auto get_arg(arg_idx<10>){ return strf::hex(1234) > 10; }
inline auto get_arg(arg_idx<11>){ return +strf::dec(1234) < 10; }
inline auto get_arg(arg_idx<12>){ return strf::hex(1234u); }
inline auto get_arg(arg_idx<13>){ return strf::hex(1234u) > 10; }
inline auto get_arg(arg_idx<14>){ return +strf::dec(1234u) < 10; }
inline auto get_arg(arg_idx<15>){ return *strf::sci(123.4) > 10; }
inline auto get_arg(arg_idx<16>){ return *strf::fixed(123.4) > 10; }
inline auto get_arg(arg_idx<17>){ return *strf::gen(123.4) > 10; }
inline auto get_arg(arg_idx<18>){ return strf::right("qwert", 10); }
inline auto get_arg(arg_idx<19>){ return true; }

constexpr int args_types_count = 20;

template <int I>
auto get_arg() { return get_arg(arg_idx<I % args_types_count>()); }

#ifndef SRC_ID
#define SRC_ID 0
#endif

#define ARG(I) get_arg<((SRC_ID) + (I))>()

#endif // define STRF_ARGS_SHUFFLE_HPP
