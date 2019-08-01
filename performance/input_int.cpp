#include <boost/stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"
#include <cstdio>
#include <climits>

#if defined(__has_include)
#if __has_include(<charconv>)
#if (__cplusplus >= 201402L) || (defined(_HAS_CXX17) && _HAS_CXX17)
#define HAS_CHARCONV
#include <charconv>
#endif
#endif
#endif

int main()
{
    namespace strf = boost::stringify;
    char dest[1000000];
    char* dest_end = dest + sizeof(dest);
    (void) dest_end;

    auto dummy_punct = strf::no_grouping<10>().decimal_point(':');

    PRINT_BENCHMARK_N(10, "strf::write(dest) (25)")
    {
        (void)strf::write(dest)(20);
        (void)strf::write(dest)(21);
        (void)strf::write(dest)(22);
        (void)strf::write(dest)(23);
        (void)strf::write(dest)(24);
        (void)strf::write(dest)(25);
        (void)strf::write(dest)(26);
        (void)strf::write(dest)(27);
        (void)strf::write(dest)(28);
        (void)strf::write(dest)(29);
    }
    PRINT_BENCHMARK("strf::write(dest) (LONG_MAX)")
    {
        (void)strf::write(dest)(LONG_MAX);
    }
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX)")
    {
        (void)strf::write(dest)(LLONG_MAX);
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest) (+strf::fmt(25))")
    {
        (void)strf::write(dest)(+strf::fmt(25));
    }
    PRINT_BENCHMARK("strf::write(dest) (+strf::fmt(LONG_MAX))")
    {
        (void)strf::write(dest)(+strf::fmt(LONG_MAX));
    }
    PRINT_BENCHMARK("strf::write(dest) (+strf::fmt(LLONG_MAX))")
    {
        (void)strf::write(dest)(+strf::fmt(LLONG_MAX));
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest) (strf::hex(25))")
    {
        (void)strf::write(dest)(strf::hex(25));
    }
    PRINT_BENCHMARK("strf::write(dest) (strf::hex(LONG_MAX))")
    {
        (void)strf::write(dest)(strf::hex(LONG_MAX));
    }
    PRINT_BENCHMARK("strf::write(dest) (strf::hex(LLONG_MAX))")
    {
        (void)strf::write(dest)(strf::hex(LLONG_MAX));
    }

    std::cout << "\n";
    strf::monotonic_grouping<10> punct3(3);

    PRINT_BENCHMARK("strf::write(dest).facets(punct3) (25)")
    {
        (void)strf::write(dest).facets(punct3)(25);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(punct3) (LONG_MAX)")
    {
        (void)strf::write(dest).facets(punct3)(LONG_MAX);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(punct3) (LLONG_MAX)")
    {
        (void)strf::write(dest).facets(punct3)(LLONG_MAX);
    }

    std::cout << "\n";
    auto punct3_bigsep = punct3.thousands_sep(0x22C4);

    PRINT_BENCHMARK("strf::write(dest).facets(punct3_bigsep) (25)")
    {
        (void)strf::write(dest).facets(punct3_bigsep)(25);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(punct3_bigsep) (LONG_MAX)")
    {
        (void)strf::write(dest).facets(punct3_bigsep)(LONG_MAX);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(punct3_bigsep) (LLONG_MAX)")
    {
        (void)strf::write(dest).facets(punct3_bigsep)(LLONG_MAX);
    }
    return 0;
}
