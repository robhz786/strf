#include <boost/stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"
#include <cstdio>

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

    PRINT_BENCHMARK("strf::write(dest)(0.111)")
    {
        (void) strf::write(dest)(0.111);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(dummy_punct)(0.111)")
    {
        (void) strf::write(dest).facets(dummy_punct)(0.111);
    }
    PRINT_BENCHMARK("strf::write(dest).tr(\"{}\", 0.111)")
    {
        (void) strf::write(dest).tr("{}", 0.111);
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(0.111))")
    {
        (void) strf::write(dest)(strf::fmt(0.111));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(0.111))")
    {
        (void) strf::write(dest)(strf::sci(0.111));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(0.111))")
    {
        (void) strf::write(dest)(strf::fixed(0.111));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(0.111).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(0.111).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(0.111).p(6))")
    {
        (void) strf::write(dest)(strf::sci(0.111).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(0.111).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(0.111).p(6));
    }
    PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", 0.111)")
    {
        fmt::format_to(dest, "{}", 0.111);
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%g\" 0.111)")
    {
        std::sprintf(dest, "%g", 0.111);
    }


    strf::write(stdout)("\n ---- 6.103515625e-05 ---- \n");

    PRINT_BENCHMARK("strf::write(dest)(6.103515625e-05)")
    {
        (void) strf::write(dest)(6.103515625e-05);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(dummy_punct)(6.103515625e-05)")
    {
        (void) strf::write(dest).facets(dummy_punct)(6.103515625e-05);
    }
    PRINT_BENCHMARK("strf::write(dest).tr(\"{}\", 6.103515625e-05)")
    {
        (void) strf::write(dest).tr("{}", 6.103515625e-05);
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(6.103515625e-05))")
    {
        (void) strf::write(dest)(strf::fmt(6.103515625e-05));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(6.103515625e-05))")
    {
        (void) strf::write(dest)(strf::sci(6.103515625e-05));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(6.103515625e-05))")
    {
        (void) strf::write(dest)(strf::fixed(6.103515625e-05));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(6.103515625e-05).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(6.103515625e-05).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(6.103515625e-05).p(6))")
    {
        (void) strf::write(dest)(strf::sci(6.103515625e-05).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(6.103515625e-05).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(6.103515625e-05).p(6));
    }
    // PRINT_BENCHMARK("fmt::format_to(dest, \"{}\", 6.103515625e-05)")
    // {
    //     fmt::format_to(dest, "{}", 6.103515625e-05);
    // }
    // PRINT_BENCHMARK("std::sprintf(dest, \"%g\" 6.103515625e-05)")
    // {
    //     std::sprintf(dest, "%g", 6.103515625e-05);
    // }

    strf::write(stdout)("\n ---- 1234567890 ---- \n");

    PRINT_BENCHMARK("strf::write(dest)(1234567890.0)")
    {
        (void) strf::write(dest)(1234567890.0);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(dummy_punct)(1234567890.0)")
    {
        (void) strf::write(dest).facets(dummy_punct)(1234567890.0);
    }
    PRINT_BENCHMARK("strf::write(dest).tr(\"{}\", 1234567890.0)")
    {
        (void) strf::write(dest).tr("{}", 1234567890.0);
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(1234567890.0))")
    {
        (void) strf::write(dest)(strf::fmt(1234567890.0));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(1234567890.0))")
    {
        (void) strf::write(dest)(strf::sci(1234567890.0));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(1234567890.0))")
    {
        (void) strf::write(dest)(strf::fixed(1234567890.0));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(1234567890.0).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(1234567890.0).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(1234567890.0).p(6))")
    {
        (void) strf::write(dest)(strf::sci(1234567890.0).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(1234567890.0).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(1234567890.0).p(6));
    }

    strf::write(stdout)("\n ---- 1234567.12890625 ---- \n");

    PRINT_BENCHMARK("strf::write(dest)(1234567.12890625)")
    {
        (void) strf::write(dest)(1234567.12890625);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(dummy_punc)(1234567.12890625)")
    {
        (void) strf::write(dest).facets(dummy_punct)(1234567.12890625);
    }
    PRINT_BENCHMARK("strf::write(dest).tr(\"{}\", 1234567.12890625)")
    {
        (void) strf::write(dest).tr("{}", 1234567.12890625);
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(1234567.12890625))")
    {
        (void) strf::write(dest)(strf::fmt(1234567.12890625));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(1234567.12890625))")
    {
        (void) strf::write(dest)(strf::sci(1234567.12890625));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(1234567.12890625))")
    {
        (void) strf::write(dest)(strf::fixed(1234567.12890625));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(1234567.12890625).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(1234567.12890625).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(1234567.12890625).p(6))")
    {
        (void) strf::write(dest)(strf::sci(1234567.12890625).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(1234567.12890625).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(1234567.12890625).p(6));
    }

    strf::write(stdout)("\n ---- 12.625 ---- \n");

    PRINT_BENCHMARK("strf::write(dest)(12.625)")
    {
        (void) strf::write(dest)(12.625);
    }
    PRINT_BENCHMARK("strf::write(dest).facets(dummy_punct)(12.625)")
    {
        (void) strf::write(dest).facets(dummy_punct)(12.625);
    }
    PRINT_BENCHMARK("strf::write(dest).tr(\"{}\", 12.625)")
    {
        (void) strf::write(dest).tr("{}", 12.625);
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(12.625))")
    {
        (void) strf::write(dest)(strf::fmt(12.625));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(12.625))")
    {
        (void) strf::write(dest)(strf::sci(12.625));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(12.625))")
    {
        (void) strf::write(dest)(strf::fixed(12.625));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(12.625).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(12.625).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::sci(12.625).p(6))")
    {
        (void) strf::write(dest)(strf::sci(12.625).p(6));
    }
    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(12.625).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(12.625).p(6));
    }

    return 0;
}
