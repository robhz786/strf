#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <stringify.hpp>
#include "loop_timer.hpp"
#define FMT_USE_GRISU 1
#include "fmt/compile.h"
#include "fmt/format.h"
#include <cstdio>
#include <cmath>
#include <clocale>

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
    char dest[1000000];
    char* dest_end = dest + sizeof(dest);
    (void) dest_end;
    auto dummy_punct = strf::no_grouping<10>().decimal_point(':');
    strf::monotonic_grouping<10> punct3(3);
    std::setlocale(LC_ALL, "en_US.UTF-8");
    escape(dest);

#if defined(FMT_USE_GRISU) && FMT_USE_GRISU == 1
    std::cout <<
        "FMT_USE_GRISU == 1\n";
#endif
    std::cout <<
        "auto fmtd = fmt::compile<double>(\"{}\");\n"
        "auto fmtg = fmt::compile<double>(\"{:g}\");\n"
        "auto fmte = fmt::compile<double>(\"{:e}\");\n"
        "auto fmtf = fmt::compile<double>(\"{:f}\");\n"
        "auto fmt_loc = fmt::compile<double>(\"{:n}\");\n";
        // "auto punct3 = strf::monotonic_grouping<10>(3);\n"
        // "auto dummy_punct = strf::no_grouping<10>().decimal_point(':');\n";

    auto fmtd = fmt::compile<double>("{}");
    auto fmtg = fmt::compile<double>("{:g}");
    auto fmte = fmt::compile<double>("{:e}");
    auto fmtf = fmt::compile<double>("{:f}");
    auto fmt_loc = fmt::compile<double>("{:n}");

    std::cout << "\n\n               Without any formatting\n";

    PRINT_BENCHMARK("strf::write(dest)(M_PI)")
    {
        (void) strf::write(dest)(M_PI);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtd, M_PI)")
    {
        (void) fmt::format_to(dest, fmtd, M_PI);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(3.33)")
    {
        (void) strf::write(dest)(3.33);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtd, 3.33)")
    {
        (void) fmt::format_to(dest, fmtd, 3.33);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(3.1234567+125)")
    {
        (void) strf::write(dest)(3.1234567+125);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtd, 3.1234567+125)")
    {
        (void) fmt::format_to(dest, fmtd, 3.1234567+125);
        clobber();
    }

    std::cout << "\n               General notation with precision = 6\n";


    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(M_PI).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(M_PI).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtg, M_PI)")
    {
        (void) fmt::format_to(dest, fmtg, M_PI);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%g\", M_PI)")
    {
        (void) std::sprintf(dest, "%g", M_PI);
        clobber();
    }


    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(3.33).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(3.33).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtg, 3.33)")
    {
        (void) fmt::format_to(dest, fmtg, 3.33);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%g\", 3.33)")
    {
        (void) std::sprintf(dest, "%g", 3.33);
        clobber();
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::fmt(3.1234567e+125).p(6))")
    {
        (void) strf::write(dest)(strf::fmt(3.1234567+125).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtg, 3.1234567e+125)")
    {
        (void) fmt::format_to(dest, fmtg, 3.1234567+125);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%g\", 3.1234567e+125)")
    {
        (void) std::sprintf(dest, "%g", 3.1234567e+125);
        clobber();
    }

    std::cout << "\n               Fixed notation with precision = 6\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(M_PI).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(M_PI).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtf, M_PI)")
    {
        (void) fmt::format_to(dest, fmtf, M_PI);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%f\", M_PI)")
    {
        (void) std::sprintf(dest, "%f", M_PI);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(3.33).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(3.33).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtf, 3.33)")
    {
        (void) fmt::format_to(dest, fmtf, 3.33);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%f\", 3.33)")
    {
        (void) std::sprintf(dest, "%f", 3.33);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::fixed(123456.789).p(6))")
    {
        (void) strf::write(dest)(strf::fixed(123456.789).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmtf, 123456.789)")
    {
        (void) fmt::format_to(dest, fmtf, 123456.789);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%f\", 123456.789)")
    {
        (void) std::sprintf(dest, "%f", 123456.789);
        clobber();
    }

    std::cout << "\n               Scientific notation with precision = 6\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::sci(M_PI).p(6))")
    {
        (void) strf::write(dest)(strf::sci(M_PI).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmte, M_PI)")
    {
        (void) fmt::format_to(dest, fmte, M_PI);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%e\", M_PI)")
    {
        (void) std::sprintf(dest, "%e", M_PI);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)(strf::sci(3.33).p(6))")
    {
        (void) strf::write(dest)(strf::sci(3.33).p(6));
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmte, 3.33)")
    {
        (void) fmt::format_to(dest, fmte, 3.33);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%e\", 3.33)")
    {
        (void) std::sprintf(dest, "%e", 3.33);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::write(dest)((strf::sci(3.1234567+125).p(6))")
    {
        (void) strf::write(dest)(3.1234567+125);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmte, 3.1234567+125)")
    {
        (void) fmt::format_to(dest, fmte, 3.1234567+125);
        clobber();
    }
    PRINT_BENCHMARK("std::sprintf(dest, \"%e\", 3.1234567+125)")
    {
        (void) std::sprintf(dest, "%e", 3.1234567+125);
        clobber();
    }

    std::cout << "\n               With punctuation\n";

    PRINT_BENCHMARK("strf::write(dest).facets(strf::monotonic_grouping<10>(3))(1000000.0)")
    {
        (void) strf::write(dest).facets(strf::monotonic_grouping<10>(3))(1000000.0);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmt_loc, 1000000.0)")
    {
        (void) fmt::format_to(dest, fmt_loc, 1000000.0);
        clobber();
    }

    std::cout << "\n";
    PRINT_BENCHMARK("strf::write(dest).facets(strf::no_grouping<10>().decimal_point(':'))(1000000.0)")
    {
        (void) strf::write(dest).facets(strf::no_grouping<10>().decimal_point(':'))(1000000.0);
        clobber();
    }


    // strf::write(stdout)("\n ---- 6.103515625e-05 ---- \n");
    // strf::write(stdout)("\n ---- 1234567890 ---- \n");
    // strf::write(stdout)("\n ---- 1234567.12890625 ---- \n");

    return 0;
}
