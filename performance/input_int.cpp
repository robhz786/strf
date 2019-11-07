#define _CRT_SECURE_NO_WARNINGS

#include <stringify.hpp>
#include "loop_timer.hpp"
#include "fmt/format.h"
#include "fmt/compile.h"
#include <cstdio>
#include <climits>
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
    char dest[1000];
    char* dest_end = dest + sizeof(dest);
    (void) dest_end;
    escape(dest);

    auto dummy_punct = strf::no_grouping<10>().decimal_point(':');

    std::cout << "auto fmt_int = fmt::compile<int>(\"{}\");\n";
    std::cout << "auto fmt_5x_int = fmt::compile<int, int, int, int, int>(\"{}{}{}{}{}\");\n";
    std::cout << "auto fmt_longlong = fmt::compile<long long>(\"{}\");\n";
    std::cout << "auto fmt_5x_longlong = fmt::compile<long long, long long, long long, long long, long long>(\"{}{}{}{}{}\");\n;";
    std::cout << "auto fmt_3_int = fmt::compile<int, int, int>(\"{:*<8}{:#x}{:>+10}\");\n";
    std::cout << "auto fmt_loc_int = fmt::compile<int>(\"{:n}\");\n";
    std::cout << "auto fmt_loc_longlong = fmt::compile<long long>(\"{:n}\");\n";

    std::cout << "\n\n               Small int\n";

    PRINT_BENCHMARK_N(10, "strf::write(dest) (25)")
    {
        (void)strf::write(dest     , dest_end)(20);
        (void)strf::write(dest + 2 , dest_end)(21);
        (void)strf::write(dest + 4 , dest_end)(22);
        (void)strf::write(dest + 6 , dest_end)(23);
        (void)strf::write(dest + 8 , dest_end)(24);
        (void)strf::write(dest + 10, dest_end)(25);
        (void)strf::write(dest + 12, dest_end)(26);
        (void)strf::write(dest + 14, dest_end)(27);
        (void)strf::write(dest + 16, dest_end)(28);
        (void)strf::write(dest + 18, dest_end)(29);
        clobber();
    }
    auto fmt_int = fmt::compile<int>("{}");
    PRINT_BENCHMARK_N(10, "fmt::format_to(dest, fmt_int, 25)")
    {
        fmt::format_to(dest,      fmt_int, 20);
        fmt::format_to(dest +  2, fmt_int, 21);
        fmt::format_to(dest +  4, fmt_int, 22);
        fmt::format_to(dest +  6, fmt_int, 23);
        fmt::format_to(dest +  8, fmt_int, 24);
        fmt::format_to(dest + 10, fmt_int, 25);
        fmt::format_to(dest + 12, fmt_int, 26);
        fmt::format_to(dest + 14, fmt_int, 27);
        fmt::format_to(dest + 16, fmt_int, 28);
        fmt::format_to(dest + 18, fmt_int, 29);
        clobber();
    }
    PRINT_BENCHMARK_N(10, "strcpy(dest, fmt::format_int{25}.c_str())")
    {
        strcpy(dest,      fmt::format_int{20}.c_str());
        strcpy(dest +  2, fmt::format_int{21}.c_str());
        strcpy(dest +  4, fmt::format_int{22}.c_str());
        strcpy(dest +  6, fmt::format_int{23}.c_str());
        strcpy(dest +  8, fmt::format_int{24}.c_str());
        strcpy(dest + 10, fmt::format_int{25}.c_str());
        strcpy(dest + 12, fmt::format_int{26}.c_str());
        strcpy(dest + 14, fmt::format_int{27}.c_str());
        strcpy(dest + 16, fmt::format_int{28}.c_str());
        strcpy(dest + 18, fmt::format_int{29}.c_str());
        clobber();
    }

#if defined(HAS_CHARCONV)

    PRINT_BENCHMARK_N(10, "std::to_chars(dest, dest_end, 25)")
    {
        std::to_chars(dest,      dest_end, 20);
        std::to_chars(dest +  2, dest_end, 21);
        std::to_chars(dest +  4, dest_end, 21);
        std::to_chars(dest +  6, dest_end, 23);
        std::to_chars(dest +  8, dest_end, 24);
        std::to_chars(dest + 10, dest_end, 25);
        std::to_chars(dest + 12, dest_end, 26);
        std::to_chars(dest + 14, dest_end, 27);
        std::to_chars(dest + 16, dest_end, 28);
        std::to_chars(dest + 18, dest_end, 29);
        clobber();
    }

#endif// ! defined(HAS_CHARCONV)

    PRINT_BENCHMARK_N(10, "std::sprintf(dest, \"%d\", 25)")
    {
        std::sprintf(dest, "%d", 20);
        std::sprintf(dest, "%d", 21);
        std::sprintf(dest, "%d", 21);
        std::sprintf(dest, "%d", 23);
        std::sprintf(dest, "%d", 24);
        std::sprintf(dest, "%d", 25);
        std::sprintf(dest, "%d", 26);
        std::sprintf(dest, "%d", 27);
        std::sprintf(dest, "%d", 28);
        std::sprintf(dest, "%d", 29);
        clobber();
    }

    std::cout << "\n";

    PRINT_BENCHMARK_N(10, "strf::write(dest) (15, 25, 35, 45, 55)")
    {
        (void)strf::write(dest     , dest_end)(10, 20, 30, 40, 50);
        (void)strf::write(dest + 2 , dest_end)(11, 21, 31, 41, 51);
        (void)strf::write(dest + 4 , dest_end)(11, 21, 31, 41, 51);
        (void)strf::write(dest + 6 , dest_end)(13, 23, 33, 43, 53);
        (void)strf::write(dest + 8 , dest_end)(14, 24, 34, 44, 54);
        (void)strf::write(dest + 10, dest_end)(15, 25, 35, 45, 55);
        (void)strf::write(dest + 12, dest_end)(16, 26, 36, 46, 56);
        (void)strf::write(dest + 14, dest_end)(17, 27, 37, 47, 57);
        (void)strf::write(dest + 16, dest_end)(18, 28, 38, 48, 58);
        (void)strf::write(dest + 18, dest_end)(19, 29, 39, 49, 59);
        clobber();
    }

    auto fmt_5x_int = fmt::compile<int, int, int, int, int>("{}{}{}{}{}");
    PRINT_BENCHMARK_N(10, "fmt::format_to(dest, fmt_5x_int, 15, 25, 35, 45, 55)")
    {
        fmt::format_to(dest,      fmt_5x_int, 10, 20, 30, 40, 50);
        fmt::format_to(dest +  2, fmt_5x_int, 11, 21, 31, 41, 51);
        fmt::format_to(dest +  4, fmt_5x_int, 11, 21, 31, 41, 51);
        fmt::format_to(dest +  6, fmt_5x_int, 13, 23, 33, 43, 53);
        fmt::format_to(dest +  8, fmt_5x_int, 14, 24, 34, 44, 54);
        fmt::format_to(dest + 10, fmt_5x_int, 15, 25, 35, 45, 55);
        fmt::format_to(dest + 12, fmt_5x_int, 16, 26, 36, 46, 56);
        fmt::format_to(dest + 14, fmt_5x_int, 17, 27, 37, 47, 57);
        fmt::format_to(dest + 16, fmt_5x_int, 18, 28, 38, 48, 58);
        fmt::format_to(dest + 18, fmt_5x_int, 19, 29, 39, 49, 59);
        clobber();
    }

    PRINT_BENCHMARK_N(10, "std::sprintf(dest, \"%d\", 25)")
    {
        std::sprintf(dest, "%d%d%d%d%d", 10, 20, 30, 40, 50);
        std::sprintf(dest, "%d%d%d%d%d", 11, 21, 31, 41, 51);
        std::sprintf(dest, "%d%d%d%d%d", 11, 21, 31, 41, 51);
        std::sprintf(dest, "%d%d%d%d%d", 13, 23, 33, 43, 53);
        std::sprintf(dest, "%d%d%d%d%d", 14, 24, 34, 44, 54);
        std::sprintf(dest, "%d%d%d%d%d", 15, 25, 35, 45, 55);
        std::sprintf(dest, "%d%d%d%d%d", 16, 26, 36, 46, 56);
        std::sprintf(dest, "%d%d%d%d%d", 17, 27, 37, 47, 57);
        std::sprintf(dest, "%d%d%d%d%d", 18, 28, 38, 48, 58);
        std::sprintf(dest, "%d%d%d%d%d", 19, 29, 39, 49, 59);
        clobber();
    }

    std::cout << "\n               Big int\n";

    PRINT_BENCHMARK_N(5, "strf::write(dest) (LLONG_MAX)")
    {
        (void)strf::write(dest     , dest_end)(LLONG_MAX    );
        (void)strf::write(dest + 20, dest_end)(LLONG_MAX - 1);
        (void)strf::write(dest + 40, dest_end)(LLONG_MAX - 2);
        (void)strf::write(dest + 60, dest_end)(LLONG_MAX - 3);
        (void)strf::write(dest + 80, dest_end)(LLONG_MAX - 4);
        clobber();
    }

    auto fmt_longlong = fmt::compile<long long>("{}");

    PRINT_BENCHMARK_N(5, "fmt::format_to(dest, \"{}\", LLONG_MAX)")
    {
        fmt::format_to(dest     , fmt_longlong, LLONG_MAX    );
        fmt::format_to(dest + 20, fmt_longlong, LLONG_MAX - 1);
        fmt::format_to(dest + 40, fmt_longlong, LLONG_MAX - 2);
        fmt::format_to(dest + 60, fmt_longlong, LLONG_MAX - 3);
        fmt::format_to(dest + 80, fmt_longlong, LLONG_MAX - 4);
        clobber();
    }
    PRINT_BENCHMARK_N(5, "strcpy(dest, fmt::format_int{LLONG_MAX}.c_str())")
    {
        strcpy(dest     , fmt::format_int{LLONG_MAX    }.c_str());
        strcpy(dest + 20, fmt::format_int{LLONG_MAX - 1}.c_str());
        strcpy(dest + 40, fmt::format_int{LLONG_MAX - 2}.c_str());
        strcpy(dest + 60, fmt::format_int{LLONG_MAX - 3}.c_str());
        strcpy(dest + 80, fmt::format_int{LLONG_MAX - 4}.c_str());
        clobber();
    }

#if defined(HAS_CHARCONV)

    PRINT_BENCHMARK_N(5, "std::to_chars(dest, dest_end, LLONG_MAX)")
    {
        std::to_chars(dest     , dest_end, LLONG_MAX);
        std::to_chars(dest + 20, dest_end, LLONG_MAX - 1);
        std::to_chars(dest + 40, dest_end, LLONG_MAX - 2);
        std::to_chars(dest + 60, dest_end, LLONG_MAX - 3);
        std::to_chars(dest + 80, dest_end, LLONG_MAX - 4);
        clobber();
    }

#endif // defined(HAS_CHARCONV)

    PRINT_BENCHMARK_N(5, "std::sprintf(dest, \"%lld\", LLONG_MAX)")
    {
        std::sprintf(dest     , "%lld", LLONG_MAX);
        std::sprintf(dest + 20, "%lld", LLONG_MAX - 1);
        std::sprintf(dest + 40, "%lld", LLONG_MAX - 2);
        std::sprintf(dest + 60, "%lld", LLONG_MAX - 3);
        std::sprintf(dest + 80, "%lld", LLONG_MAX - 4);
        clobber();
    }

    std::cout << "\n";
    PRINT_BENCHMARK("strf::write(dest) (LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        (void)strf::write(dest , dest_end)( LLONG_MAX - 10, LLONG_MAX - 20
                                               , LLONG_MAX - 30, LLONG_MAX - 40, LLONG_MAX - 50);
        clobber();
    }
    auto fmt_5x_longlong = fmt::compile<long long, long long, long long, long long, long long>("{}{}{}{}{}");
    PRINT_BENCHMARK("fmt::format_to(dest, fmt_5x_longlong, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        fmt::format_to( dest, fmt_5x_longlong, LLONG_MAX - 10, LLONG_MAX - 20
                      , LLONG_MAX - 30, LLONG_MAX - 40, LLONG_MAX - 50);
        clobber();
    }

    PRINT_BENCHMARK("std::sprintf(dest, \"%lld\", LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX)")
    {
        std::sprintf( dest , "%lld%lld%lld%lld%lld", LLONG_MAX - 10, LLONG_MAX - 20
                    , LLONG_MAX - 30, LLONG_MAX - 40, LLONG_MAX - 50);
        clobber();
    }

    std::cout << "\n               With formatting \n";

    PRINT_BENCHMARK_N(5, "strf::write(dest) (strf::dec(25).fill('*')<8, ~strf::hex(225), +strf::dec(25).p(5)>10)")
    {
        (void)strf::write(dest    , dest_end)(strf::dec(21).fill('*')<8, ' ', ~strf::hex(221), ' ', +strf::dec(21)>10);
        (void)strf::write(dest + 2, dest_end)(strf::dec(22).fill('*')<8, ' ', ~strf::hex(222), ' ', +strf::dec(22)>10);
        (void)strf::write(dest + 4, dest_end)(strf::dec(23).fill('*')<8, ' ', ~strf::hex(223), ' ', +strf::dec(23)>10);
        (void)strf::write(dest + 6, dest_end)(strf::dec(24).fill('*')<8, ' ', ~strf::hex(224), ' ', +strf::dec(24)>10);
        (void)strf::write(dest + 8, dest_end)(strf::dec(25).fill('*')<8, ' ', ~strf::hex(225), ' ', +strf::dec(25)>10);
        clobber();
    }

    auto fmt_3_int = fmt::compile<int, int, int>("{:*<8}{:#x}{:>+10}");
    PRINT_BENCHMARK_N(5, "fmt::format_to(dest, fmt_3_int, 25, 225, 25)")
    {
        fmt::format_to(dest,      fmt_3_int, 21, 221, 21);
        fmt::format_to(dest +  2, fmt_3_int, 22, 222, 22);
        fmt::format_to(dest +  4, fmt_3_int, 23, 223, 23);
        fmt::format_to(dest +  6, fmt_3_int, 24, 224, 24);
        fmt::format_to(dest +  8, fmt_3_int, 25, 225, 25);
        clobber();
    }
    PRINT_BENCHMARK_N(5, "std::sprintf(dest, \"%-8d%#8x%+10.5d\", 25, 225, 25)")
    {
        std::sprintf(dest, "%-8d%#x%+10d", 21, 221, 21);
        std::sprintf(dest, "%-8d%#x%+10d", 22, 222, 22);
        std::sprintf(dest, "%-8d%#x%+10d", 23, 223, 23);
        std::sprintf(dest, "%-8d%#x%+10d", 24, 224, 24);
        std::sprintf(dest, "%-8d%#x%+10d", 25, 225, 25);
        clobber();
    }

    std::cout << "\n               With punctuation\n";
    std::setlocale(LC_ALL, "en_US.UTF-8");
    auto fmt_loc_int = fmt::compile<int>("{:n}");
    strf::monotonic_grouping<10> punct3(3);

    PRINT_BENCHMARK("strf::write(dest).facets(punct3) (25)")
    {
        (void)strf::write(dest).facets(punct3)(25);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmt_loc_int, 25)")
    {
        fmt::format_to(dest, fmt_loc_int, 25);
        clobber();
    }

#if defined(__GNU_LIBRARY__)
    PRINT_BENCHMARK("std::sprintf(dest, \"%'d\", 25)")
    {
        std::sprintf(dest, "%'d", 25);
        clobber();
    }
#else
    std::cout << "\n";
#endif
    std::cout << "\n";

    auto fmt_loc_longlong = fmt::compile<long long>("{:n}");

    PRINT_BENCHMARK("strf::write(dest).facets(punct3) (LLONG_MAX)")
    {
        (void)strf::write(dest).facets(punct3)(LLONG_MAX);
        clobber();
    }
    PRINT_BENCHMARK("fmt::format_to(dest, fmt_loc_longlong, LLONG_MAX)")
    {
        fmt::format_to(dest, fmt_loc_longlong, LLONG_MAX);
        clobber();
    }
#if defined(__GNU_LIBRARY__)
    PRINT_BENCHMARK("std::sprintf(dest, \"%'lld\", LLONG_MAX)")
    {
        std::sprintf(dest, "%'lld", LLONG_MAX);
        clobber();
    }
#else
    std::cout << "\n";
#endif
    std::cout << "\n";

    std::cout << "\n               With punctuation, using a non-ascci character U+22C4\n";
    auto punct3_bigsep = punct3.thousands_sep(0x22C4);

    PRINT_BENCHMARK("strf::write(dest).facets(punct3_bigsep) (25)")
    {
        (void)strf::write(dest).facets(punct3_bigsep)(25);
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest).facets(punct3_bigsep) (LLONG_MAX)")
    {
        (void)strf::write(dest).facets(punct3_bigsep)(LLONG_MAX);
        clobber();
    }

    return 0;
}
