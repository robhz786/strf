//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "loop_timer.hpp"

int main()
{
    char dest[1000000];
    char* dest_end = dest + sizeof(dest);
    (void) dest_end;
    escape(dest);

    PRINT_BENCHMARK("strf::to(dest) (strf::join('a', 'b', 'c', 'd'))")
    {
        strf::to(dest) (strf::join('a', 'b', 'c', 'd'));
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) ('a', 'b', 'c', 'd')")
    {
        strf::to(dest) ('a', 'b', 'c', 'd');
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) (strf::join('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))")
    {
        strf::to(dest) (strf::join('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'));
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')")
    {
        strf::to(dest) ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h');
        clobber();
    }
    std::cout << "\n";
    PRINT_BENCHMARK("strf::to(dest) (strf::join_right(15)('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))")
    {
        strf::to(dest) (strf::join_right(15)('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'));
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) (strf::join_split(15, 2)('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))")
    {
        strf::to(dest) (strf::join_split(15, 2)('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'));
        clobber();
    }

    std::cout << "\n";

    PRINT_BENCHMARK("strf::to(dest) (strf::join_right(15)(\"Hello World\"))")
    {
        strf::to(dest) (strf::join_right(15)("Hello World"));
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) (strf::join(\"Hello World\") > 15)")
    {
        strf::to(dest) (strf::join("Hello World") > 15);
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) (strf::fmt(\"Hello World\") > 15)")
    {
        strf::to(dest) (strf::fmt("Hello World") > 15);
        clobber();
    }
    std::cout << "\n";

    PRINT_BENCHMARK("strf::to(dest) (strf::join_right(4)(25))")
    {
        strf::to(dest) (strf::join_right(4)(25));
        clobber();
    }
    PRINT_BENCHMARK("strf::to(dest) (strf::dec(25) > 4)")
    {
        strf::to(dest) (strf::dec(25) > 4);
        clobber();
    }
    std::cout << "\n";

    return 1;
}
