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

    int array_1_to_6[] = {1, 2, 3, 4, 5, 6};

    PRINT_BENCHMARK("strf::write(dest) (strf::range(array_1_to_6))")
    {
        strf::write(dest) (strf::range(array_1_to_6));
        clobber();
    }
    PRINT_BENCHMARK("for(int x : array_1_to_6) ptr = strf::write(ptr, dest_end) (x) .ptr")
    {
        auto ptr = dest;
        for(int x : array_1_to_6) ptr = strf::write(ptr, dest_end) (x) .ptr;
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) (1, 2, 3, 4, 5, 6)")
    {
        strf::write(dest) (1, 2, 3, 4, 5, 6);
        clobber();
    }

    std::cout << "\n               Formatted range\n";
    PRINT_BENCHMARK("strf::write(dest) (+strf::fmt_range(array_1_to_6))")
    {
        strf::write(dest) (strf::range(array_1_to_6));
        clobber();
    }
    PRINT_BENCHMARK("for(int x : array_1_to_6) ptr = strf::write(ptr, dest_end) (+strf::dec(x)) .ptr")
    {
        auto ptr = dest;
        for(int x : array_1_to_6) ptr = strf::write(ptr, dest_end) (+strf::dec(x)) .ptr;
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) (+dec(1), +dec(2), +dec(3), +dec(4), +dec(5), +dec(6))")
    {
        strf::write(dest) ( +strf::dec(1), +strf::dec(2), +strf::dec(3)
                            , +strf::dec(4), +strf::dec(5), +strf::dec(6) );
        clobber();
    }
    std::cout << "\n               Range with separator\n";
    PRINT_BENCHMARK("strf::write(dest) (strf::range(array_1_to_6, \"; \"))")
    {
        strf::write(dest) (strf::range(array_1_to_6, "; "));
        clobber();
    }
    PRINT_BENCHMARK("/* loop by hand */")
    {
        auto ptr = dest;
        int* it = array_1_to_6;
        int* end = it + 6;
        ptr = strf::write(ptr, dest_end) (*it).ptr;
        while (++it != end)
        {
            ptr = strf::write(ptr, dest_end) ("; ", *it).ptr;
        }
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) (1, \"; \", 2, \"; \", 3, \"; \", 4, \"; \", 5, \"; \", 6)")
    {
        strf::write(dest) (1, "; ", 2, "; ", 3, "; ", 4, "; ", 5, "; ", 6);
        clobber();
    }

    std::cout << "\n               Formatted range with separator\n";
    PRINT_BENCHMARK("strf::write(dest) (+strf::fmt_range(array_1_to_6, \"; \"))")
    {
        strf::write(dest) (strf::range(array_1_to_6));
        clobber();
    }
    PRINT_BENCHMARK("/* loop by hand */")
    {
        auto ptr = dest;
        int* it = array_1_to_6;
        int* end = it + 6;
        ptr = strf::write(ptr, dest_end) (+strf::dec(*it)).ptr;
        while (++it != end)
        {
            ptr = strf::write(ptr, dest_end) ("; ", +strf::dec(*it)).ptr;
        }
        clobber();
    }
    PRINT_BENCHMARK("strf::write(dest) (+dec(1), \"; \", +dec(2), \"; \", +dec(3), \"; \", +dec(4), \"; \", +dec(5), \"; \", +dec(6))")
    {
        strf::write(dest) ( +strf::dec(1), "; ", +strf::dec(2), "; ", +strf::dec(3)
                          , +strf::dec(4), "; ", +strf::dec(5), "; ", +strf::dec(6) );
        clobber();
    }

}
