//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

#define TEST testf<__LINE__>

int main()
{
    namespace strf = boost::stringify;

    TEST("abcdef   123") () ({strf::join = 12, {"abc", "def", 1, 23}});
    TEST("abcdef123   ") () ({strf::join < 12, {"abc", "def", 123}});
    TEST("   abcdef123") () ({strf::join > 12, {"abc", "def", 123}});
    TEST("~~~abcdef123") () ({strf::join('~') > 12, {"abc", "def", 123}});
    TEST("~~~abcdef123") (strf::fill(U'~')) ({strf::join > 12, {"abc", "def", 123}});
    TEST("abcdef123") () ({strf::join = 9, {"abc", "def", 1, 23}});
    TEST("abcdef123") () ({strf::join < 9, {"abc", "def", 123}});
    TEST("abcdef123") () ({strf::join > 9, {"abc", "def", 123}});

    
    return  boost::report_errors();
}
