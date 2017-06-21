//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

#define TEST testf<__LINE__>

int main()
{
    namespace strf = boost::stringify::v0;

    TEST("abcdef   123") ({strf::join_internal(12, 3), {"abc", "de", "f", 123}});
    TEST("abcdef123   ") ({strf::join_left(12), {"abc", "def", 123}});
    TEST("   abcdef123") ({strf::join_right(12), {"abc", "def", 123}});
    TEST("~~~abcdef123") ({strf::join_right(12, '~'), {"abc", "def", 123}});
    TEST("~~~abcdef123") .with(strf::fill(U'~')) ({strf::join_right(12), {"abc", "def", 123}});
    TEST("abcdef123") ({strf::join_internal(9), {"abc", "def", 1, 23}});
    TEST("abcdef123") ({strf::join_left(9), {"abc", "def", 123}});
    TEST("abcdef123") ({strf::join_right(9), {"abc", "def", 123}});

    int rc = boost::report_errors();
    return rc;
}
