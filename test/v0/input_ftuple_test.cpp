//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

int main()
{
    namespace strf = boost::stringify::v0;

    auto ft1 = strf::make_ftuple
        ( strf::monotonic_grouping<10>(3)
        , strf::monotonic_grouping<16>(4).thousands_sep('\'')
        , strf::monotonic_grouping<8>(6).thousands_sep('\'')
        );

    auto ft2 = strf::make_ftuple
        ( strf::monotonic_grouping<10>(3).thousands_sep('.')
        , strf::monotonic_grouping<16>(2).thousands_sep('^')
        );

    TEST("1,0,0,0,0 10000 1000000 10,000 1'0000 1'000000 10.000 1^00^00 1'000000")
        .with(strf::monotonic_grouping<10>(1)) &=
    {
        10000, ' ', strf::hex(0x10000), ' ', strf::oct(01000000), ' ', 
        { ft1
        ,   { 10000, ' ', strf::hex(0x10000), ' ', strf::oct(01000000), ' ' 
            , {ft2, {10000, ' ', strf::hex(0x10000), ' ', strf::oct(01000000)}}
            }
        }
    };

    int rc = report_errors() || boost::report_errors();
    return rc;
}

