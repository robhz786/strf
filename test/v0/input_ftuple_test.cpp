//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

int main()
{
    namespace strf = boost::stringify::v0;
    auto fmt1 = strf::make_ftuple(strf::hex, strf::showbase, strf::fill(U'.'));
    auto fmt2 = strf::make_ftuple(strf::noshowbase, strf::uppercase, strf::fill(U'~'));

    TEST( "---10..0xb..0xc~~~~D~14~~~~F.0x10---17---18")
        .with(strf::width(5), strf::fill(U'-'))
        (10, {fmt1, {11, 12, {fmt2, {13, {14, {3, "d"}}, 15}}, 16}}, 17, 18);

    TEST( "1011") (10, fmt1, {fmt2, {}}, 11);


    int rc = boost::report_errors();
    return rc;
}













