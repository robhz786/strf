//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <stringify.hpp>
#include <limits>
#include "test_utils.hpp"

int main()
{
    using namespace strf::width_literal;

    BOOST_TEST(1.125_w  + 10.625_w == 11.75_w);
    BOOST_TEST(10.625_w - 1.125_w  ==  +9.5_w);
    BOOST_TEST(1.125_w  - 10.625_w ==  -9.5_w);

    BOOST_TEST(1_w   / 1._w == 1);
    BOOST_TEST(10.5_w / 2     == 5.25_w);
    BOOST_TEST(10_w   / 2.5_w == 4);
    BOOST_TEST(25     / 2.5_w == 10);
    BOOST_TEST(10.5_w / 2.5_w == 4.2_w);

    BOOST_TEST(5.5_w *  2      == 11);
    BOOST_TEST(2     *  2.5_w  == 5);
    BOOST_TEST(1.5_w *  5.5_w  == 8.25_w);
    BOOST_TEST(1.5_w * -5.5_w  == -8.25_w);

    {
        strf::width_t x = 12;
        x /= 3;
        BOOST_TEST(x == 4);
    }
    {
        strf::width_t x = 1;
        x /= 2;
        x = 4 * x;
        BOOST_TEST(x == 2);
    }
    {
        strf::width_t x = 1.5_w;
        x *= 3;
        BOOST_TEST(x == 4.5_w);
    }

    {
        strf::width_t x = 1.5_w;
        BOOST_TEST(x.round() == 2);
        BOOST_TEST(x.round() != 1);
    }
    {
        strf::width_t x = 2.5_w;
        BOOST_TEST(x.round() == 2);
        BOOST_TEST(x.round() != 3);
    }


    return boost::report_errors();
}

