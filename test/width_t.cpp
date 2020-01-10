//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <limits>
#include "test_utils.hpp"

int main()
{
    using namespace strf::width_literal;

    TEST_TRUE(1.125_w  + 10.625_w == 11.75_w);
    TEST_TRUE(10.625_w - 1.125_w  ==  +9.5_w);
    TEST_TRUE(1.125_w  - 10.625_w ==  -9.5_w);

    TEST_TRUE(1_w   / 1._w == 1);
    TEST_TRUE(10.5_w / 2     == 5.25_w);
    TEST_TRUE(10_w   / 2.5_w == 4);
    TEST_TRUE(25     / 2.5_w == 10);
    TEST_TRUE(10.5_w / 2.5_w == 4.2_w);

    TEST_TRUE(5.5_w *  2      == 11);
    TEST_TRUE(2     *  2.5_w  == 5);
    TEST_TRUE(1.5_w *  5.5_w  == 8.25_w);
    TEST_TRUE(1.5_w * -5.5_w  == -8.25_w);

    {
        strf::width_t x = 12;
        x /= 3;
        TEST_TRUE(x == 4);
    }
    {
        strf::width_t x = 1;
        x /= 2;
        x = 4 * x;
        TEST_TRUE(x == 2);
    }
    {
        strf::width_t x = 1.5_w;
        x *= 3;
        TEST_TRUE(x == 4.5_w);
    }

    {
        strf::width_t x = 1.5_w;
        TEST_TRUE(x.round() == 2);
        TEST_TRUE(x.round() != 1);
    }
    {
        strf::width_t x = 2.5_w;
        TEST_TRUE(x.round() == 2);
        TEST_TRUE(x.round() != 3);
    }


    return test_finish();
}

