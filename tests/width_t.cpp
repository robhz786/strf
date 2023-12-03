//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

// NOLINTNEXTLINE(google-readability-function-size,hicpp-function-size)
STRF_TEST_FUNC void test_width_t()
{
    using namespace strf::width_literal;
    using namespace strf::detail::cast_sugars;

    using strf::width_t;
    constexpr strf::width_t width_max = (strf::width_t::max)();
    constexpr strf::width_t width_min = (strf::width_t::min)();
    constexpr auto max_underlying = width_max.underlying_value();
    constexpr auto min_underlying = width_min.underlying_value();
    constexpr strf::width_t epsilon = strf::width_t::from_underlying(1);

    TEST_TRUE(1.125_w  + 10.625_w == 11.75_w);
    TEST_TRUE(10.625_w -  1.125_w  ==  +9.5_w);
    TEST_TRUE(-1.125_w + -1.125_w  ==  -2.25_w);

    TEST_TRUE(1_w    / 1._w == 1);
    TEST_TRUE(10.5_w /  2  == +5.25_w);
    TEST_TRUE(10.5_w / -2_w  == -5.25_w);
    TEST_TRUE(10.5_w / -2  == -5.25_w);

    TEST_TRUE(10_w    /  2.5_w  ==  4);
    TEST_TRUE(25      /  2.5_w  ==  10);
    TEST_TRUE(10.5_w  /  2.5_w  ==  4.2_w);
    TEST_TRUE(10.5_w  /  2_w    ==  5.25_w);
    TEST_TRUE(10.25_w / -0.5_w  == -20.5_w);

    TEST_TRUE( 5.5_w *  2      ==  11);
    TEST_TRUE(-5.5_w * +2      == -11);

    TEST_TRUE(  2     *  2.5_w  ==  5);
    TEST_TRUE(  1.5_w *  5.5_w  ==  8.25_w);
    TEST_TRUE(- 1.5_w * -5.5_w  ==  8.25_w);
    TEST_TRUE( 10.5_w *  0.5_w  ==  5.25_w);
    TEST_TRUE(-10.5_w *  0.5_w  == -5.25_w);
    TEST_TRUE( 10.5_w * -0.5_w  == -5.25_w);
    TEST_TRUE(-10.5_w * -0.5_w  == +5.25_w);

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

    TEST_TRUE((0_w).zero());
    TEST_TRUE((0_w).ge_zero());
    TEST_TRUE((0_w).le_zero());
    TEST_FALSE((0_w).not_zero());
    TEST_TRUE((1_w).gt_zero());
    TEST_TRUE((1_w).ge_zero());
    TEST_TRUE((-1_w).lt_zero());
    TEST_TRUE((-1_w).le_zero());

    TEST_EQ(1,  (1.725_w).floor());
    TEST_EQ(-2, (-1.725_w).floor());
    TEST_EQ( 0x7FFF, width_max.floor());
    TEST_EQ(-0x8000, width_min.floor());

    TEST_EQ(1,  (1.725_w).non_negative_floor());
    TEST_EQ(0, (-1.725_w).non_negative_floor());
    TEST_EQ(0x7FFF, width_max.non_negative_floor());
    TEST_EQ(0, width_min.non_negative_floor());

    TEST_EQ(2,  (1.725_w).ceil());
    TEST_EQ(-1, (-1.725_w).ceil());
    TEST_EQ(0,  (-0.725_w).ceil());
    TEST_EQ( 0x8000, width_max.ceil());
    TEST_EQ(-0x8000, width_min.ceil());

    TEST_EQ(2,  (1.725_w).non_negative_ceil());
    TEST_EQ(0, (-1.725_w).non_negative_ceil());
    TEST_EQ(0, (-0.725_w).non_negative_ceil());
    TEST_EQ( 0x8000, width_max.non_negative_ceil());
    TEST_EQ(      0, width_min.non_negative_ceil());

    TEST_EQ(0, (0_w).round());
    TEST_EQ(0, (0.001_w).round());
    TEST_EQ(0, (0.5_w).round());
    TEST_EQ(1, (1.5_w).round());
    TEST_EQ(2, (1.50001_w).round());
    TEST_EQ(2, (2.5_w).round());
    TEST_EQ(3, (2.50001_w).round());

    TEST_EQ(-1, (-1.5_w).round());
    TEST_EQ(-2, (-1.50001_w).round());
    TEST_EQ(-2, (-2.5_w).round());
    TEST_EQ(-3, (-2.50001_w).round());

    TEST_EQ( 100, (+100_w).round());
    TEST_EQ(-100, (-100_w).round());
    TEST_EQ( 0x8000, width_max.round());
    TEST_EQ(-0x8000, width_min.round());

    // sat_add(width_t, width_t)
    TEST_TRUE(sat_add(1_w, 1_w)        == 2_w);
    TEST_TRUE(sat_add(1_w, -1_w)       == 0_w);
    TEST_TRUE(sat_add(1.25_w, 1.25_w)  == 2.5_w);
    TEST_TRUE(sat_add(2.75_w, -1.25_w) == 1.5_w);
    TEST_TRUE(sat_add(32000_w, 32000_w)   == width_max);
    TEST_TRUE(sat_add(-32000_w, -32000_w) == width_min);
    TEST_TRUE(sat_add(width_max, -width_max) == 0_w);
    //TEST_TRUE(sat_add(width_max,  width_min) == ???);

    // sat_sub(width_t, width_t)
    TEST_TRUE(sat_sub(1_w, 1_w)       == 0_w);
    TEST_TRUE(sat_sub(2.75_w, 1.25_w) == 1.5_w);
    TEST_TRUE(sat_sub(32000_w, -32000_w) == width_max);
    TEST_TRUE(sat_sub(-32000_w, 32000_w) == width_min);
    TEST_TRUE(sat_sub(width_max, width_max) == 0_w);

    // sat_add(width_t, int16_t)
    TEST_TRUE(sat_add(2.5_w,     cast_i16(2))  == 4.5_w);

    // sat_add(width_t, int32_t)
    TEST_TRUE(sat_add(1_w,       cast_i32(+1))  == 2_w);
    TEST_TRUE(sat_add(2.5_w,     cast_i32(-1))  == 1.5_w);
    TEST_TRUE(sat_add(width_max, cast_i32(-1))  == width_max - 1_w);
    TEST_TRUE(sat_add(width_max, cast_i32(+1))  == width_max);
    TEST_TRUE(sat_add(width_min, cast_i32(-1))  == width_min);
    TEST_TRUE(sat_add(0_w,       cast_i32(+0x8000))  == width_max);
    TEST_TRUE(sat_add(0_w,       cast_i32(-0x10000))  == width_min);
    TEST_EQ  (sat_add(width_max, cast_i32(-0xFFFF)).underlying_value(), (int32_t)0x8000FFFF);
    TEST_TRUE(sat_add(width_max, cast_i32(-0x10000)) == width_min);
    TEST_TRUE(sat_add(width_min, cast_i32( 0xFFFF))  == std::int16_t(0x7FFF) );
    TEST_TRUE(sat_add(width_min, cast_i32(0x10000))  == width_max);

    // sat_sub(width_t, int32_t)
    TEST_TRUE(sat_sub(1_w,       cast_i32(-1))  == 2_w);
    TEST_TRUE(sat_sub(2.5_w,     cast_i32(+1))  == 1.5_w);
    TEST_TRUE(sat_sub(width_max, cast_i32(+1))  == width_max - 1_w);
    TEST_TRUE(sat_sub(width_max, cast_i32(-1))  == width_max);
    TEST_TRUE(sat_sub(width_min, cast_i32(+1))  == width_min);
    TEST_TRUE(sat_sub(0_w,       cast_i32(-0x8000))  == width_max);
    TEST_TRUE(sat_sub(0_w,       cast_i32(+0x10000))  == width_min);
    TEST_EQ  (sat_sub(width_max, cast_i32(+0xFFFF)).underlying_value(), (int32_t)0x8000FFFF);
    TEST_TRUE(sat_sub(width_max, cast_i32(+0x10000)) == width_min);
    TEST_TRUE(sat_sub(width_min, cast_i32(-0xFFFF))  == std::int16_t(0x7FFF) );
    TEST_TRUE(sat_sub(width_min, cast_i32(-0x10000))  == width_max);

    // sat_add(width_t, int64_t)
    TEST_TRUE(sat_add(1_w,        1LL)  == 2_w);
    TEST_TRUE(sat_add(2.5_w,     -1LL)  == 1.5_w);
    TEST_TRUE(sat_add(width_max, -1LL)  == width_max - 1_w);
    TEST_TRUE(sat_add(width_max,  1LL)  == width_max);
    TEST_TRUE(sat_add(width_min, -1LL)  == width_min);
    TEST_TRUE(sat_add(width_min,  0x8000LL)  == 0);
    TEST_TRUE(sat_add(width_min,  0x8000LL + 0x7FFF) == std::int16_t(0x7FFF) );
    TEST_TRUE(sat_add(width_min,  0x10000LL)  == width_max);
    TEST_TRUE(sat_add(width_max, -0x10000LL)  == width_min);
    TEST_TRUE(sat_add(width_max, -0x10001LL)  == width_min);
    TEST_EQ  (sat_add(width_max,  -0xFFFFLL).underlying_value(), (int32_t)0x8000FFFF);
    TEST_TRUE(sat_add(0_w,  0x7FFFLL)   < width_max);
    TEST_TRUE(sat_add(0_w,  0x8000LL)  == width_max);
    TEST_TRUE(sat_add(0_w, -0x8000LL)  == width_min);
    TEST_TRUE(sat_add(0_w, -0x8001LL)  == width_min);

    // sat_sub(width_t, int64_t)
    TEST_TRUE(sat_sub(1_w,       -1LL)  == 2_w);
    TEST_TRUE(sat_sub(2.5_w,     +1LL)  == 1.5_w);
    TEST_TRUE(sat_sub(width_max, +1LL)  == width_max - 1_w);
    TEST_TRUE(sat_sub(width_max, -1LL)  == width_max);
    TEST_TRUE(sat_sub(width_min, +1LL)  == width_min);
    TEST_TRUE(sat_sub(width_min, -0x8000LL)  == 0);
    TEST_TRUE(sat_sub(width_min, -0x8000LL - 0x7FFF) == std::int16_t(0x7FFF) );
    TEST_TRUE(sat_sub(width_min, -0x10000LL)  == width_max);
    TEST_TRUE(sat_sub(width_max, +0x10000LL)  == width_min);
    TEST_TRUE(sat_sub(width_max, +0x10001LL)  == width_min);
    TEST_EQ  (sat_sub(width_max,  +0xFFFFLL).underlying_value(), (int32_t)0x8000FFFF);
    TEST_TRUE(sat_sub(0_w, -0x7FFFLL)   < width_max);
    TEST_TRUE(sat_sub(0_w, -0x8000LL)  == width_max);
    TEST_TRUE(sat_sub(0_w, +0x8000LL)  == width_min);
    TEST_TRUE(sat_sub(0_w, +0x8001LL)  == width_min);

    // sat_mul(width_t, width_t)
    TEST_TRUE(sat_mul(1.5_w, 1.5_w) == 2.25_w);
    TEST_TRUE(sat_mul(1_w, width_max - epsilon) == width_max - epsilon);
    TEST_TRUE(sat_mul(1_w, width_max) == width_max);
    TEST_TRUE(sat_mul(1_w + epsilon, width_max) == width_max);
    TEST_TRUE(sat_mul(1_w, -epsilon) == -epsilon);
    TEST_TRUE(sat_mul(0.25_w,  -epsilon) == 0_w);
    TEST_TRUE(sat_mul(epsilon, -epsilon) == 0_w);
    TEST_TRUE(sat_mul(1_w, width_min + epsilon) == width_min + epsilon);
    TEST_TRUE(sat_mul(1_w, width_min)   == width_min);
    TEST_TRUE(sat_mul(1_w + epsilon, width_min) == width_min);

    // sat_mul(width_t, int16_t)
    TEST_TRUE(sat_mul(1.5_w,  cast_i16(2))  == 3_w);

    // sat_mul(width_t, int32_t)
    TEST_TRUE(sat_mul(0_w,        cast_i32(400000)) == 0_w);
    TEST_TRUE(sat_mul(width_max,  cast_i32(0)) == 0_w);
    TEST_TRUE(sat_mul(+100.125_w, cast_i32(+2)) ==  200.25_w);
    TEST_TRUE(sat_mul(+100.125_w, cast_i32(-2)) == -200.25_w);
    TEST_TRUE(sat_mul(+18000_w,   cast_i32(+2)) == width_max);
    TEST_TRUE(sat_mul(+18000_w,   cast_i32(-2)) == width_min);
    TEST_TRUE(sat_mul(-18000_w,   cast_i32(+2)) == width_min);
    TEST_TRUE(sat_mul(-18000_w,   cast_i32(-2)) == width_max);
    TEST_TRUE(sat_mul(+1_w,       cast_i32(+40000)) == width_max);
    TEST_TRUE(sat_mul(+1_w,       cast_i32(-40000)) == width_min);
    TEST_TRUE(sat_mul(-1_w,       cast_i32(+40000)) == width_min);
    TEST_TRUE(sat_mul(-1_w,       cast_i32(-40000)) == width_max);

    // sat_mul(width_t, uint32_t)
    TEST_TRUE(sat_mul(0_w,        cast_u32(400000)) == 0_w);
    TEST_TRUE(sat_mul(width_max,  cast_u32(0)) == 0_w);
    TEST_TRUE(sat_mul(+100.125_w, cast_u32(2)) ==  200.25_w);
    TEST_TRUE(sat_mul(+18000_w,   cast_u32(2)) == width_max);
    TEST_TRUE(sat_mul(-18000_w,   cast_u32(2)) == width_min);
    TEST_TRUE(sat_mul(+1_w,       cast_u32(40000)) == width_max);
    TEST_TRUE(sat_mul(-1_w,       cast_u32(40000)) == width_min);

    // sat_mul(width_t, int64_t)
    TEST_TRUE(sat_mul(0_w, 10LL )   == 0_w);
    TEST_TRUE(sat_mul(-2.5_w, 10LL ) == -25_w);
    TEST_TRUE(sat_mul(+20000_w, 20000LL ) == width_max);
    TEST_TRUE(sat_mul(-20000_w, 20000LL ) == width_min);

    TEST_TRUE(sat_mul(+epsilon, +cast_i64(max_underlying) - 1)  < width_max);
    TEST_TRUE(sat_mul(+epsilon, +cast_i64(max_underlying)    ) == width_max);
    TEST_TRUE(sat_mul(+epsilon, +cast_i64(max_underlying) + 1) == width_max);
    TEST_TRUE(sat_mul(-epsilon, -cast_i64(max_underlying) + 1)  < width_max);
    TEST_TRUE(sat_mul(-epsilon, -cast_i64(max_underlying)    ) == width_max);
    TEST_TRUE(sat_mul(-epsilon, -cast_i64(max_underlying) - 1) == width_max);

    TEST_TRUE(sat_mul(+epsilon, +cast_i64(min_underlying) + 1)  > width_min);
    TEST_TRUE(sat_mul(+epsilon, +cast_i64(min_underlying)    ) == width_min);
    TEST_TRUE(sat_mul(+epsilon, +cast_i64(min_underlying) - 1) == width_min);
    TEST_TRUE(sat_mul(-epsilon, -cast_i64(min_underlying) - 1)  > width_min);
    TEST_TRUE(sat_mul(-epsilon, -cast_i64(min_underlying)    ) == width_min);
    TEST_TRUE(sat_mul(-epsilon, -cast_i64(min_underlying) + 1) == width_min);

    // sat_mul(width_t, uint64_t)
    TEST_TRUE(sat_mul(0_w, 10ULL )   == 0_w);
    TEST_TRUE(sat_mul(-2.5_w, 10ULL ) == -25_w);
    TEST_TRUE(sat_mul(+20000_w, 20000ULL ) == width_max);
    TEST_TRUE(sat_mul(-20000_w, 20000ULL ) == width_min);
    TEST_TRUE(sat_mul(+epsilon, cast_u64(max_underlying) - 1)  < width_max);
    TEST_TRUE(sat_mul(+epsilon, cast_u64(max_underlying)    ) == width_max);
    TEST_TRUE(sat_mul(+epsilon, cast_u64(max_underlying) + 1) == width_max);
    TEST_TRUE(sat_mul(-epsilon, cast_u64(-cast_i64(min_underlying) - 1))  > width_min);
    TEST_TRUE(sat_mul(-epsilon, cast_u64(-cast_i64(min_underlying)    )) == width_min);
    TEST_TRUE(sat_mul(-epsilon, cast_u64(-cast_i64(min_underlying) + 1)) == width_min);

    // compare(width_t, int16_t)
    TEST_TRUE(compare(2_w, cast_i16(1)) > 0);
    TEST_TRUE(compare(2_w, cast_i16(2)) == 0);
    TEST_TRUE(compare(1_w, cast_i16(2)) < 0);
    TEST_TRUE(compare(width_max, cast_i16(0x7FFF)) > 0);
    TEST_TRUE(compare(epsilon, cast_i16(0)) > 0);
    TEST_TRUE(compare(-epsilon, cast_i16(0)) < 0);
    TEST_TRUE(compare(width_min, cast_i16(0x8000)) == 0);
    TEST_TRUE(compare(width_min, cast_i16(0x8001))  < 0);

    // compare(width_t, uint16_t)
    TEST_TRUE(compare(2_w, cast_u16(1)) > 0);
    TEST_TRUE(compare(2_w, cast_u16(2)) == 0);
    TEST_TRUE(compare(1_w, cast_u16(2)) < 0);
    TEST_TRUE(compare(width_max, cast_u16(0xFFFF)) < 0);
    TEST_TRUE(compare(width_max, cast_u16(0x8000)) < 0);
    TEST_TRUE(compare(width_max, cast_u16(0x7FFF)) > 0);
    TEST_TRUE(compare(-epsilon, cast_u16(0)) < 0);

    // compare(width_t, int32_t)
    TEST_TRUE(compare(2_w, cast_i32(1)) > 0);
    TEST_TRUE(compare(2_w, cast_i32(2)) == 0);
    TEST_TRUE(compare(1_w, cast_i32(2)) < 0);
    TEST_TRUE(compare(width_max, cast_i32( 0x8000))  < 0);
    TEST_TRUE(compare(width_max, cast_i32( 0x7FFF))  > 0);
    TEST_TRUE(compare(width_min, cast_i32(-0x8000)) == 0);
    TEST_TRUE(compare(width_min, cast_i32(-0x8001))  > 0);

    // compare(width_t, uint32_t)
    TEST_TRUE(compare(2_w, cast_u32(1)) > 0);
    TEST_TRUE(compare(2_w, cast_u32(2)) == 0);
    TEST_TRUE(compare(1_w, cast_u32(2)) < 0);
    TEST_TRUE(compare(width_max, cast_u32(0x8000)) < 0);
    TEST_TRUE(compare(width_max, cast_u32(0x7FFF)) > 0);

    // compare(width_t, int64_t)
    TEST_TRUE(compare(2_w, cast_i64(1)) > 0);
    TEST_TRUE(compare(2_w, cast_i64(2)) == 0);
    TEST_TRUE(compare(1_w, cast_i64(2)) < 0);
    TEST_TRUE(compare(width_max, cast_i64( 0x8000))  < 0);
    TEST_TRUE(compare(width_max, cast_i64( 0x7FFF))  > 0);
    TEST_TRUE(compare(width_min, cast_i64(-0x8000)) == 0);
    TEST_TRUE(compare(width_min, cast_i64(-0x8001))  > 0);

    // compare(width_t, uint64_t)
    TEST_TRUE(compare(2_w, cast_u64(1)) > 0);
    TEST_TRUE(compare(2_w, cast_u64(2)) == 0);
    TEST_TRUE(compare(1_w, cast_u64(2)) < 0);
    TEST_TRUE(compare(width_max, cast_u64(0x8000)) < 0);
    TEST_TRUE(compare(width_max, cast_u64(0x7FFF)) > 0);

    // compare(integral, width_t)
    TEST_TRUE(compare(1, 2_w)  < 0);
    TEST_TRUE(compare(2, 2_w) == 0);
    TEST_TRUE(compare(3, 2_w)  > 0);

    // casting ints
    {
        int above_max = 0x8000;
        width_t w = above_max;
        TEST_TRUE(w == width_max);
    }
    {
        int below_min = -0x8001;
        width_t w = below_min;
        TEST_TRUE(w == width_min);
    }
    {
        std::uint16_t above_max = 0x8000;
        width_t w = above_max;
        TEST_TRUE(w == width_max);
    }
    {
        int within = 1;
        width_t w = within;
        TEST_TRUE(w == 1_w);
    }
}

REGISTER_STRF_TEST(test_width_t)

