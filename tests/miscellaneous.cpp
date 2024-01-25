//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

using namespace strf::width_literal;

STRF_TEST_FUNC void test_miscellaneous()
{
    {
        // write into an destination reference
        char buff[100];
        strf::cstr_destination str_writer{buff};
        strf::destination<char>& dst = str_writer;

        strf::to(dst)
            .with(strf::numpunct<10>(3))
            ("abc", ' ', strf::punct(1000000000LL));

        str_writer.finish();

        TEST_CSTR_EQ(buff, "abc 1,000,000,000");
    }
    {
        // write line
        char buff[100];
        auto res = strf::to(buff).line("abc", 1, 2, 3);

        TEST_FALSE(res.truncated);
        TEST_EQ(res.ptr, &buff[7]);
        TEST_CSTR_EQ(buff, "abc123\n");
    }
    {   // test discarder
        strf::discarder<char> dst;
        TEST_FALSE(dst.good());
        dst.recycle();
        TEST_TRUE(dst.buffer_space() >= strf::min_destination_buffer_size);
        TEST_FALSE(dst.good());
        char buff[200];
        dst.write(buff, sizeof(buff)/sizeof(buff[0]));
        TEST_FALSE(dst.good());
    }
    {   // measure size
        strf::premeasurements<strf::size_presence::yes, strf::width_presence::no> p;

        strf::measure<char>(&p, strf::pack());
        TEST_EQ(p.accumulated_ssize(), 0);

        strf::measure<char>(&p, strf::pack(), 1, 23, 456, 7890);
        TEST_EQ(p.accumulated_ssize(), 10);
    }

    {   // measure size and width
        strf::premeasurements<strf::size_presence::yes, strf::width_presence::yes>p{1000};

        strf::measure<char>(&p, strf::pack());
        TEST_EQ(p.accumulated_ssize(), 0);
        TEST_TRUE(p.remaining_width() == 1000);

        strf::measure<char>(&p, strf::pack(), 1, 23, 456, 7890);
        TEST_EQ(p.accumulated_ssize(), 10);
        TEST_TRUE(p.remaining_width() == 1000 - 10);
    }

    {   // measure width
        strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>p{8_w};

        strf::measure<char>(&p, strf::pack());
        TEST_TRUE(p.remaining_width() == 8_w);

        strf::measure<char>(&p, strf::pack(), 1, 23, 456);
        TEST_TRUE(p.remaining_width() == 2_w);
        strf::measure<char>(&p, strf::pack(), 1, 23, 456);
        TEST_TRUE(p.remaining_width() == 0);
        strf::measure<char>(&p, strf::pack(), 1);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // measure width
        strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>p{8_w};
        p.add_width(8);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // measure width
        strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>p{8_w};
        p.add_width(9);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // saturate_width
        const auto limit = 20.25_w;
        strf::premeasurements<strf::size_presence::no, strf::width_presence::yes> p{limit};
        p.saturate_width();
        p.add_width(10_w);
        TEST_TRUE(p.remaining_width() == 0);
        TEST_TRUE(p.accumulated_width() == limit);
    }
    {   // don't measure anything
        strf::no_premeasurements p;
        strf::measure<char>(&p, strf::pack());
        strf::measure<char>(&p, strf::pack(), 1, 23, 456);
        TEST_EQ(p.accumulated_ssize(), 0);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // make_simple_string_view
        const char16_t* str = u"abcdefghijklmnop";
        auto sv = strf::detail::make_simple_string_view(str, 5);
        TEST_TRUE(sv.data() == str);
        TEST_EQ(sv.size(), 5);
    }
    {
        // strf::detail::slow_countl_zero_ll
        TEST_EQ(64, strf::detail::slow_countl_zero_ll(0));
        for (int i = 0; i <= 63; ++i) {
            TEST_SCOPE_DESCRIPTION("i: ", i);
            TEST_EQ(63 - i, strf::detail::slow_countl_zero_ll(1ULL << i));
        }
    }

    {   // strf::detail::all_base_fmtfn_classes_are_empty

        using strf::detail::all_base_fmtfn_classes_are_empty;

        static_assert( all_base_fmtfn_classes_are_empty<decltype(strf::fmt(0))>::value, "");
        static_assert(!all_base_fmtfn_classes_are_empty<decltype(strf::fmt(0).p(5))>::value, "");
    }
}

REGISTER_STRF_TEST(test_miscellaneous)

