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
        strf::cstr_writer str_writer{buff};
        strf::destination<char>& dest = str_writer;

        strf::to(dest)
            .with(strf::numpunct<10>(3))
            ("abc", ' ', strf::punct(1000000000ll));

        str_writer.finish();

        TEST_CSTR_EQ(buff, "abc 1,000,000,000");
    }
    {   // test discarder
        strf::discarder<char> dest;
        TEST_FALSE(dest.good());
        dest.recycle_buffer();
        TEST_TRUE(dest.buffer_space() >= strf::destination_space_after_flush);
        TEST_FALSE(dest.good());
        char buff[200];
        dest.write(buff, sizeof(buff)/sizeof(buff[0]));
        TEST_FALSE(dest.good());
    }
    {   // preview size
        strf::pre_printing<strf::precalc_size::yes, strf::precalc_width::no> p;

        strf::precalculate<char>(p, strf::pack());
        TEST_EQ(p.accumulated_size(), 0);

        strf::precalculate<char>(p, strf::pack(), 1, 23, 456, 7890);
        TEST_EQ(p.accumulated_size(), 10);
    }

    {   // preview size and width
        strf::pre_printing<strf::precalc_size::yes, strf::precalc_width::yes>p{1000};

        strf::precalculate<char>(p, strf::pack());
        TEST_EQ(p.accumulated_size(), 0);
        TEST_TRUE(p.remaining_width() == 1000);

        strf::precalculate<char>(p, strf::pack(), 1, 23, 456, 7890);
        TEST_EQ(p.accumulated_size(), 10);
        TEST_TRUE(p.remaining_width() == 1000 - 10);
    }

    {   // preview width
        strf::pre_printing<strf::precalc_size::no, strf::precalc_width::yes>p{8_w};

        strf::precalculate<char>(p, strf::pack());
        TEST_TRUE(p.remaining_width() == 8_w);

        strf::precalculate<char>(p, strf::pack(), 1, 23, 456);
        TEST_TRUE(p.remaining_width() == 2_w);
        strf::precalculate<char>(p, strf::pack(), 1, 23, 456);
        TEST_TRUE(p.remaining_width() == 0);
        strf::precalculate<char>(p, strf::pack(), 1);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // preview width
        strf::pre_printing<strf::precalc_size::no, strf::precalc_width::yes>p{8_w};
        p.subtract_width(8);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // preview width
        strf::pre_printing<strf::precalc_size::no, strf::precalc_width::yes>p{8_w};
        p.subtract_width(9);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // clear_remaining_width
        strf::pre_printing<strf::precalc_size::no, strf::precalc_width::yes> p{8_w};
        p.clear_remaining_width();
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // no preview
        strf::no_pre_printing p;
        strf::precalculate<char>(p, strf::pack());
        strf::precalculate<char>(p, strf::pack(), 1, 23, 456);
        TEST_EQ(p.accumulated_size(), 0);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // make_simple_string_view
        const char16_t* str = u"abcdefghijklmnop";
        auto sv = strf::detail::make_simple_string_view(str, 5);
        TEST_TRUE(sv.data() == str);
        TEST_EQ(sv.size(), 5);
    }
}

REGISTER_STRF_TEST(test_miscellaneous);

