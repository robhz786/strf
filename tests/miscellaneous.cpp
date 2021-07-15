//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

using namespace strf::width_literal;

STRF_TEST_FUNC void test_miscellaneous()
{
    {
        // write into an outbuff reference
        char buff[100];
        strf::cstr_writer str_writer{buff};
        strf::outbuff& ob = str_writer;

        strf::to(ob)
            .with(strf::numpunct<10>(3))
            ("abc", ' ', strf::punct(1000000000ll));

        str_writer.finish();

        TEST_CSTR_EQ(buff, "abc 1,000,000,000");
    }
    {   // test discarded_outbuff
        strf::discarded_outbuff<char> ob;
        TEST_FALSE(ob.good());
        ob.recycle();
        TEST_TRUE(ob.space() >= strf::min_space_after_recycle<char>());
        TEST_FALSE(ob.good());
        char buff[200];
        ob.write(buff, sizeof(buff)/sizeof(buff[0]));
        TEST_FALSE(ob.good());
    }
    {   // preview size
        strf::print_preview<strf::preview_size::yes, strf::preview_width::no> p;

        strf::preview<char>(p, strf::pack());
        TEST_EQ(p.accumulated_size(), 0);

        strf::preview<char>(p, strf::pack(), 1, 23, 456, 7890);
        TEST_EQ(p.accumulated_size(), 10);
    }

    {   // preview size and width
        strf::print_preview<strf::preview_size::yes, strf::preview_width::yes>p{1000};

        strf::preview<char>(p, strf::pack());
        TEST_EQ(p.accumulated_size(), 0);
        TEST_TRUE(p.remaining_width() == 1000);

        strf::preview<char>(p, strf::pack(), 1, 23, 456, 7890);
        TEST_EQ(p.accumulated_size(), 10);
        TEST_TRUE(p.remaining_width() == 1000 - 10);
    }

    {   // preview width
        strf::print_preview<strf::preview_size::no, strf::preview_width::yes>p{8_w};

        strf::preview<char>(p, strf::pack());
        TEST_TRUE(p.remaining_width() == 8_w);

        strf::preview<char>(p, strf::pack(), 1, 23, 456);
        TEST_TRUE(p.remaining_width() == 2_w);
        strf::preview<char>(p, strf::pack(), 1, 23, 456);
        TEST_TRUE(p.remaining_width() == 0);
        strf::preview<char>(p, strf::pack(), 1);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // preview width
        strf::print_preview<strf::preview_size::no, strf::preview_width::yes>p{8_w};
        p.subtract_width(8);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // preview width
        strf::print_preview<strf::preview_size::no, strf::preview_width::yes>p{8_w};
        p.subtract_width(9);
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // clear_remaining_width
        strf::print_preview<strf::preview_size::no, strf::preview_width::yes> p{8_w};
        p.clear_remaining_width();
        TEST_TRUE(p.remaining_width() == 0);
    }
    {   // no preview
        strf::no_print_preview p;
        strf::preview<char>(p, strf::pack());
        strf::preview<char>(p, strf::pack(), 1, 23, 456);
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

