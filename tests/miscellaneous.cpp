//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

using namespace strf::width_literal;

void STRF_TEST_FUNC test_miscellaneous()
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
    }

    {   // no preview
        strf::no_print_preview p;
        strf::preview<char>(p, strf::pack());
        strf::preview<char>(p, strf::pack(), 1, 23, 456);
        TEST_EQ(p.accumulated_size(), 0);
        TEST_TRUE(p.remaining_width() == 0);
    }
}
