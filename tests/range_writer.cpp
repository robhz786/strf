//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

template <typename CharT>
STRF_TEST_FUNC void char_range_basic_operations()
{
    constexpr std::size_t buff_size = 8;
    CharT buff[buff_size];
    {   // cover strf::to_range
        constexpr std::size_t expected_size = 8;
        const CharT expected[expected_size] = { '1', '2', '3', '4', '5', 'a', 'b', 'c' };
        {
            auto res = strf::to_range(buff)
                ( 12345, static_cast<CharT>('a')
                , static_cast<CharT>('b'), static_cast<CharT>('c'));
            TEST_EQ(res.ptr - buff, 8);
            TEST_FALSE(res.truncated);
            TEST_TRUE(strf::detail::str_equal(buff, expected, expected_size));
        }
        {
            auto res = strf::to_range(buff)
                ( 12345, static_cast<CharT>('a'), static_cast<CharT>('b')
                , static_cast<CharT>('c'), static_cast<CharT>('d') );
            TEST_EQ(res.ptr - buff, 8);
            TEST_TRUE(res.truncated);
            TEST_TRUE(strf::detail::str_equal(buff, expected, expected_size));
        }
    }
    {   // construct from array
        strf::array_destination<CharT> sw(buff);
        TEST_TRUE(sw.buffer_ptr() == &buff[0]);
        TEST_EQ(sw.buffer_space(), buff_size);
        TEST_TRUE(sw.good());
    }
    {   // construct from pointer and size
        strf::array_destination<CharT> sw(buff, 4);
        TEST_TRUE(sw.buffer_ptr() == &buff[0]);
        TEST_EQ(sw.buffer_space(), 4);
        TEST_TRUE(sw.good());
    }
    {   // construct from range
        strf::array_destination<CharT> sw(buff, buff + 4);
        TEST_TRUE(sw.buffer_ptr() == &buff[0]);
        TEST_EQ(sw.buffer_space(), 4);
        TEST_TRUE(sw.good());
    }
    {   // Calling flush() always fails
        strf::array_destination<CharT> sw(buff);
        sw.flush();
        TEST_TRUE(sw.buffer_ptr() != &buff[0]);
        TEST_TRUE(sw.buffer_space() >= strf::min_destination_buffer_size)
        TEST_FALSE(sw.good());

        // and causes buffer_ptr() to point to somewhere else than
        // anywhere inside the initial range
        TEST_FALSE(&buff[0] <= sw.buffer_ptr() && sw.buffer_ptr() < buff + buff_size);
    }
    {   // When calling finish() after flush(),
        // the returned pointer is equal to the value buffer_ptr()
        // would have returned just before flush()

        strf::array_destination<CharT> sw(buff);
        strf::put(sw, static_cast<CharT>('a'));
        strf::put(sw, static_cast<CharT>('b'));
        strf::put(sw, static_cast<CharT>('c'));
        sw.flush();
        auto r = sw.finish();
        TEST_TRUE(r.truncated);
        TEST_TRUE(r.ptr == buff + 3);
    }
    // {   // Copy constructor
    //     strf::array_destination<CharT> sw1(buff);
    //     strf::array_destination<CharT> sw2{sw1};
    //     TEST_TRUE(sw1 == sw2);
    //     TEST_TRUE(sw1.buffer_ptr() == sw2.buffer_ptr());
    //     TEST_TRUE(sw1.buffer_end() == sw2.buffer_end());
    //     TEST_EQ(sw1.good(), sw2.good());
    //     auto r1 = sw1.finish();
    //     auto r2 = sw2.finish();
    //     TEST_TRUE(r1.ptr == r2.ptr);
    //     TEST_EQ(r1.truncated, r2.truncated);
    // }
    // {   // Copy constructor
    //     // copy "bad" object

    //     strf::array_destination<CharT> sw1(buff);
    //     strf::put(sw1, static_cast<CharT>('a'));
    //     strf::put(sw1, static_cast<CharT>('b'));
    //     strf::put(sw1, static_cast<CharT>('c'));

    //     sw1.flush();

    //     strf::array_destination<CharT> sw2{sw1};
    //     TEST_TRUE(sw1 == sw2);
    //     TEST_TRUE(sw1.buffer_ptr() == sw2.buffer_ptr());
    //     TEST_TRUE(sw1.buffer_end() == sw2.buffer_end());
    //     TEST_EQ(sw1.good(), sw2.good());
    //     auto r1 = sw1.finish();
    //     auto r2 = sw2.finish();
    //     TEST_TRUE(r1.ptr == r2.ptr);
    //     TEST_EQ(r1.truncated, r2.truncated);
    // }
    // {   // Copy assignment

    //     CharT buff2[10];
    //     strf::array_destination<CharT> sw1(buff);
    //     strf::array_destination<CharT> sw2(buff2);

    //     TEST_FALSE(sw1 == sw2);

    //     strf::put(sw1, static_cast<CharT>('a'));
    //     strf::put(sw1, static_cast<CharT>('b'));
    //     strf::put(sw1, static_cast<CharT>('c'));

    //     strf::put(sw2, static_cast<CharT>('a'));
    //     strf::put(sw2, static_cast<CharT>('b'));
    //     strf::put(sw2, static_cast<CharT>('c'));

    //     sw1 = sw2;
    //     TEST_TRUE(sw1 == sw2);
    //     TEST_TRUE(sw1.buffer_ptr() == sw2.buffer_ptr());
    //     TEST_TRUE(sw1.buffer_end() == sw2.buffer_end());
    //     TEST_EQ(sw1.good(), sw2.good());
    //     auto r1 = sw1.finish();
    //     auto r2 = sw2.finish();
    //     TEST_TRUE(r1.ptr == r2.ptr);
    //     TEST_EQ(r1.truncated, r2.truncated);
    // }
    // {   // Copy assignment
    //     // copy a "bad" object

    //     CharT buff2[10];
    //     strf::array_destination<CharT> sw1(buff);
    //     strf::array_destination<CharT> sw2(buff2);

    //     TEST_FALSE(sw1 == sw2);

    //     strf::put(sw1, static_cast<CharT>('a'));
    //     strf::put(sw1, static_cast<CharT>('b'));
    //     strf::put(sw1, static_cast<CharT>('c'));

    //     strf::put(sw2, static_cast<CharT>('a'));
    //     strf::put(sw2, static_cast<CharT>('b'));
    //     strf::put(sw2, static_cast<CharT>('c'));
    //     sw2.flush();

    //     sw1 = sw2;
    //     TEST_TRUE(sw1 == sw2);
    //     TEST_TRUE(sw1.buffer_ptr() == sw2.buffer_ptr());
    //     TEST_TRUE(sw1.buffer_end() == sw2.buffer_end());
    //     TEST_EQ(sw1.good(), sw2.good());
    //     auto r1 = sw1.finish();
    //     auto r2 = sw2.finish();
    //     TEST_TRUE(r1.ptr == r2.ptr);
    //     TEST_EQ(r1.truncated, r2.truncated);
    // }
}

static STRF_TEST_FUNC void char_range_destination_too_small()
{
    {
        char buff[4];
        strf::array_destination<char> sw(buff);
        TEST_EQ(sw.buffer_space(), 4);
        strf::put(sw, 'a');
        TEST_EQ(sw.buffer_space(), 3);
        strf::put(sw, 'b');
        strf::put(sw, 'c');
        TEST_TRUE(sw.good());
        strf::put(sw, 'd');
        TEST_EQ(sw.buffer_space(), 0);
        strf::put(sw, 'e');
        TEST_FALSE(sw.good());
        strf::put(sw, 'f');

        auto r = sw.finish();

        TEST_TRUE(r.truncated);
        TEST_TRUE(r.ptr == buff + 4);
        TEST_STRVIEW_EQ(buff, "abcd", 4);
    }
    {
        char buff[8];
        strf::array_destination<char> sw(buff);
        write(sw, "Hello");
        write(sw, " World");
        write(sw, "blah blah blah");
        auto r = sw.finish();

        TEST_TRUE(r.truncated);
        TEST_TRUE(r.ptr == buff + 8);
        TEST_STRVIEW_EQ(buff, "Hello Wo", 8);
    }
}

STRF_TEST_FUNC void test_to_range()
{
    char_range_basic_operations<char>();
    char_range_basic_operations<char16_t>();
    char_range_destination_too_small();
}


REGISTER_STRF_TEST(test_to_range);

