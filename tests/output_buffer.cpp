//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

STRF_TEST_FUNC void test_output_buffer_functions()
{

    {   // Cover strf::detail::output_buffer_interchar_copy

        const unsigned char sample[10] =
            { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};

        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 5> dest
                { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 6> dest
                { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 4> dest
                { {"abcdefghi", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char16_t, 5, 5> dest
                { {u"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char16_t, 5, 6> dest
                { {u"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char16_t, 5, 4> dest
                { {u"abcdefghi", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::output_buffer_interchar_copy(dest, sample, sizeof(sample));

            dest.finish();
        }
    }
    {   // Cover strf::detail::write_fill
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 8> dest
                { {"aaaaaaaaaa", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::write_fill(dest, 10, 'a');
            TEST_TRUE(dest.good());
            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 5> dest
                { {"aaaaaaaaaa", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::write_fill(dest, 10, 'a');
            TEST_TRUE(dest.good());
            dest.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 3> dest
                { {"aaaaaaaa", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::write_fill(dest, 10, 'a');
            TEST_FALSE(dest.good());
            dest.finish();
        }
    }
    {
        // Cover write(output_buffer<CharT>&, const CharT*)
        test_utils::input_tester_with_fixed_spaces<char, 5> dest
            { {"abcde", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };
        write(dest, "abcde");
        dest.finish();
    }
    {
        // Cover put(output_buffer<CharT>&, CharT)

        test_utils::input_tester_with_fixed_spaces<char, 3, 5> dest
            { {"abcd", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        // When put does not call recycle_buffer()
        strf::put(dest, 'a');
        strf::put(dest, 'b');
        strf::put(dest, 'c');

        // When it does
        strf::put(dest, 'd');

        dest.finish();
    }
    {
        // Cover output_buffer<CharT>::ensure()
        test_utils::input_tester_with_fixed_spaces<char, 4, 4> dest
            { {"abcdefgh", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        dest.ensure(2); // when argument is less than buffer_space()
        dest.buffer_ptr()[0] = 'a';
        dest.buffer_ptr()[1] = 'b';
        dest.advance(2);

        dest.ensure(2); // when argument is equal to buffer_space()
        dest.buffer_ptr()[0] = 'c';
        dest.buffer_ptr()[1] = 'd';
        dest.advance(2);

        dest.ensure(4); // When argument is greater than buffer_space()
        dest.buffer_ptr()[0] = 'e';
        dest.buffer_ptr()[1] = 'f';
        dest.buffer_ptr()[2] = 'g';
        dest.buffer_ptr()[3] = 'h';
        dest.advance(4);

        dest.finish();
    }
    {
        // When output_buffer<CharT>::do_write() calls recycle_buffer()
        // and the buffer_space() afterwards is greater than needed

        test_utils::input_tester_with_fixed_spaces<char, 5, 10> dest
            { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        dest.write("abcdefghij", 10);
        dest.finish();
    }
    {
        // When output_buffer<CharT>::do_write() calls recycle_buffer()
        // and the buffer_space() afterwards is exactly as needed
        test_utils::input_tester_with_fixed_spaces<char, 5, 5> dest
            { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        dest.write("abcdefghij", 10);
        dest.finish();
    }
    {
        // When output_buffer<CharT>::do_write() calls recycle_buffer()
        // and the good() afterwards is false
        test_utils::input_tester_with_fixed_spaces<char, 5> dest
            { {"abcde", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        dest.write("abcdefghij", 10);
        dest.finish();
    }
}
REGISTER_STRF_TEST(test_output_buffer_functions);

