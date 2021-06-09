//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

void STRF_TEST_FUNC test_basic_outbuff()
{

    {   // Cover strf::detail::outbuff_interchar_copy

        const unsigned char sample[10] =
            { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};

        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 5> ob
                { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::outbuff_interchar_copy(ob, sample, sizeof(sample));

            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 6> ob
                { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::outbuff_interchar_copy(ob, sample, sizeof(sample));

            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 4> ob
                { {"abcdefghi", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::outbuff_interchar_copy(ob, sample, sizeof(sample));

            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char16_t, 5, 5> ob
                { {u"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::outbuff_interchar_copy(ob, sample, sizeof(sample));

            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char16_t, 5, 6> ob
                { {u"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::outbuff_interchar_copy(ob, sample, sizeof(sample));

            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char16_t, 5, 4> ob
                { {u"abcdefghi", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::outbuff_interchar_copy(ob, sample, sizeof(sample));

            ob.finish();
        }
    }
    {   // Cover strf::detail::write_fill
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 8> ob
                { {"aaaaaaaaaa", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::write_fill(ob, 10, 'a');
            TEST_TRUE(ob.good());
            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 5> ob
                { {"aaaaaaaaaa", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::write_fill(ob, 10, 'a');
            TEST_TRUE(ob.good());
            ob.finish();
        }
        {
            test_utils::input_tester_with_fixed_spaces<char, 5, 3> ob
                { {"aaaaaaaa", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

            strf::detail::write_fill(ob, 10, 'a');
            TEST_FALSE(ob.good());
            ob.finish();
        }
    }
    {
        // Cover write(basic_outbuff<CharT>&, const CharT*)
        test_utils::input_tester_with_fixed_spaces<char, 5> ob
            { {"abcde", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };
        write(ob, "abcde");
        ob.finish();
    }
    {
        // Cover put(basic_outbuff<CharT>&, CharT)

        test_utils::input_tester_with_fixed_spaces<char, 3, 5> ob
            { {"abcd", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        // When put does not call recycle()
        strf::put(ob, 'a');
        strf::put(ob, 'b');
        strf::put(ob, 'c');

        // When it does
        strf::put(ob, 'd');

        ob.finish();
    }
    {
        // Cover basic_outbuff<CharT>::ensure()
        test_utils::input_tester_with_fixed_spaces<char, 4, 4> ob
            { {"abcdefgh", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        ob.ensure(2); // when argument is less than space()
        ob.pointer()[0] = 'a';
        ob.pointer()[1] = 'b';
        ob.advance(2);

        ob.ensure(2); // when argument is equal to space()
        ob.pointer()[0] = 'c';
        ob.pointer()[1] = 'd';
        ob.advance(2);

        ob.ensure(4); // When argument is greater than space()
        ob.pointer()[0] = 'e';
        ob.pointer()[1] = 'f';
        ob.pointer()[2] = 'g';
        ob.pointer()[3] = 'h';
        ob.advance(4);

        ob.finish();
    }
    {
        // When basic_outbuff<CharT>::do_write() calls recycle()
        // and the space() afterwards is greater than needed

        test_utils::input_tester_with_fixed_spaces<char, 5, 10> ob
            { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        ob.write("abcdefghij", 10);
        ob.finish();
    }
    {
        // When basic_outbuff<CharT>::do_write() calls recycle()
        // and the space() afterwards is exactly as needed
        test_utils::input_tester_with_fixed_spaces<char, 5, 5> ob
            { {"abcdefghij", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        ob.write("abcdefghij", 10);
        ob.finish();
    }
    {
        // When basic_outbuff<CharT>::do_write() calls recycle()
        // and the good() afterwards is false
        test_utils::input_tester_with_fixed_spaces<char, 5> ob
            { {"abcde", __FILE__, __LINE__, BOOST_CURRENT_FUNCTION} };

        ob.write("abcdefghij", 10);
        ob.finish();
    }
}

REGISTER_STRF_TEST(test_basic_outbuff);

