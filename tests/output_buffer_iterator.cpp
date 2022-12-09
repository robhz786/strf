//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/iterator.hpp>
#include <strf/to_string.hpp>
#include <algorithm>

STRF_TEST_FUNC void test_output_buffer_iterator()
{
    {
        const std::string sample = "small string";
        strf::string_maker dest;
        std::copy(sample.begin(), sample.end(), make_iterator(dest));
        auto result = dest.finish();
        TEST_EQ(sample, result);
    }
    {
        const auto sample_ = test_utils::make_double_string<char>();
        const std::string sample(sample_.begin(), sample_.end());
        strf::string_maker dest;
        std::copy(sample.begin(), sample.end(), make_iterator(dest));
        auto result = dest.finish();
        TEST_EQ(sample, result);
    }
}

REGISTER_STRF_TEST(test_output_buffer_iterator)

