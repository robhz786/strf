//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

void STRF_TEST_FUNC test_miscellaneous()
{
    {
        // write into an outbuff reference
        char buff[100];
        strf::cstr_writer str_writer{buff};
        strf::outbuff& ob = str_writer;

        strf::to(ob)
            .with(strf::numpunct<10>(3))
            ("abc", ' ', 1000000000ll);

        str_writer.finish();

        TEST_CSTR_EQ(buff, "abc 1,000,000,000");
    }
}
