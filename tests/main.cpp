//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_cfile.hpp>

namespace test_utils {

static strf::destination<char>*& test_messages_destination_ptr()
{
    static strf::destination<char>* ptr = nullptr;
    return ptr;
}

void set_test_messages_destination(strf::destination<char>& dest)
{
    test_messages_destination_ptr() = &dest;
}

strf::destination<char>& test_messages_destination()
{
    auto * ptr = test_messages_destination_ptr();
    return *ptr;
}

} // namespace test_utils


int main() {
    strf::narrow_cfile_writer<char, 1024> test_msg_dest(stdout);
    test_utils::set_test_messages_destination(test_msg_dest);
    test_utils::run_all_tests();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_msg_dest, "All test passed!\n");
    } else {
        strf::to(test_msg_dest) (err_count, " tests failed!\n");
    }
    return err_count;
}
