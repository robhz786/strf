//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_cfile.hpp>

int main() {
    strf::narrow_cfile_writer<char, 1024> test_msg_dest(stdout);
    const test_utils::test_messages_destination_guard g{test_msg_dest};
    test_utils::run_all_tests();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_msg_dest, "All test passed!\n");
    } else {
        strf::to(test_msg_dest) (err_count, " tests failed!\n");
    }
    test_utils::test_messages_destination_ptr() = nullptr; // to silence clang-tidy
    return err_count;
}
