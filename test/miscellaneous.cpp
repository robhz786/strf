#include "test_utils.hpp"
#include <strf.hpp>


int main()
{
    {
        // write into an outbuf reference
        strf::basic_string_maker<char> str_maker;
        strf::outbuf& ob = str_maker;

        strf::to(ob)
            .with(strf::monotonic_grouping<10>(3))
            ("abc", ' ', 1000000000ll);

        TEST_TRUE(str_maker.finish() == "abc 1,000,000,000");
    }
    return test_finish();
}
