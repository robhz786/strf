#include "test_utils.hpp"
#include <strf.hpp>


int main()
{
    {
        // write into an outbuff reference
        strf::basic_string_maker<char> str_maker;
        strf::outbuff& ob = str_maker;

        strf::to(ob)
            .with(strf::numpunct<10>(3))
            ("abc", ' ', 1000000000ll);

        TEST_TRUE(str_maker.finish() == "abc 1,000,000,000");
    }
    return test_finish();
}
