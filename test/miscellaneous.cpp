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

        BOOST_TEST(str_maker.finish() == "abc 1,000,000,000");
    }
    return boost::report_errors();
}
