#include <boost/stringify.hpp>
#include <boost/assert.hpp>

int sample_formatting()
{
    //[trivial_formatting_sample
    namespace strf = boost::stringify::v1;

    int value = 255;

    // write in hexadecimal
    std::string result = strf::make_string(value, " in hexadecimal is ", {value, /*<<
    `'#'` = show base, `'x'` = hexadecimal. >>*/"#x"});
    BOOST_ASSERT(result == "255 in hexadecimal is 0xff");

    // with width equal to 6
    result = strf::make_string("----", {value, 6}, "----");
    BOOST_ASSERT(result == "----   255---");

    // width and formatting string
    result = strf::make_string("----", {value, {6, /*<<
    `'<'` = justify left >>*/"#x"}}, "----");
    BOOST_ASSERT(result == "----  0xff---");
    //]
}

