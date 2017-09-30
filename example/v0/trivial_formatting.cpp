#include <boost/stringify.hpp>
#include <boost/assert.hpp>

int main()
{
    namespace strf = boost::stringify::v0;

    //[trivial_formatting_sample
    int value = 255;

    strf::expected_string result; /*< `expected_string` is similar to
     [@http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0323r1.pdf
     `std::expected`]`<std::string, std::error_code>` >*/

    // write in hexadecimal
    result = strf::make_string("{} in hexadecimal is {}") = {value, {value, /*<<
    `'#'` = show base, `'x'` = hexadecimal. >>*/"#x"}};
    BOOST_ASSERT(*result == "255 in hexadecimal is 0xff");

    // with width equal to 6
    result = strf::make_string("--{}--") = {{value, 6}};
    //BOOST_ASSERT(*result == "--   255--");

    // with width and format string
    result = strf::make_string("--{}--") = {{value, {6, /*<<
       `'<'` = justify left >>*/"<#x"}}};
    //BOOST_ASSERT(*result == "--0xff  --");
    //]

    return 0;
}

