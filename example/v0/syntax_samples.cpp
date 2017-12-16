#include <boost/stringify.hpp>
#include <boost/assert.hpp>




void arg_formatting()
{                                   
    //[trivial_formatting_sample
    namespace strf = boost::stringify::v0;
    int value = 255;

    strf::expected_string result; /*< `expected_string` is similar to
     [@http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0323r1.pdf
     `std::expected`]`<std::string, std::error_code>` >*/

    // write in hexadecimal
    result = strf::make_string("{} in hexadecimal is {}") = {value, {value, /*<<
    This is what is here called as the /format string/.
    `'x'` implies hexadecimal base and `'#'` to show the base indication. >>*/"#x"}};
    BOOST_ASSERT(*result == "255 in hexadecimal is 0xff");

    // with width equal to 6
    result = strf::make_string("--{}--") = {{value, 6}};
    //BOOST_ASSERT(*result == "--   255--");

    // with width and format string
    result = strf::make_string("--{}--") = {{value, {6, /*<<
       `'<'` justifies to left >>*/"<#x"}}};
    //BOOST_ASSERT(*result == "--0xff  --");
    //]
}



void make_string()
{
       
//[ trivial_make_string_sample
    namespace strf = boost::stringify::v0;
    strf::expected_string xstr = strf::make_string("ten = {}, and twenty = {}") = {10, 20};

    BOOST_ASSERT(xstr && *xstr == "ten = 10, and twenty = 20");
//]
}

void make_string_is_not_assignable()
{
//[ make_string_is_not_assignable
    namespace strf = boost::stringify::v0;
    /*
      auto xstr1 = strf::make_string = {"blah", "blah", "blah"}; // compilation error
    */
    auto xstr2 = strf::make_string() = {"blah", "blah", "blah"}; // ok

    auto xstr3 = strf::make_string [{"blah", "blah", "blah"}]; // ok
//]
}


int main()
{
    arg_formatting();
    make_string();
    make_string_is_not_assignable();
    
    return 0;
}
