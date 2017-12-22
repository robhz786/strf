#include <boost/stringify.hpp>
#include <boost/assert.hpp>




void arg_formatting()
{                                   
    //[trivial_formatting_sample
    namespace strf = boost::stringify::v0;
    int value = 255;

    /*<< `expected` is a bare implementation of __STD_EXPECTED__
      >>*/strf::expected<std::string, std::error_code> result;

    // write in hexadecimal
    result = strf::make_string["{} in hexadecimal is {}"] = {value, {value, /*<<
    This is what is here called as the /format string/.
    `'x'` implies hexadecimal base and `'#'` to show the base indication. >>*/"#x"}};
    BOOST_ASSERT(result.value() == "255 in hexadecimal is 0xff");

    // with width equal to 6
    result = strf::make_string ["--{}--"] = {{value, 6}};
    BOOST_ASSERT(result.value() == "--   255--");

    // with width and format string
    result = strf::make_string ["--{}--"] = {{value, {6, /*<<
       `The '<'` flag means left alignment>>*/"<#x"}}};
    BOOST_ASSERT(result.value() == "--0xff  --");
    //]
}



void make_string()
{
       
//[ trivial_make_string_sample
    namespace strf = boost::stringify::v0;
    auto str = strf::make_string["ten = {}, and twenty = {}"] = {10, 20};

    BOOST_ASSERT(str.value() == "ten = 10, and twenty = 20");
//]
}

void make_string_is_not_assignable()
{
//[ make_string_is_not_assignable
    namespace strf = boost::stringify::v0;

    auto xstr1 = strf::make_string ({"blah", "blah", "blah"}); // ok

  //auto xstr2 = strf::make_string = {"blah", "blah", "blah"}; // compilation error

    // the only purpose of adding an () is to return an assignable equivalent:
    auto xstr3 = strf::make_string() = {"blah", "blah", "blah"}; // ok

    auto xstr4 = strf::make_string.with() = {"blah", "blah", "blah"}; // ok
//]
}


void reserve()
{
//[ syntax_reserve
    namespace strf = boost::stringify::v0;

    // reserving a bigger size because there are some further appends:
    std::string output = strf::make_string.reserve(500) &={"blah", "blah"};

    auto append = strf::append_to(output).no_reserve();

    append &={"bleh", "bleh"};
    append &={"blih", "blih"};
    append &={"bloh", "bloh"};
    append.reserve_auto() &={"bluh", "bluh"};
//]
}
    




int main()
{
    arg_formatting();
    make_string();
    make_string_is_not_assignable();
    reserve();

    return 0;
}
