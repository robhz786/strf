#include <boost/stringify.hpp>
#include <boost/assert.hpp>


void make_string_is_not_assignable()
{
    //[ make_string_is_not_assignable
    namespace strf = boost::stringify::v0;

    // make_string is not assignable:
    // auto str = strf::make_string  = {"blah", "blah", "blah"}; // compilation error
    // auto str = strf::make_string &= {"blah", "blah", "blah"}; // compilation error

    auto str1 = strf::make_string["{}{}{}"] &= {"blah", "blah", "blah"}; // ok

    auto str2 = strf::make_string.with() &= {"blah", "blah", "blah"}; // ok

    // the only purpose of adding an () is to get an assignable expression:
    auto str3 = strf::make_string() &= {"blah", "blah", "blah"}; // ok

    //]
}


void reserve()
{
    //[ syntax_reserve
    namespace strf = boost::stringify::v0;

    // reserving a bigger size because there are some further appends:
    std::string output = strf::make_string.reserve(500) &={"blah", "blah"};

    auto append = strf::append_to(output).no_reserve();

    append &= {"bleh", "bleh"};
    append &= {"blih", "blih"};
    append &= {"bloh", "bloh"};
    append.reserve_auto() &= {"bluh", "bluh"};
    //]
}


int main()
{
    make_string_is_not_assignable();
    reserve();

    return 0;
}
