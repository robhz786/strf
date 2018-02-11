#include <boost/stringify.hpp>
#include <boost/assert.hpp>


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
    make_string_is_not_assignable();
    reserve();

    return 0;
}
