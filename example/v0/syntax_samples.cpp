#include <boost/stringify.hpp>
#include <boost/assert.hpp>



void reserve()
{
    //[ syntax_reserve
    namespace strf = boost::stringify::v0;

    // reserving a bigger size because there are some further appends:
    std::string str = strf::make_string.reserve(500) .exception("blah", "blah");

    // by the way, note that, in order to avoid repetition,
    // you can store part of the syntax in a variable:
    auto append_to_str = strf::append(str).no_reserve();

    append_to_str.exception("_bleh", "_bleh");
    append_to_str("--{}--{}--").exception("blih", "blih");
    append_to_str.exception("bloh", "bloh");
    append_to_str.reserve_auto() .exception("bluh", "bluh");

    BOOST_ASSERT(str == "blahblah_bleh_bleh--blih--blih--blohblohbluhbluh");
    //]
}


int main()
{
    reserve();

    return 0;
}
