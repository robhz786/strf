#include <boost/stringify.hpp>
#include <boost/assert.hpp>

int main()
{
    namespace strf = boost::stringify::v1;
    auto result = strf::make_string("---", {strf::join > 14, {"abc", "def", 123}}, "---");
    BOOST_ASSERT(result == "---     abcdef123---");

    result = strf::make_string({strf::join(U'.') < 14, {"abc", "def", 123}});
    BOOST_ASSERT(result == "abcdef123.....");

    auto mk_string_f = strf::make_string.with(strf::fill(U'~'));

    result = mk_string_f({strf::join > 14, {"abc", "def", 123}});
    BOOST_ASSERT(result == "~~~~~abcdef123");

    result = mk_string_f({strf::join('.') > 14, {"abc", "def", 123}});
    BOOST_ASSERT(result == ".....abcdef123");
    
} 
