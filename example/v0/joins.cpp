#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <iostream>

int main()
{
    //[joins_example
    namespace strf = boost::stringify::v0;

    auto result = strf::make_string()
        = {"<|", {strf::join_right(14), {"abc", "def", 123}}, "|>"};
    BOOST_ASSERT(result.value() == "<|     abcdef123|>");

    
    result = strf::make_string()
        = {"<|", {strf::join_left(14, U'.'), {"abc", {"def", 5}, 123}}, "|>"};
    BOOST_ASSERT(result.value() == "<|abc  def123...|>");

    
    result = strf::make_string.with(strf::fill(U'~'))
        = {"<|", {strf::join_internal(14, 2), {"abc", "def", 123}}, "|>"};
    BOOST_ASSERT(result.value() == "<|abcdef~~~~~123|>");

    
    result = strf::make_string.with(strf::fill(U'='))
        = {"<|", {strf::join_internal(14, 1, '.'), {{"abc", {5, "<"}}, "def", 123}}, "|>"};
    BOOST_ASSERT(result.value() == "<|abc==...def123|>");
    //]
    return 0;
} 
