#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <iostream>

int main()
{
    //[joins_example
    namespace strf = boost::stringify::v0;

    auto result = strf::make_string
       [{ "___[", {strf::join_right(14), {"abc", "def", 123}}, "]___"}];
    BOOST_ASSERT(result.value() == "___[     abcdef123]___");

    
    result = strf::make_string
       [{ "___[", {strf::join_left(14, U'.'), {"abc", {"def", 5}, 123}}, "]___"}];
    BOOST_ASSERT(result.value() == "___[abc  def123...]___");

    
    result = strf::make_string.with(strf::fill(U'~'))
       [{ "___[", {strf::join_internal(14, 2), {"abc", "def", 123}}, "]___"}];
    BOOST_ASSERT(result.value() == "___[abcdef~~~~~123]___");

    
    result = strf::make_string.with(strf::fill(U'='))
       [{ "___[", {strf::join_internal(14, 1, '.'), {{"abc", {5, "<"}}, "def", 123}}, "]___"}];
    BOOST_ASSERT(result.value() == "___[abc==...def123]___");
    //]
    return 0;
} 
