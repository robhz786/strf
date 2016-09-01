#include "test_utils.hpp"
#include <string.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/stringify.hpp>

#define TEST test<__LINE__>

int main()
{
    TEST ( "asdf", {}, "asdf");
    TEST ( "asdf", {}, std::string("asdf"));


    

  return  boost::report_errors();
}





