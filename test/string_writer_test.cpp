#include "test_utils.hpp"
#include <string.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/stringify.hpp>

int main()
{
    test<__LINE__> ( "asdf", {}, "asdf");
    test_with_traits<__LINE__, char, std::char_traits<char> > ( "asdf", {}, std::string("asdf"));
  
    return  boost::report_errors();
}





