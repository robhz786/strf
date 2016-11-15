#include "test_utils.hpp"
#include <string.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/stringify.hpp>

int main()
{
    namespace strf = boost::stringify;

    testf<__LINE__>( "asdf") () ("asdf");
    testf<__LINE__>("~~AA~~BB") (strf::fill('~'), strf::width(4)) ("AA", "BB");
  
    return  boost::report_errors();
}





