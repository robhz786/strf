#include "test_utils.hpp"
#include <string.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/stringify.hpp>

int main()
{
    testf<__LINE__>( "asdf") () ("asdf");
    
  
    return  boost::report_errors();
}





