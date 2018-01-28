#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <iostream>

int main()
{
    //[joins_example
    namespace strf = boost::stringify::v0;

    auto result = strf::make_string()
       &= {"---", {strf::join_right(15), {"abc", "def", 123}}, "---"};
    BOOST_ASSERT(result == "---      abcdef123---");


    result = strf::make_string()
       &= {"---", {strf::join_center(15), {"abc", "def", 123}}, "---"};
    BOOST_ASSERT(result == "---   abcdef123   ---");


    result = strf::make_string()
       &= { "---"
          , { strf::join_left(15, U'.')
            , { "abc"
              , {"def", 5}
              , 123
              }
            }
          , "---"
          };

    BOOST_ASSERT(result == "---abc  def123....---");

    result = strf::make_string()
       &= { "---"
          , { strf::join_internal(15, '.', 1)
            , { {"abc", {5, "<"}}
              , "def"
              , 123
              }
            }
          , "---"};
    BOOST_ASSERT(result == "---abc  ....def123---");
    //]
    return 0;
}
