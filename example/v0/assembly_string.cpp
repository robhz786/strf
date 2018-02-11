#include <boost/assert.hpp>
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

int main()
{
    {
        //[ asmstr_escape_sample
        auto str = strf::make_string["} }/ {/ } {}"] &= {"aaa"};
        BOOST_ASSERT(str == "} }/ { } aaa");
        //]
    }

    {
        //[ asmstr_comment_sample
        auto str = strf::make_string
            ["You can learn more about python{-the programming language, not the reptile} at {}"]
            &= {"www.python.org"};
        BOOST_ASSERT(str == "You can learn more about python at www.python.org");
        //]
    }

    
    {
        //[ asmstr_positional_arg
        auto str = strf::make_string ["{1 person} likes {0 food}."] &= {"sandwich", "Paul"};
        BOOST_ASSERT(str == "Paul likes sandwich.");
        //]
    }

    {
        //[ asmstr_non_positional_arg
        auto str = strf::make_string ["{person} likes {food}."] &= {"Paul", "sandwich"};
        BOOST_ASSERT(str == "Paul likes sandwich.");
        //]
    }

    return 0;    
};
