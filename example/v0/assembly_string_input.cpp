//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


void sample()
{
    //[ assembly_string_as_input
    namespace strf = boost::stringify::v0;
    
    auto result = strf::make_string["{} --- {} --- {}"] &=
    {   "aaa"
    ,   {    strf::assm("( {} {} )")
        ,    {   "bbb"
             ,   "ccc"
             }
        }
    ,   "ddd"
    };

    BOOST_ASSERT(result == "aaa --- ( bbb ccc ) --- ddd");
    //]
}
    
