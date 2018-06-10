//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


void sample()
{
    //[ range_sample
    namespace strf = boost::stringify::v0;
    
    std::vector<int> vec = { 11, 22, 33 };

    auto str = strf::to_string()
        ( " ("
        , strf::iterate(vec, ", ")
        , ") "
        );

    BOOST_ASSERT(str.value() == " (11, 22, 33) ");
    //]
}


void sample2()
{
    //[ range_sample_2
    namespace strf = boost::stringify::v0;
    
    std::vector<int> vec = { 250, 251, 252 };

    auto str = strf::to_string()
        ( " ("
        , ~strf::hex(strf::iterate(vec, ", "))
        , ") "
        );

    BOOST_ASSERT(str.value() == " (0xfa, 0xfb, 0xfc) ");
    //]
}
