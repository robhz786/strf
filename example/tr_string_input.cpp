//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


void sample()
{
    //[ tr_string_as_input
    auto result = strf::to_string.tr( "{} --- {} --- {}"
                                    , "aaa"
                                    , strf::as("( {} {} )", "bbb", "ccc")
                                    , "ddd" );

    assert(result == "aaa --- ( bbb ccc ) --- ddd");
    //]
}

