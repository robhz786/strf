//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <iostream>

void samples()
{
    //[joins_example
    namespace strf = boost::stringify::v0;

    auto str = strf::to_string
        ("---", strf::join_right(15) ("abc", "def", 123), "---");

    BOOST_ASSERT(str == "---      abcdef123---");


    str = strf::to_string
        ("---", strf::join_center(15) ("abc", "def", 123), "---");
    BOOST_ASSERT(str == "---   abcdef123   ---");


    str = strf::to_string
        ( "---"
        , strf::join_left(15, U'.') ("abc", strf::right("def", 5), 123)
        , "---" );

    BOOST_ASSERT(str == "---abc  def123....---");

    str = strf::to_string
        ( "---"
        , strf::join_split(15, '.', 1) (strf::left("abc", 5), "def", 123)
        , "---" );
    BOOST_ASSERT(str == "---abc  ....def123---");
    //]
}

//[ join_with_tr_string_part1

// in some header file

enum class language { English, Spanish, French };

language get_current_language();

const char* msg_the_ip_address_of_X_is_X();

//]

//[ join_with_tr_string_part2

// in some source file you shall not edit

const char* msg_the_ip_address_of_X_is_X()
{
    switch(get_current_language())
    {
        case language::Spanish:
            return "La direcci\u00F3n IP de {} es {}";

        case language::French:
            return "L'adresse IP de {} est {}";

        default:
            return "The IP address of {} is {}";
    }
}
//]


void sample()
{
//[ join_with_tr_string_part3

    // in your code:

    namespace strf = boost::stringify::v0;

    std::string host_name = "boost.org";
    unsigned char ip_addr [4] = {146, 20, 110, 251};

    auto str = strf::to_string .tr
        ( msg_the_ip_address_of_X_is_X()
        , host_name
        , strf::join_left(0)
            (ip_addr[0], '.', ip_addr[1], '.', ip_addr[2], '.', ip_addr[3]) );

    if (get_current_language() == language::English)
    {
        BOOST_ASSERT(str == "The IP address of boost.org is 146.20.110.251");
    }

//]
}



language get_current_language()
{
    return language::English;
}


int main()
{
    samples();
    sample();
    return 0;
}
