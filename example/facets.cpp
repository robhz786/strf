//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <iostream>

void basic_facet_sample()
{

//[ basic_facet_sample
    namespace strf = boost::stringify::v0;
    constexpr int base = 10;
    auto punct = strf::str_grouping<base>{"\4\3\2"}.thousands_sep(U'.');
    auto s = strf::to_string
        .facets(punct)
        ("one hundred billions = ", 100000000000ll);

    BOOST_ASSERT(s == "one hundred billions = 1.00.00.000.0000");
//]
}


void constrained_facet()
{
    //[ constrained_facet_sample

    namespace strf = boost::stringify::v0;

    auto facet_obj = strf::constrain<std::is_signed>(strf::monotonic_grouping<10>{3});

    auto s = strf::to_string.facets(facet_obj)(100000u, "  ", 100000);

    BOOST_ASSERT(s == "100000  100,000");
    //]
}


void overriding_sample()
{
    //[ facets_overriding
    namespace strf = boost::stringify::v0;

    auto punct_dec_1 = strf::monotonic_grouping<10>{1};
    auto punct_dec_2 = strf::monotonic_grouping<10>{2}.thousands_sep('.');
    auto punct_dec_3 = strf::monotonic_grouping<10>{3}.thousands_sep('^');;

    // Below, punct_dec_3 overrides punct_dec_2, but only for signed types.
    // punct_dec_2 overrides punct_dec_1 for all input types,
    // hence the presence of punt_dec_1 bellow has no effect.

    auto s = strf::to_string
        .facets( punct_dec_1
               , punct_dec_2
               , strf::constrain<std::is_signed>(punct_dec_3) )
        ( 100000, "  ", 100000u ) ;

    BOOST_ASSERT(s == "100^000  10.00.00");
    //]
}


void get_facet_sample()
{
    //[ get_facet_sample
    namespace strf = boost::stringify::v0;

    auto punct_hex  = strf::monotonic_grouping<16>{4}.thousands_sep('\'');
    auto punct_dec  = strf::monotonic_grouping<10>{3}.thousands_sep('.');

    auto fp = strf::pack
        ( std::ref(punct_hex) // note the use of std::ref here
        , strf::constrain<strf::is_int_number>(std::ref(punct_dec)) );//and here

    decltype(auto) f1 = strf::get_facet<strf::numpunct_category<16>, int>(fp);
    BOOST_ASSERT(&f1 == &punct_hex);

    decltype(auto) f2 = strf::get_facet<strf::numpunct_category<10>, int>(fp);
    BOOST_ASSERT(&f2 == &punct_dec);

    decltype(auto) f3 = strf::get_facet<strf::numpunct_category<10>, double>(fp);
    BOOST_ASSERT(&f3 == &strf::numpunct_category<10>::get_default());
    //]
    (void)f1;
    (void)f2;
    (void)f3;
}


int main()
{
    basic_facet_sample();
    constrained_facet();
    overriding_sample();
    get_facet_sample();
    return 0;
}
