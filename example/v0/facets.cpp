#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <iostream>

void basic_facet_sample()
{
    //[ basic_facet_sample
    namespace strf = boost::stringify::v0;

    // str_grouping<10> facet belongs to numpunct_category<10> category
    strf::str_grouping<10> facet_obj{"\2\2\3"};

    auto str = strf::make_string.facets(facet_obj).exception(10000000000000LL);

    BOOST_ASSERT(str == "1,000,000,000,00,00");
    //]
}


void constrained_facet()
{
    //[ constrained_facet_sample

    namespace strf = boost::stringify::v0;

    auto facet_obj = strf::constrain<std::is_signed>(strf::monotonic_grouping<10>{3});

    auto str = strf::make_string.facets(facet_obj)("{}  {}").exception(100000u, 100000 );

    BOOST_ASSERT(str == "100000  100,000");
    //]
}


void overriding_sample()
{
    //[ facets_overriding
    namespace strf = boost::stringify::v0;

    auto punct_hex   = strf::monotonic_grouping<16>{4}.thousands_sep('\'');
    auto punct_oct   = strf::monotonic_grouping< 8>{3}.thousands_sep('_');
    auto punct_dec_1 = strf::monotonic_grouping<10>{1};
    auto punct_dec_2 = strf::monotonic_grouping<10>{2}.thousands_sep('.');
    auto punct_dec_3 = strf::monotonic_grouping<10>{3}.thousands_sep('^');;

    // punct_dec_1, punct_dec_2 and punct_dec_3 belong to the same facet category.
    // In the use below, punct_dec_3 overrides punct_dec_2, but only for signed types.
    // And punct_dec_2 overrides punct_dec_1 for all input types,
    // hence the presence of punt_dec_1 bellow has no effect.

    auto str = strf::make_string
        .facets
            ( punct_hex
            , punct_oct
            , punct_dec_1
            , punct_dec_2
            , strf::constrain<std::is_signed>(punct_dec_3)
            )
        ("{}  {}  {}  {}")
        .exception(100000, 100000u, strf::hex(100000), strf::oct(100000));

    BOOST_ASSERT(str == "100^000  10.00.00  1'86a0  303_240");
    //]
}


void get_facet_sample()
{
    //[ get_facet_sample
    namespace strf = boost::stringify::v0;

    auto punct_hex  = strf::monotonic_grouping<16>{4}.thousands_sep('\'');
    auto punct_dec  = strf::monotonic_grouping<10>{3}.thousands_sep('.');

    auto ftuple_obj = strf::make_ftuple
        ( std::ref(punct_hex) // note the use of std::ref here
        , strf::constrain<strf::is_int_number>(std::ref(punct_dec)) // and here
        );

    const auto& f1 = strf::get_facet<strf::numpunct_category<16>, int>(ftuple_obj);
    BOOST_ASSERT(&f1 == &punct_hex);

    const auto& f2 = strf::get_facet<strf::numpunct_category<10>, int>(ftuple_obj);
    BOOST_ASSERT(&f2 == &punct_dec);

    const auto& f3 = strf::get_facet<strf::numpunct_category<10>, double>(ftuple_obj);
    BOOST_ASSERT(&f3 == &strf::numpunct_category<10>::get_default());
    //]
}


int main()
{
    basic_facet_sample();
    constrained_facet();
    overriding_sample();
    get_facet_sample();
    return 0;
}
