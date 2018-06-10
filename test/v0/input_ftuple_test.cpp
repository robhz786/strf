//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

int main()
{
    namespace strf = boost::stringify::v0;

    TEST("1,0,0,0,0 10000 1000000 10,000 1'0000 1'000000 10.000 1^00^00 1'000000")
        .facets(strf::monotonic_grouping<10>(1))
        ( 10000, ' '
        , strf::hex(0x10000), ' '
        , strf::oct(01000000), ' '
        , strf::facets
            ( strf::monotonic_grouping<10>(3)
            , strf::monotonic_grouping<16>(4).thousands_sep('\'')
            , strf::monotonic_grouping<8>(6).thousands_sep('\'')
            )
            ( 10000, ' '
            , strf::hex(0x10000), ' '
            , strf::oct(01000000), ' '
            , strf::facets
                ( strf::monotonic_grouping<10>(3).thousands_sep('.')
                , strf::monotonic_grouping<16>(2).thousands_sep('^')
                )
                ( 10000, ' '
                , strf::hex(0x10000), ' '
                , strf::oct(01000000)
                )
            )
        );

    static_assert
        ( strf::detail::all_are_constrainable<>::value
        , "all_are_constrainable ill implemented");

    static_assert
        ( strf::detail::all_are_constrainable
          < strf::width_calculator
          , strf::numpunct<10> >
          :: value
        , "these facets should be constrainble");

    static_assert
        ( ! strf::detail::all_are_constrainable
          < strf::encoding<char>
          , strf::width_calculator
          , strf::numpunct<10> >
          :: value
        , "encoding shall not be constrainble");

    static_assert
        ( ! strf::detail::all_are_constrainable
          < strf::width_calculator
          , strf::allow_surrogates
          , strf::numpunct<10> >
          :: value
        , "allow_surrogates shall not be constrainble");

    static_assert
        ( ! strf::detail::all_are_constrainable
          < strf::width_calculator
          , strf::numpunct<10>
          , strf::encoding_error >
          :: value
        , "encoding_error shall not be constrainble");

    int rc = report_errors() || boost::report_errors();
    return rc;
}

