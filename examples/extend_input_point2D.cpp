//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>

namespace xxx {

template <typename FloatT>
struct point2D
{
    FloatT x, y;
};

} // namespace xxx

namespace strf {

template <typename FloatT>
struct printable_def<xxx::point2D<FloatT>> {

    using representative = xxx::point2D<FloatT>;
    using forwarded_type = xxx::point2D<FloatT>;
    using format_specifiers = strf::tag<strf::alignment_format_specifier, strf::float_format_specifier>;
    using is_overridable = std::true_type;

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::value_and_format<T...> arg ) noexcept
    {
        auto p = arg.value(); // the Point2D<FloatT> value
        auto arg2 = strf::join
            ( static_cast<CharT>('(')
            , strf::fmt(p.x).set_float_format(arg.get_float_format())
            , strf::transcode(u", ")
            , strf::fmt(p.y).set_float_format(arg.get_float_format())
            , static_cast<CharT>(')') )
            .set_alignment_format(arg.get_alignment_format());
        return strf::make_printer<CharT>(pre, fp, arg2);
    }
};

} // namespace strf

#include <strf/to_string.hpp>

int main()
{
    xxx::point2D<double> pt{1.5555, 2.5555};

    // basic sample
    auto str = strf::to_string(pt);
    assert(str == "(1.5555, 2.5555)");

    // now in UTF-16
    auto u16str = strf::to_u16string(pt);
    assert(u16str == u"(1.5555, 2.5555)");

    // applying alignment
    str = strf::to_string(strf::center(pt, 30, U'_'));
    assert(str ==  "_______(1.5555, 2.5555)_______");

    // applying alignment and float formatting
    str = strf::to_string(strf::center(pt, 30, U'_').sci().p(2));
    assert(str ==  "_____(1.56e+00, 2.56e+00)_____");

    str = strf::to_string(strf::sci(pt).p(2) ^ 30);
    assert(str ==  "     (1.56e+00, 2.56e+00)     ");

    return 0;
}


