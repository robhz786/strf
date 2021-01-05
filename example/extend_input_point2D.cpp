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
struct print_traits<xxx::point2D<FloatT>> {

    using forwarded_type = xxx::point2D<FloatT>;

    using formatters = strf::tag
        < strf::alignment_formatter
        , strf::float_formatter<strf::float_notation::general> >;

    template <typename CharT, typename Preview, typename FPack>
    constexpr static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , xxx::point2D<FloatT> arg ) noexcept
    {
        return strf::make_printer_input<CharT>
            ( preview
            , fp
            , strf::join
                ( (CharT)'('
                , arg.x
                , strf::conv(u", ")
                , arg.y
                , (CharT)')' ) );
    }

    template <typename CharT, typename Preview, typename FPack, typename... T>
    constexpr static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<T...> arg ) noexcept
    {
        auto pt = arg.value(); // the Point2D<FloatT> value

        auto float_fmt = arg.get_float_format();
        auto alignment_fmt = arg.get_alignment_format();

        auto arg2 = strf::join
            ( (CharT)'('
            , strf::fmt(pt.x).set_float_format(float_fmt)
            , strf::conv(u", ")
            , strf::fmt(pt.y).set_float_format(float_fmt)
            , (CharT)')' )
            .set_alignment_format(alignment_fmt);

        return strf::make_printer_input<CharT>(preview, fp, arg2);
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


