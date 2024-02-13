//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _USE_MATH_DEFINES // NOLINT(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#include <strf/to_cfile.hpp>
#include <strf/to_string.hpp>
#include <utility>
#include <complex>
#include <cmath>

//--------------------------------------------------------------------------------
// Define format functions
//--------------------------------------------------------------------------------

enum class complex_form { vector, algebric, polar };

struct std_complex_format_specifier {

    template <class T>
    class fn
    {
    public:

        constexpr fn() noexcept = default;

        constexpr explicit fn(complex_form f) noexcept
            : form_(f)
        {
        }

        template <class U>
        constexpr explicit fn(const fn<U>& u) noexcept
            : form_(u.get_complex_form())
        {
        }

        // format functions

        constexpr T&& vector() && noexcept
        {
            form_ = complex_form::vector;
            return static_cast<T&&>(*this);
        }
        constexpr T& vector() & noexcept
        {
            form_ = complex_form::vector;
            return static_cast<T&>(*this);
        }
        constexpr T vector() const & noexcept
        {
            return T{ static_cast<const T&>(*this)
                    , strf::tag<std_complex_format_specifier> {}
                    , complex_form::vector };
        }

        constexpr T&& algebric() && noexcept
        {
            form_ = complex_form::algebric;
            return static_cast<T&&>(*this);
        }
        constexpr T& algebric() & noexcept
        {
            form_ = complex_form::algebric;
            return static_cast<T&>(*this);
        }
        constexpr T algebric() const & noexcept
        {
            return T{ static_cast<const T&>(*this)
                    , strf::tag<std_complex_format_specifier> {}
                    , complex_form::algebric };
        }

        constexpr T&& polar() && noexcept
        {
            form_ = complex_form::polar;
            return static_cast<T&&>(*this);
        }
        constexpr T& polar() & noexcept
        {
            form_ = complex_form::polar;
            return static_cast<T&>(*this);
        }
        constexpr T polar() const & noexcept
        {
            return T{ static_cast<const T&>(*this)
                    , strf::tag<std_complex_format_specifier> {}
                    , complex_form::polar };
        }

        // observers

        constexpr complex_form get_complex_form() const
        {
            return form_;
        }

    private:

        complex_form form_ = complex_form::vector;
    };
};

//--------------------------------------------------------------------------------
// Auxiliar function
//--------------------------------------------------------------------------------

template <typename FloatT>
std::pair<FloatT, FloatT> complex_coordinates
    ( complex_form form, std::complex<FloatT> x ) noexcept
{
    std::pair<FloatT, FloatT> coordinates{0, 0};
    if (form == complex_form::polar) {
        if (x.real() != 0.0) {
            coordinates.first = std::sqrt(x.real() * x.real() + x.imag() * x.imag());
            coordinates.second = std::atan(x.imag() / x.real());
        } else if (x.imag() != 0.0) {
            coordinates.first = std::fabs(x.imag());
            coordinates.second = x.imag() > 0 ? M_PI_2 : -M_PI_2;
        } else {
            return {0, 0};
        }
    } else {
        coordinates.first = x.real();
        coordinates.second = x.imag();
    }
    return coordinates;
}

//--------------------------------------------------------------------------------
// Define the printable_traits class
//--------------------------------------------------------------------------------
static constexpr char32_t anglechar = 0x2220;

template <typename FloatT>
struct complex_printer_base
{
    using category = strf::printable_overrider_c;

    using representative_type = std::complex<FloatT>;
    using forwarded_type = std::complex<FloatT>;
    using format_specifiers = strf::tag
        < std_complex_format_specifier
        , strf::float_format_specifier
        , strf::alignment_format_specifier >;

    template <typename CharT, typename PreMeasurements, typename FPack>
    static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , std::complex<FloatT> arg)
    {
        pre->add_width(4);
        pre->add_size(4);

        const auto write_real_coord = strf::make_printer<CharT>(pre, facets, arg.real());
        const auto write_imag_coord = strf::make_printer<CharT>(pre, facets, arg.imag());

        return [write_real_coord, write_imag_coord] (strf::destination<CharT>& dst)
               {
                   strf::to(dst) ((CharT)'(');
                   write_real_coord(dst);
                   strf::to(dst) ((CharT)',', (CharT)' ');
                   write_imag_coord(dst);
                   strf::to(dst) ((CharT)')');
               };
    }

    template < typename CharT, typename PreMeasurements, typename FPack
             , typename PrintableDef, typename FloatFmt >
    static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , const strf::value_and_format
            < PrintableDef
            , std_complex_format_specifier
            , FloatFmt
            , strf::alignment_format_specifier_q<false> >& arg )
    {
        const auto form = arg.get_complex_form();
        measure_without_coordinates<CharT>(pre, facets, form);

        const auto coordinates = complex_coordinates(form, arg.value());
        const auto float_fmt = arg.get_float_format();
        const auto coord1 = strf::fmt(coordinates.first).set_float_format(float_fmt);
        const auto coord2 = strf::fmt(coordinates.second).set_float_format(float_fmt);
        const auto write_coord1 = strf::make_printer<CharT>(pre, facets, coord1);
        const auto write_coord2 = strf::make_printer<CharT>(pre, facets, coord2);
        const auto charset = strf::use_facet<strf::charset_c<CharT>, representative_type>(facets);

        return [charset, form, write_coord1, write_coord2] (strf::destination<CharT>& dst)
            {
                switch (form) {
                case complex_form::polar:
                    write_coord1(dst);
                    to(dst) (charset, anglechar, static_cast<CharT>(' '));
                    write_coord2(dst);
                    break;

                case complex_form::algebric:
                    to(dst) (static_cast<CharT>('('));
                    write_coord1(dst);
                    to(dst) (charset, strf::unsafe_transcode(" + i*"));
                    write_coord2(dst);
                    to(dst) (static_cast<CharT>(')'));
                    break;

                default:
                    assert(form == complex_form::vector);
                    to(dst) (static_cast<CharT>('('));
                    write_coord1(dst);
                    to(dst) (charset, strf::unsafe_transcode(", "));
                    write_coord2(dst);
                    to(dst) (static_cast<CharT>(')'));
                }
            };
    }


    template < typename CharT, typename PreMeasurements, typename FPack>
    static void measure_without_coordinates
        ( PreMeasurements* pre
        , const FPack& facets
        , complex_form form )
    {
        switch (form) {
            case complex_form::algebric:
                pre->add_width(7);
                pre->add_size(7);
                break;
            case complex_form::vector:
                pre->add_width(4);
                pre->add_size(4);
                break;
            default:
                assert(form == complex_form::polar);
                using rt = representative_type;
                if (pre->has_remaining_width()) {
                    auto wcalc = strf::use_facet<strf::width_calculator_c, rt>(facets);
                    pre->add_width(wcalc.char_width(strf::utf_t<char32_t>{}, anglechar));
                    pre->add_width(1);
                }
                if (PreMeasurements::size_demanded) {
                    auto encoding = strf::use_facet<strf::charset_c<CharT>, rt>(facets);
                    pre->add_size(encoding.encoded_char_size(anglechar));
                    pre->add_size(1);
                }
        }
    }
};


namespace strf {

template <typename FloatT>
struct printable_def<std::complex<FloatT>> : complex_printer_base<FloatT>
{
    template <typename T>
    using is_complex = std::is_same<T, std::complex<FloatT>>;

    using complex_printer_base<FloatT>::make_printer;

    template < typename CharT, typename PreMeasurements, typename FPack, typename FloatFmt >
    static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , const strf::value_and_format
            < printable_def
            , std_complex_format_specifier
            , FloatFmt
            , strf::alignment_format_specifier_q<true> >& arg )
    {
        using base = complex_printer_base<FloatT>;
        const auto overrider = strf::constrain<is_complex>(base());

        return strf::make_printer<CharT>
            ( pre
            , strf::pack(facets, overrider)
            , strf::join(arg.clear_alignment_format())
                .set_alignment_format(arg.get_alignment_format()) );
    }
};

} // namespace strf

//--------------------------------------------------------------------------------
//  Test
//--------------------------------------------------------------------------------

#include "../tests/test_utils.hpp"

template <typename T> struct is_float32: std::false_type {};
template <> struct is_float32<float>: std::true_type {};

void tests()
{
    // Using strf internal test framework ( defined in tests/test_utils.hpp )

    std::complex<double> x{3000, 4000};

    auto punct = strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130);

    TEST(u"(3000, 4000)") (x);

    TEST("________________(3000., 4000.)") (*strf::right(x, 30, '_'));

    TEST("_____________(3000. + i*4000.)") (*strf::right(x, 30, '_').algebric());

    TEST(u"_____5000.\u2220 0.9272952180016122") (*strf::right(x, 30, '_').polar());


    TEST("(3000., 4000.)________________") (*strf::left(x, 30, '_'));

    TEST("(3000. + i*4000.)_____________") (*strf::left(x, 30, '_').algebric());

    TEST(u"5000.\u2220 0.9272952180016122_____") (*strf::left(x, 30, '_').polar());


    TEST("________(3000., 4000.)________") (*strf::center(x, 30, '_'));

    TEST("______(3000. + i*4000.)_______") (*strf::center(x, 30, '_').algebric());

    TEST(u"__5000.\u2220 0.9272952180016122___") (*strf::center(x, 30, '_').polar());

    TEST("________________________(3\251E+03, 4\251E+03)")
        ( strf::iso_8859_3<char>
        , punct
        , strf::uppercase
        , * !strf::right(x, 40, '_').sci() );

    TEST("_____________________(3\251E+03 + i*4\251E+03)")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::right(x, 40, '_').sci().algebric());

    TEST("___________5\251E+03? 9\251""272952180016122E-01")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::right(x, 40, '_').sci().polar());

    TEST("(3\251E+03, 4\251E+03)________________________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::left(x, 40, '_').sci());

    TEST("(3\251E+03 + i*4\251E+03)_____________________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::left(x, 40, '_').sci().algebric());

    TEST("5\251E+03? 9\251""272952180016122E-01___________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::left(x, 40, '_').sci().polar());

    TEST("____________(3\251E+03, 4\251E+03)____________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::center(x, 40, '_').sci());

    TEST("__________(3\251E+03 + i*4\251E+03)___________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::center(x, 40, '_').sci().algebric());

    TEST("_____5\251E+03? 9\251""272952180016122E-01______")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::center(x, 40, '_').sci().polar());

    TEST("________________________(3\251E+03, 4\251E+03)")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::right(x, 40, '_').sci());

    TEST("_____________________(3\251E+03 + i*4\251E+03)")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::right(x, 40, '_').sci().algebric());

    TEST("___________5\251E+03? 9\251""272952180016122E-01")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::right(x, 40, '_').sci().polar());

    TEST("(3\251E+03, 4\251E+03)________________________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::left(x, 40, '_').sci());

    TEST("(3\251E+03 + i*4\251E+03)_____________________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::left(x, 40, '_').sci().algebric());

    TEST("5\251E+03? 9\251""272952180016122E-01___________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::left(x, 40, '_').sci().polar());

    TEST("____________(3\251E+03, 4\251E+03)____________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::center(x, 40, '_').sci());

    TEST("__________(3\251E+03 + i*4\251E+03)___________")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::center(x, 40, '_').sci().algebric());

    TEST("_____5\251E+03? 9\251""272952180016122E-01______")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::center(x, 40, '_').sci().polar());

    // with punctuation

    TEST("(1\xA9""5E+10, 2\xA9""5E+10)")
        ( punct
        , strf::uppercase, strf::iso_8859_3<char>
        , strf::punct(std::complex<double>{1.5e+10, 2.5e+10}) );

    TEST("(1\xA9""5E+10 + i*2\xA9""5E+10)")
        ( punct
        , strf::uppercase
        , strf::iso_8859_3<char>
        , strf::punct(std::complex<double>{1.5e+10, 2.5e+10}).algebric() );

    TEST("1\xA9""5E+10? 1\xA9""6666666666666666E-10")
        ( punct
        , strf::uppercase
        , strf::iso_8859_3<char>
        , strf::punct(std::complex<double>{1.5e+10, 2.5}).polar() );

    TEST("(1.5E+10 + i*2.5E+10)")
        ( strf::constrain<is_float32>(punct)
        , strf::constrain<is_float32>(strf::uppercase)
        , strf::iso_8859_3<char>
        , strf::fmt(std::complex<double>{1.5e+10, 2.5e+10}).algebric() );

    // size and width pre-calculation
    {
        strf::full_premeasurements pre;
        strf::measure<char>(&pre, strf::pack(), *strf::fmt(x));
        TEST_EQ(pre.accumulated_ssize(), 14);
        TEST_TRUE(pre.accumulated_width() == 14);
    }
    {
        strf::full_premeasurements pre;
        strf::measure<char>(&pre, strf::pack(), *strf::fmt(x).algebric());
        TEST_EQ(pre.accumulated_ssize(), 17);
        TEST_TRUE(pre.accumulated_width() == 17);
    }
    {
        strf::full_premeasurements pre;
        strf::measure<char>(&pre, strf::pack(), *strf::fmt(x).polar());
        TEST_EQ(pre.accumulated_ssize(), 27);
        TEST_TRUE(pre.accumulated_width() == 25);
    }
}

int main()
{
    strf::narrow_cfile_writer<char, 512> test_msg_dst(stdout);
    const test_utils::test_messages_destination_guard g(test_msg_dst);

    tests();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_msg_dst, "All test passed!\n");
    } else {
        strf::to(test_msg_dst) (err_count, " tests failed!\n");
    }
    return err_count;
}
