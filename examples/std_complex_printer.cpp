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

#include "../tests/test_utils.hpp"

//--------------------------------------------------------------------------------
// 1 //  Define complex_form facet
//--------------------------------------------------------------------------------

enum class complex_form { vector, algebric, polar };

struct complex_form_c {
    static constexpr complex_form get_default() noexcept {
        return complex_form::vector;
    }
};

namespace strf {
template <> struct facet_traits<complex_form> {
    using category = complex_form_c;
};
} // namespace strf

//--------------------------------------------------------------------------------
// 2 //  Define format functions
//--------------------------------------------------------------------------------

struct std_complex_formatter {

    enum class complex_form_fmt {
        vector   = static_cast<int>(complex_form::vector),
        algebric = static_cast<int>(complex_form::algebric),
        polar    = static_cast<int>(complex_form::polar),
        use_facet = 1 + (std::max)({vector, algebric, polar})
    };

    template <class T>
    class fn
    {
    public:

        constexpr fn() noexcept = default;

        constexpr explicit fn(complex_form_fmt f) noexcept
            : form_(f)
        {
        }

        template <class U>
        constexpr explicit fn(const fn<U>& u) noexcept
            : form_(u.form())
        {
        }

        // format functions

        constexpr T&& vector() && noexcept
        {
            form_ = complex_form_fmt::vector;
            return static_cast<T&&>(*this);
        }
        constexpr T& vector() & noexcept
        {
            form_ = complex_form_fmt::vector;
            return static_cast<T&>(*this);
        }
        constexpr T vector() const & noexcept
        {
            return T{ static_cast<const T&>(*this)
                    , strf::tag<std_complex_formatter> {}
                    , complex_form_fmt::vector };
        }

        constexpr T&& algebric() && noexcept
        {
            form_ = complex_form_fmt::algebric;
            return static_cast<T&&>(*this);
        }
        constexpr T& algebric() & noexcept
        {
            form_ = complex_form_fmt::algebric;
            return static_cast<T&>(*this);
        }
        constexpr T algebric() const & noexcept
        {
            return T{ static_cast<const T&>(*this)
                    , strf::tag<std_complex_formatter> {}
                    , complex_form_fmt::algebric };
        }

        constexpr T&& polar() && noexcept
        {
            form_ = complex_form_fmt::polar;
            return static_cast<T&&>(*this);
        }
        constexpr T& polar() & noexcept
        {
            form_ = complex_form_fmt::polar;
            return static_cast<T&>(*this);
        }
        constexpr T polar() const & noexcept
        {
            return T{ static_cast<const T&>(*this)
                    , strf::tag<std_complex_formatter> {}
                    , complex_form_fmt::polar };
        }

        // observers

        constexpr complex_form form(complex_form f) const
        {
            return form_ == complex_form_fmt::use_facet ? f : static_cast<complex_form>(form_);
        }
        constexpr complex_form_fmt form() const
        {
            return form_;
        }

    private:

        complex_form_fmt form_ = complex_form_fmt::use_facet;
    };
};

//--------------------------------------------------------------------------------
// 3 // Define printer classes
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
// 3.1 // std_complex_stringifier to handle values without formatting
//--------------------------------------------------------------------------------

template <typename CharT, typename FloatT>
class std_complex_stringifier: public strf::stringifier<CharT>
{
public:

    template <typename... T>
    explicit std_complex_stringifier(strf::usual_stringifier_input<T...> x);

    void print_to(strf::destination<CharT>& dest) const override;

private:

    template <typename PrePrinting, typename WidthCalc>
    void preprinting_(PrePrinting& pre, const WidthCalc& wcalc) const;

    strf::dynamic_charset<CharT> encoding_;
    strf::numpunct<10> numpunct_;
    strf::lettercase lettercase_;
    complex_form form_;
    std::pair<FloatT, FloatT> coordinates_;

    static constexpr char32_t anglechar_ = 0x2220;

};

template <typename CharT, typename FloatT>
template <typename... T>
inline std_complex_stringifier<CharT, FloatT>::std_complex_stringifier
    ( strf::usual_stringifier_input<T...> x )
    : encoding_(strf::use_facet<strf::charset_c<CharT>, FloatT>(x.facets))
    , numpunct_(strf::use_facet<strf::numpunct_c<10>, FloatT>(x.facets))
    , lettercase_(strf::use_facet<strf::lettercase_c, FloatT>(x.facets))
    , form_(strf::use_facet<complex_form_c, std::complex<FloatT>>(x.facets))
    , coordinates_(::complex_coordinates(form_, x.arg))
{
    preprinting_( x.pre
            , strf::use_facet<strf::width_calculator_c, std::complex<FloatT>>(x.facets) );
}

template <typename CharT, typename FloatT>
template <typename PrePrinting, typename WidthCalc>
void std_complex_stringifier<CharT, FloatT>::preprinting_(PrePrinting& pp, const WidthCalc& wcalc) const
{
    switch (form_) {
        case complex_form::algebric:
            pp.subtract_width(7);
            pp.add_size(7);
            break;

        case complex_form::vector:
            pp.subtract_width(4);
            pp.add_size(4);
            break;

        default:
            assert(form_ == complex_form::polar);
            if (pp.remaining_width() > 0) {
                pp.subtract_width(wcalc.char_width(strf::utf_t<char32_t>{}, anglechar_));
                pp.subtract_width(1);
            }
            pp.add_size(encoding_.encoded_char_size(anglechar_));
            pp.add_size(1);
    }

    auto facets = strf::pack(lettercase_, numpunct_, encoding_);
    strf::precalculate<CharT>(pp, facets, coordinates_.first, coordinates_.second);
}

template <typename CharT, typename FloatT>
void std_complex_stringifier<CharT, FloatT>::print_to(strf::destination<CharT>& dest) const
{
    auto print = strf::to(dest).with(lettercase_, numpunct_, encoding_);
    if (form_ == complex_form::polar) {
        print(coordinates_.first, U'\u2220', static_cast<CharT>(' ') );
        print(coordinates_.second );
    } else {
        print(static_cast<CharT>('('), coordinates_.first);
        print(strf::conv(form_ == complex_form::algebric ? " + i*" : ", ") );
        print(coordinates_.second, static_cast<CharT>(')'));
    }
}

//--------------------------------------------------------------------------------
// 3.2 // fmt_std_complex_stringifier to handle formatted values
//--------------------------------------------------------------------------------

template <typename CharT, typename FloatT>
class fmt_std_complex_stringifier: public strf::stringifier<CharT>
{
    using complex_type_ = std::complex<FloatT>;
    static constexpr char32_t anglechar_ = 0x2220;

public:

    template <typename... T>
    explicit fmt_std_complex_stringifier(strf::usual_stringifier_input<T...> x)
        : encoding_(strf::use_facet<strf::charset_c<CharT>, complex_type_>(x.facets))
        , numpunct10_(strf::use_facet<strf::numpunct_c<10>, FloatT>(x.facets))
        , numpunct16_(strf::use_facet<strf::numpunct_c<16>, FloatT>(x.facets))
        , lettercase_(strf::use_facet<strf::lettercase_c, FloatT>(x.facets))
        , float_fmt_(x.arg.get_float_format())
        , form_(x.arg.form(
                    (strf::use_facet<complex_form_c, std::complex<FloatT>>(x.facets))))
        , coordinates_{::complex_coordinates(form_, x.arg.value())}
        , fillchar_(x.arg.get_alignment_format().fill)
        , alignment_(x.arg.get_alignment_format().alignment)
    {
        init_fillcount_and_do_preprinting_
            ( x.pre
            , strf::use_facet<strf::width_calculator_c, complex_type_>(x.facets)
            , x.arg.width() );
    }

    void print_to(strf::destination<CharT>& dest) const override;

private:

    template < strf::precalc_size PrecalcSize
             , strf::precalc_width PrecalcWidth
             , typename WidthCalc >
    void init_fillcount_and_do_preprinting_
        ( strf::preprinting<PrecalcSize, PrecalcWidth>& pre
        , WidthCalc wcalc
        , strf::width_t fmt_width );

    void print_complex_value_( strf::destination<CharT>& dest ) const;

    template <typename PrePrinting, typename WidthCalc>
    void do_preprinting_without_fill_(PrePrinting& pre, WidthCalc wcalc) const;

    strf::dynamic_charset<CharT> encoding_;
    strf::numpunct<10> numpunct10_;
    strf::numpunct<16> numpunct16_;
    strf::lettercase lettercase_;
    strf::float_format float_fmt_;
    complex_form form_;
    std::pair<FloatT, FloatT> coordinates_;
    std::uint16_t fillcount_ = 0;
    char32_t fillchar_;
    strf::text_alignment alignment_;

};

template <typename CharT, typename FloatT>
template <strf::precalc_size PrecalcSize, strf::precalc_width PrecalcWidth, typename WidthCalc>
void fmt_std_complex_stringifier<CharT, FloatT>::init_fillcount_and_do_preprinting_
    ( strf::preprinting<PrecalcSize, PrecalcWidth>& pre
    , WidthCalc wcalc
    , strf::width_t fmt_width )
{
    strf::width_t fillchar_width = wcalc.char_width(strf::utf_t<char32_t>{}, fillchar_);
    if (fmt_width >= pre.remaining_width() || ! static_cast<bool>(PrecalcWidth) ) {
        pre.clear_remaining_width();
        strf::preprinting<PrecalcSize, strf::precalc_width::yes> sub_pre{fmt_width};
        do_preprinting_without_fill_(sub_pre, wcalc);
        fillcount_ = static_cast<std::uint16_t>
            ((sub_pre.remaining_width() / fillchar_width).round());
        pre.add_size(sub_pre.accumulated_size());
    } else {
        auto previous_remaining_width = pre.remaining_width();
        do_preprinting_without_fill_(pre, wcalc);
        if (pre.remaining_width() > 0) {
            auto content_width = previous_remaining_width - pre.remaining_width();
            if (fmt_width > content_width) {
                fillcount_ = static_cast<std::uint16_t>
                    (((fmt_width - content_width) / fillchar_width).round());
                pre.subtract_width(fillcount_);
            }
        }
    }
    if (fillcount_ && static_cast<bool>(PrecalcSize)) {
        pre.add_size(fillcount_ * encoding_.encoded_char_size(fillchar_));
    }
}

template <typename CharT, typename FloatT>
template <typename PrePrinting, typename WidthCalc>
void fmt_std_complex_stringifier<CharT, FloatT>::do_preprinting_without_fill_
    ( PrePrinting& pp, WidthCalc wcalc) const
{
    auto facets = strf::pack(wcalc, lettercase_, numpunct10_, numpunct16_, encoding_);
    strf::precalculate<CharT>
        ( pp, facets
        , strf::fmt(coordinates_.first).set_float_format(float_fmt_)
        , strf::fmt(coordinates_.second).set_float_format(float_fmt_) ) ;

    switch (form_) {
        case complex_form::algebric:
            pp.subtract_width(7);
            pp.add_size(7);
            break;

        case complex_form::vector:
            pp.subtract_width(4);
            pp.add_size(4);
            break;

        default:
            assert(form_ == complex_form::polar);
            if (pp.remaining_width() > 0) {
                pp.subtract_width(wcalc.char_width(strf::utf_t<char32_t>{}, anglechar_));
                pp.subtract_width(1);
            }
            pp.add_size(encoding_.encoded_char_size(anglechar_));
            pp.add_size(1);
    }
}

template <typename CharT, typename FloatT>
void fmt_std_complex_stringifier<CharT, FloatT>::print_to
    ( strf::destination<CharT>& dest ) const
{
    if (fillcount_ == 0) {
        print_complex_value_(dest);
    } else {
        switch (alignment_) {
            case strf::text_alignment::left:
                print_complex_value_(dest);
                encoding_.encode_fill(dest, fillcount_, fillchar_);
                break;
            case strf::text_alignment::right:
                encoding_.encode_fill(dest, fillcount_, fillchar_);
                print_complex_value_(dest);
                break;
            default: {
                assert(alignment_ == strf::text_alignment::center);
                auto halfcount = fillcount_ / 2;
                encoding_.encode_fill(dest, halfcount, fillchar_);
                print_complex_value_(dest);
                encoding_.encode_fill(dest, fillcount_ - halfcount, fillchar_);
            }
        }
    }
}

template <typename CharT, typename FloatT>
void fmt_std_complex_stringifier<CharT, FloatT>::print_complex_value_
    ( strf::destination<CharT>& dest ) const
{
    auto facets = strf::pack(lettercase_, numpunct10_, numpunct16_, encoding_);
    auto first_val = strf::fmt(coordinates_.first).set_float_format(float_fmt_);
    auto second_val = strf::fmt(coordinates_.second).set_float_format(float_fmt_);
    if (form_ == complex_form::polar) {
        strf::to(dest).with(facets)
            ( first_val, U'\u2220', static_cast<CharT>(' '), second_val);
    } else {
        const char* middle_str = ( form_ == complex_form::algebric
                                 ? " + i*"
                                 : ", " );
        strf::to(dest).with(facets)
            ( static_cast<CharT>('(')
            , first_val, strf::conv(middle_str), second_val
            , static_cast<CharT>(')') );
    }
}

//--------------------------------------------------------------------------------
// 4 // Define the PrintingTraits class
//--------------------------------------------------------------------------------

namespace strf {

template <typename FloatT>
struct printable_traits<std::complex<FloatT>>
{
    using representative_type = std::complex<FloatT>;
    using forwarded_type = std::complex<FloatT>;
    using formatters = strf::tag
        < std_complex_formatter
        , strf::float_formatter
        , strf::alignment_formatter >;

    template <typename CharT, typename PrePrinting, typename FPack>
    static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , std::complex<FloatT> arg)
        -> strf::usual_stringifier_input
            < CharT, PrePrinting, FPack, std::complex<FloatT>
            , std_complex_stringifier<CharT, FloatT> >
    {
        return {pre, fp, arg};
    }

    template < typename CharT, typename PrePrinting, typename FPack, typename... T >
    static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , strf::value_with_formatters<T...> arg )
        -> strf::usual_stringifier_input
            < CharT, PrePrinting, FPack, strf::value_with_formatters<T...>
            , fmt_std_complex_stringifier<CharT, FloatT> >
    {
        return {pre, fp, arg};
    }
};

} // namespace strf

//--------------------------------------------------------------------------------
// 5 // Test
//--------------------------------------------------------------------------------

template <typename T> struct is_float32: std::false_type {};
template <> struct is_float32<float>: std::true_type {};

void tests()
{
    // Using strf internal test framework ( defined in tests/test_utils.hpp )

    std::complex<double> x{3000, 4000};

    auto punct = strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130);

    TEST(u"(3000, 4000)") (x);

    // using facets

    TEST(u"(3000 + i*4000)") .with(complex_form::algebric) (x);

    TEST(u"5000\u2220 0.9272952180016122") .with(complex_form::polar) (x);

    TEST(u"(3\u02D9" u"000 + i*4\u02D9" u"000)") .with(complex_form::algebric, punct)
        (strf::punct(x));

    TEST("(1\xA9""5E+10, 2\xA9""5E+10)")
        .with( punct, strf::uppercase, strf::iso_8859_3<char> )
        (strf::punct(std::complex<double>{1.5e+10, 2.5e+10}));

    TEST("(1\xA9""5E+10 + i*2\xA9""5E+10)")
        .with( complex_form::algebric, punct, strf::uppercase, strf::iso_8859_3<char> )
        (strf::punct(std::complex<double>{1.5e+10, 2.5e+10}));

    TEST("1\xA9""5E+10? 1\xA9""6666666666666666E-10")
        .with(complex_form::polar, punct, strf::uppercase, strf::iso_8859_3<char> )
        (strf::punct(std::complex<double>{1.5e+10, 2.5}));

    TEST("(1.5e+10 + i*2.5e+10)") .with( complex_form::algebric
                              , strf::constrain<is_float32>(punct)
                              , strf::constrain<is_float32>(strf::uppercase)
                              , strf::iso_8859_3<char> )
        (std::complex<double>{1.5e+10, 2.5e+10});

    // using format functions

    TEST("________________(3000., 4000.)") (*strf::right(x, 30, '_'));

    TEST("_____________(3000. + i*4000.)") (*strf::right(x, 30, '_').algebric());

    TEST(u"_____5000.\u2220 0.9272952180016122") (*strf::right(x, 30, '_').polar());


    TEST("(3000., 4000.)________________") (*strf::left(x, 30, '_'));

    TEST("(3000. + i*4000.)_____________") (*strf::left(x, 30, '_').algebric());

    TEST(u"5000.\u2220 0.9272952180016122_____") (*strf::left(x, 30, '_').polar());


    TEST("________(3000., 4000.)________") (*strf::center(x, 30, '_'));

    TEST("______(3000. + i*4000.)_______") (*strf::center(x, 30, '_').algebric());

    TEST(u"__5000.\u2220 0.9272952180016122___") (*strf::center(x, 30, '_').polar());

    // using format functions and facets

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

    // size and width pre-calculation
    {
        strf::full_preprinting pp{strf::width_max};
        strf::precalculate<char>(pp, strf::pack(), *strf::fmt(x));
        TEST_EQ(pp.accumulated_size(), 14);
        TEST_TRUE(strf::width_max - pp.remaining_width() == 14);
    }
    {
        strf::full_preprinting pp{strf::width_max};
        strf::precalculate<char>(pp, strf::pack(), *strf::fmt(x).algebric());
        TEST_EQ(pp.accumulated_size(), 17);
        TEST_TRUE(strf::width_max - pp.remaining_width() == 17);
    }
    {
        strf::full_preprinting pp{strf::width_max};
        strf::precalculate<char>(pp, strf::pack(), *strf::fmt(x).polar());
        TEST_EQ(pp.accumulated_size(), 27);
        TEST_TRUE(strf::width_max - pp.remaining_width() == 25);
    }
}

int main()
{
    strf::narrow_cfile_writer<char, 512> test_msg_dest(stdout);
    test_utils::test_messages_destination_guard g(test_msg_dest);

    tests();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_msg_dest, "All test passed!\n");
    } else {
        strf::to(test_msg_dest) (err_count, " tests failed!\n");
    }
    return err_count;
}
