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

template <typename FloatT>
std::pair<FloatT, FloatT> complex_coordinates
    ( std::complex<FloatT> x, complex_form form ) noexcept
{
    FloatT first = 0.0;
    FloatT second = 0.0;
    if (form == complex_form::polar) {
        if (x.real() != 0.0) {
            first = std::sqrt(x.real() * x.real() + x.imag() * x.imag());
            second = std::atan(x.imag() / x.real());
        } else if (x.imag() != 0.0) {
            first = std::fabs(x.imag());
            second = x.imag() > 0 ? M_PI_2 : -M_PI_2;
        }
    } else {
        first = x.real();
        second = x.imag();
    }
    return {first, second};
}

namespace strf {

template <typename FloatT>
struct printable_def<std::complex<FloatT>>
{
    using representative = std::complex<FloatT>;
    using forwarded_type = std::complex<FloatT>;
    using is_overridable = std::true_type;
    using format_specifiers = strf::tag
        < std_complex_format_specifier
        , strf::float_format_specifier >;

    template <typename CharT, typename FPack>
    static void print
        ( strf::destination<CharT>& dst
        , const FPack& facets
        , std::complex<FloatT> arg)
    {
        to(dst).with(facets)
            ( (CharT)'(', arg.real()
            , strf::unsafe_transcode(", ")
            , arg.imag(), (CharT)')');
    }

    template <typename CharT, typename FPack, typename... T>
    static void print
        ( strf::destination<CharT>& dst
        , const FPack& facets
        , strf::value_and_format<printable_def, T...> arg )
    {
        auto form = arg.get_complex_form();
        auto v = ::complex_coordinates(arg.value(), form);
        const bool has_brackets = form != complex_form::polar;
        if (has_brackets) {
            to(dst) (static_cast<CharT>('('));            
        }
        to(dst).with(facets)
            ( strf::fmt(v.first).set_float_format(arg.get_float_format())
            , strf::transcode(middle_string(form), strf::utf_t<char16_t>())
            , strf::fmt(v.second).set_float_format(arg.get_float_format()) );
        if (has_brackets) {
            to(dst) (static_cast<CharT>(')'));
        }
    }

private:

    static const char16_t* middle_string(complex_form form)
    {
        switch(form) {
            case complex_form::algebric: return u" + i*";
            case complex_form::polar: return u"\u2220 "; // the angle character ∠
            default: return u", ";
        }
    }

};
} // namespace strf

//--------------------------------------------------------------------------------
//  Test
//--------------------------------------------------------------------------------

#include "../tests/test_utils.hpp"

template <typename CharT>
constexpr STRF_HD CharT char_type_of(const CharT*) { return CharT{}; }

template <typename CharT>
constexpr STRF_HD CharT char_type_of(strf::detail::simple_string_view<CharT>)
{ return CharT{}; }


#define TEST_NO_RESERVE(EXPECTED)                                           \
    strf::make_printing_syntax                                              \
    ( test_utils::input_tester_creator<decltype(char_type_of(EXPECTED))>    \
            { (EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, 1.0}  \
        , strf::no_reserve() )

template <typename T> struct is_float32: std::false_type {};
template <> struct is_float32<float>: std::true_type {};

void tests()
{
    // Using strf internal test framework ( defined in tests/test_utils.hpp )

    std::complex<double> x{3000, 4000};

    auto punct = strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130);

    TEST_NO_RESERVE(u"(3000, 4000)") (x);

    TEST_NO_RESERVE("(3000. + i*4000.)") (*strf::fmt(x).algebric());

    TEST_NO_RESERVE(u"5000.\u2220 0.9272952180016122") (*strf::fmt(x).polar());



    TEST_NO_RESERVE("(3\251E+03, 4\251E+03)")
        ( strf::iso_8859_3<char>
        , punct
        , strf::uppercase
        , * !strf::sci(x) );

    TEST_NO_RESERVE("(3\251E+03 + i*4\251E+03)")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::sci(x).algebric());

    TEST_NO_RESERVE("5\251E+03? 9\251""272952180016122E-01")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::sci(x).polar());

    TEST_NO_RESERVE("(3\251E+03, 4\251E+03)")
        .with(strf::iso_8859_3<char>, punct, strf::uppercase)
        (* !strf::sci(x).vector());


    // with punctuation

    TEST_NO_RESERVE("(1\xA9""5E+10, 2\xA9""5E+10)")
        ( punct
        , strf::uppercase, strf::iso_8859_3<char>
        , strf::punct(std::complex<double>{1.5e+10, 2.5e+10}) );

    TEST_NO_RESERVE("(1\xA9""5E+10 + i*2\xA9""5E+10)")
        ( punct
        , strf::uppercase
        , strf::iso_8859_3<char>
        , strf::punct(std::complex<double>{1.5e+10, 2.5e+10}).algebric() );

    TEST_NO_RESERVE("1\xA9""5E+10? 1\xA9""6666666666666666E-10")
        ( punct
        , strf::uppercase
        , strf::iso_8859_3<char>
        , strf::punct(std::complex<double>{1.5e+10, 2.5}).polar() );

    TEST_NO_RESERVE("(1.5E+10 + i*2.5E+10)")
        ( strf::constrain<is_float32>(punct)
        , strf::constrain<is_float32>(strf::uppercase)
        , strf::iso_8859_3<char>
        , strf::fmt(std::complex<double>{1.5e+10, 2.5e+10}).algebric() );
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
