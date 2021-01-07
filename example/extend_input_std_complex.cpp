//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _USE_MATH_DEFINES
#include <strf/to_cfile.hpp>
#include <strf/to_string.hpp>
#include <utility>
#include <complex>
#include <cmath>

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
} // namespace strf;

struct std_complex_formatter {

    enum class complex_form_fmt {
        vector   = (int)complex_form::vector,
        algebric = (int)complex_form::algebric,
        polar    = (int)complex_form::polar,
        use_facet = 1 + std::max({vector, algebric, polar})
    };

    template <class T>
    class fn
    {
    public:

        constexpr fn() noexcept = default;

        constexpr fn(complex_form_fmt f) noexcept
            : form_(f)
        {
        }

        template <class U>
        constexpr fn(const fn<U>& u) noexcept
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
struct print_traits<std::complex<FloatT>>
{
    using override_tag = std::complex<FloatT>;
    using forwarded_type = std::complex<FloatT>;
    using formatters = strf::tag
        < std_complex_formatter
        , strf::alignment_formatter
        , strf::float_formatter<strf::float_notation::general> >;

    // template <typename CharT, typename Preview, typename FPack>
    // static auto make_printer_input
    //     ( Preview& preview
    //     , const FPack& fp
    //     , std::complex<FloatT> arg)
    // {
    //     auto form = strf::get_facet<complex_form_c, std::complex<FloatT>>(fp);
    //     auto v = ::complex_coordinates(arg, form);
    //     unsigned has_brackets = form != complex_form::polar;
    //     auto arg2 = strf::join
    //         ( strf::multi((CharT)'(', has_brackets)
    //         , v.first
    //         , strf::conv(middle_string(form), strf::utf16<char16_t>())
    //         , v.second
    //         , strf::multi((CharT)')', has_brackets) );

    //     return strf::make_default_printer_input<CharT>(preview, fp, arg2);
    // }

    template <typename CharT, typename Preview, typename FPack, typename... T>
    static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<T...> arg )
    {
        auto form = arg.form(strf::get_facet<complex_form_c, std::complex<FloatT>>(fp));
        auto v = ::complex_coordinates(arg.value(), form);
        unsigned has_brackets = form != complex_form::polar;
        auto arg2 = strf::join
            ( strf::multi((CharT)'(', has_brackets)
            , strf::fmt(v.first).set_float_format(arg.get_float_format())
            , strf::conv(middle_string(form), strf::utf16<char16_t>())
            , strf::fmt(v.second).set_float_format(arg.get_float_format())
            , strf::multi((CharT)')', has_brackets) );
        auto arg3 = arg2.set_alignment_format(arg.get_alignment_format());
        return strf::make_default_printer_input<CharT>(preview, fp, arg3);
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

int main()
{
    std::complex<double> x{3, 4};

    auto str = strf::to_string(x);
    assert(str == "(3, 4)");

    // using facets
    str = strf::to_string.with(complex_form::algebric) (x);
    assert(str == "(3 + i*4)");

    // using format function
    auto u16str = strf::to_u16string.with(complex_form::algebric)
        ( x, u" == ", strf::sci(x).p(5).polar() );
    assert(u16str == u"(3 + i*4) == 5.00000e+00\u2220 9.27295e-01");

    // format functions on const
    const auto f1 = strf::fmt(std::complex<double>(x));
    auto f2 = f1.algebric();
    assert(f2.form() == std_complex_formatter::complex_form_fmt::algebric);
    str = strf::to_string(f2);
    assert(str == "(3 + i*4)");

    (void)str;
    (void)u16str;
    return 0;
}
