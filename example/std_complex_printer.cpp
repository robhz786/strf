//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _USE_MATH_DEFINES
#include <strf/to_cfile.hpp>
#include <strf/to_string.hpp>
#include <utility>
#include <complex>
#include <cmath>

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
} // namespace strf;

//--------------------------------------------------------------------------------
// 2 //  Define format functions
//--------------------------------------------------------------------------------

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
// 3.1 // std_complex_printer to handle values without formatting
//--------------------------------------------------------------------------------

template <typename CharT, typename FloatT>
class std_complex_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    std_complex_printer(strf::usual_printer_input<T...> input)
        : encoding_(get_facet_<strf::char_encoding_c<CharT>>(input.fp))
        , numpunct_(strf::get_facet<strf::numpunct_c<10>, FloatT>(input.fp))
        , lettercase_(strf::get_facet<strf::lettercase_c, FloatT>(input.fp))
        , form_(get_facet_<complex_form_c>(input.fp))
        , coordinates_(::complex_coordinates(form_, input.arg))
    {
        preview_(input.preview, get_facet_<strf::width_calculator_c>(input.fp));
    }

    void print_to(strf::basic_outbuff<CharT>& dest) const override;

private:

    template <typename Category, typename FPack>
    static decltype(auto) get_facet_(const FPack& fp)
    {
        return strf::get_facet<Category, std::complex<FloatT>>(fp);
    }

    template <typename Preview, typename WCalc>
    void preview_(Preview& preview, const WCalc& wcalc) const;

    strf::dynamic_char_encoding<CharT> encoding_;
    strf::numpunct<10> numpunct_;
    strf::lettercase lettercase_;
    complex_form form_;
    std::pair<FloatT, FloatT> coordinates_;

    static constexpr char32_t anglechar_ = 0x2220;

};

template <typename CharT, typename FloatT>
template <typename Preview, typename WCalc>
void std_complex_printer<CharT, FloatT>::preview_(Preview& preview, const WCalc& wcalc) const
{
    auto facets = strf::pack(lettercase_, numpunct_, encoding_);
    strf::preview<CharT>(preview, facets, coordinates_.first, coordinates_.second);

    switch (form_) {
        case complex_form::algebric:
            preview.subtract_width(6);
            preview.add_size(6);
            break;

        case complex_form::vector:
            preview.subtract_width(4);
            preview.add_size(4);
            break;

        default:
            assert(form_ == complex_form::polar);

            preview.subtract_width(wcalc.char_width(strf::utf32<char32_t>{}, anglechar_));
            preview.subtract_width(1);

            preview.add_size(encoding_.encoded_char_size(anglechar_));
            preview.add_size(1);
    }
}

template <typename CharT, typename FloatT>
void std_complex_printer<CharT, FloatT>::print_to(strf::basic_outbuff<CharT>& dest) const
{
    auto facets = strf::pack(lettercase_, numpunct_, encoding_);
    if (form_ == complex_form::polar) {
        strf::to(dest).with(facets) ( coordinates_.first
                                , U'\u2220', static_cast<CharT>(' ')
                                , coordinates_.second);
    } else {
        const char* middle_str = ( form_ == complex_form::algebric
                                 ? " + i*"
                                 : ", " );
        strf::to(dest).with(facets)
            ( (CharT)'(', coordinates_.first
            , strf::conv(middle_str)
            , coordinates_.second, (CharT)')');
    }
}

//--------------------------------------------------------------------------------
// 3.2 // fmt_std_complex_printer to handle formatted values
//--------------------------------------------------------------------------------

template <typename CharT, typename FloatT, strf::float_notation Notation>
class fmt_std_complex_printer: public strf::printer<CharT>
{
    using complex_type_ = std::complex<FloatT>;
    static constexpr char32_t anglechar_ = 0x2220;
    static constexpr int numbase_ =
        Notation == strf::float_notation::hex ? 16 : 10;

public:

    template <typename... T>
    fmt_std_complex_printer(strf::usual_printer_input<T...> input)
        : encoding_(strf::get_facet<strf::char_encoding_c<CharT>, complex_type_>(input.fp))
        , numpunct_(strf::get_facet<strf::numpunct_c<numbase_>, FloatT>(input.fp))
        , lettercase_(strf::get_facet<strf::lettercase_c, FloatT>(input.fp))
        , float_fmt_(input.arg.get_float_format())
        , form_(input.arg.form(
                    (strf::get_facet<complex_form_c, std::complex<FloatT>>(input.fp))))
        , coordinates_{::complex_coordinates(form_, input.arg.value())}
    {
        init_and_preview_
            ( input.preview
            , input.arg.get_alignment_format()
            , strf::get_facet<strf::width_calculator_c, complex_type_>(input.fp) );
    }

    void print_to(strf::basic_outbuff<CharT>& dest) const override;

private:

    template < strf::preview_size PreviewSize, strf::preview_width PreviewWidth
             , typename WidthCalc, typename AlignmentFormat >
    void init_and_preview_
        ( strf::print_preview<PreviewSize, PreviewWidth>& preview
        , AlignmentFormat afmt
        , const WidthCalc& wcalc );

    template <strf::preview_size PreviewSize, strf::preview_width PreviewWidth>
    void init_fillcount_and_preview_
        ( strf::print_preview<PreviewSize, PreviewWidth>& preview
        , strf::width_t anglechar_width
        , strf::width_t fillchar_width
        , strf::width_t fmt_width );

    void print_complex_value_( strf::basic_outbuff<CharT>& dest ) const;
    void print_complex_value_split_( strf::basic_outbuff<CharT>& dest ) const;

    template <typename Preview>
    void preview_without_fill_(Preview& preview, strf::width_t anglechar_width) const;

    strf::dynamic_char_encoding<CharT> encoding_;
    strf::numpunct<numbase_> numpunct_;
    strf::lettercase lettercase_;
    strf::float_format<Notation> float_fmt_;
    complex_form form_;
    std::pair<FloatT, FloatT> coordinates_;
    std::uint16_t fillcount_ = 0;
    char32_t fillchar_;
    strf::text_alignment alignment_;

};


template <typename CharT, typename FloatT, strf::float_notation Notation>
template < strf::preview_size PreviewSize
         , strf::preview_width PreviewWidth
         , typename WidthCalc
         , typename AlignmentFormat >
void fmt_std_complex_printer<CharT, FloatT, Notation>::init_and_preview_
    ( strf::print_preview<PreviewSize, PreviewWidth>& preview
    , AlignmentFormat afmt
    , const WidthCalc& wcalc )
{
    fillchar_ = afmt.fill;
    alignment_ = afmt.alignment;
    strf::width_t anglechar_width =
        ( (afmt.width > 0 || (bool)PreviewWidth) && form_ == complex_form::polar
        ? wcalc.char_width(strf::utf32<char32_t>{}, anglechar_)
        : 0 );
    strf::width_t fillchar_width =
        ( afmt.width > 0 && afmt.alignment != strf::text_alignment::split
        ? wcalc.char_width(strf::utf32<char32_t>{}, afmt.fill)
        : 1 );
    init_fillcount_and_preview_(preview, anglechar_width, fillchar_width, afmt.width);
}


template <typename CharT, typename FloatT, strf::float_notation Notation>
template <strf::preview_size PreviewSize, strf::preview_width PreviewWidth>
void fmt_std_complex_printer<CharT, FloatT, Notation>::init_fillcount_and_preview_
    ( strf::print_preview<PreviewSize, PreviewWidth>& preview
    , strf::width_t anglechar_width
    , strf::width_t fillchar_width
    , strf::width_t fmt_width )
{
    if (fmt_width >= preview.remaining_width() || ! (bool)PreviewWidth ) {
        preview.clear_remaining_width();
        strf::print_preview<PreviewSize, strf::preview_width::yes> sub_preview{fmt_width};
        preview_without_fill_(sub_preview, anglechar_width);
        fillcount_ = static_cast<std::uint16_t>
            ((sub_preview.remaining_width() / fillchar_width).round());
        preview.add_size(sub_preview.accumulated_size());
    } else {
        auto previous_remaining_width = preview.remaining_width();
        preview_without_fill_(preview, anglechar_width);
        if (preview.remaining_width() > 0) {
            auto content_width = previous_remaining_width - preview.remaining_width();
            if (fmt_width > content_width) {
                fillcount_ = static_cast<std::uint16_t>
                    (((fmt_width - content_width) / fillchar_width).round());
                preview.subtract_width(fillcount_);
            }
        }
    }
    if (fillcount_ && static_cast<bool>(PreviewSize)) {
        preview.add_size(fillcount_ * encoding_.encoded_char_size(fillchar_));
    }
}

template <typename CharT, typename FloatT, strf::float_notation Notation>
template <typename Preview>
void fmt_std_complex_printer<CharT, FloatT, Notation>::preview_without_fill_
    ( Preview& preview, strf::width_t anglechar_width) const
{
    auto facets = strf::pack(lettercase_, numpunct_, encoding_);
    strf::preview<CharT>
        ( preview, facets
        , strf::fmt(coordinates_.first).set_float_format(float_fmt_)
        , strf::fmt(coordinates_.second).set_float_format(float_fmt_) ) ;

    switch (form_) {
        case complex_form::algebric:
            preview.subtract_width(7);
            preview.add_size(7);
            break;

        case complex_form::vector:
            preview.subtract_width(4);
            preview.add_size(4);
            break;

        default:
            assert(form_ == complex_form::polar);

            preview.subtract_width(anglechar_width);
            preview.subtract_width(1);

            preview.add_size(encoding_.encoded_char_size(anglechar_));
            preview.add_size(1);
    }
}


template <typename CharT, typename FloatT, strf::float_notation Notation>
void fmt_std_complex_printer<CharT, FloatT, Notation>::print_to
    ( strf::basic_outbuff<CharT>& dest ) const
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
            case strf::text_alignment::split:
                print_complex_value_split_(dest);
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

template <typename CharT, typename FloatT, strf::float_notation Notation>
void fmt_std_complex_printer<CharT, FloatT, Notation>::print_complex_value_
    ( strf::basic_outbuff<CharT>& dest ) const
{
    auto facets = strf::pack(lettercase_, numpunct_, encoding_);
    auto first_val = strf::fmt(coordinates_.first).set_float_format(float_fmt_);
    auto second_val = strf::fmt(coordinates_.second).set_float_format(float_fmt_);
    if (form_ == complex_form::polar) {
        strf::to(dest).with(facets) ( first_val, U'\u2220'
                                    , static_cast<CharT>(' '), second_val);
    } else {
        const char* middle_str = ( form_ == complex_form::algebric
                                 ? " + i*"
                                 : ", " );
        strf::to(dest).with(facets)
            ( (CharT)'(', first_val, strf::conv(middle_str)
            , second_val, (CharT)')');
    }
}

template <typename CharT, typename FloatT, strf::float_notation Notation>
void fmt_std_complex_printer<CharT, FloatT, Notation>::print_complex_value_split_
    ( strf::basic_outbuff<CharT>& dest ) const
{
    // I decided to ignore fillchar_ here and use space instead

    auto facets = strf::pack(lettercase_, numpunct_, encoding_);
    auto first_val  = strf::fmt(coordinates_.first).set_float_format(float_fmt_);
    auto second_val = strf::fmt(coordinates_.second).set_float_format(float_fmt_);
    auto halfcount = (fillcount_)/ 2;
    switch(form_) {
        case complex_form::vector:
            strf::to(dest).with(facets)
                ( (CharT)'('
                , first_val
                , strf::multi((CharT)' ', fillcount_ - halfcount)
                , (CharT)','
                , strf::multi((CharT)' ', 1 + halfcount)
                , second_val
                , (CharT)')' );
            break;

        case complex_form::polar:
            strf::to(dest).with(facets)
                ( first_val
                , strf::multi((CharT)' ', fillcount_ - halfcount)
                , U'\u2220'
                , strf::multi((CharT)' ', 1 + halfcount)
                , second_val );
            break;

        case complex_form::algebric:
            strf::to(dest).with(facets)
                ( (CharT)'('
                , first_val
                , strf::multi((CharT)' ', 1 + fillcount_ - halfcount)
                , (CharT)'+'
                , strf::multi((CharT)' ', 1 + halfcount)
                , (CharT)'i'
                , (CharT)'*'
                , second_val
                , (CharT)')' );
            break;
    }
}

//--------------------------------------------------------------------------------
// 4 // Define the PrintTraits class
//--------------------------------------------------------------------------------

namespace strf {

template <typename FloatT>
struct print_traits<std::complex<FloatT>>
{
    using facet_tag = std::complex<FloatT>;
    using forwarded_type = std::complex<FloatT>;
    using formatters = strf::tag
        < std_complex_formatter
        , strf::float_formatter<strf::float_notation::general>
        , strf::alignment_formatter >;

    template <typename CharT, typename Preview, typename FPack>
    static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , std::complex<FloatT> arg)
        -> strf::usual_printer_input
            < CharT, Preview, FPack, std::complex<FloatT>
            , std_complex_printer<CharT, FloatT> >
    {
        return {preview, fp, arg};
    }

    template < typename CharT, typename Preview, typename FPack, typename... T >
    static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<T...> arg )
        -> strf::usual_printer_input
            < CharT, Preview, FPack, decltype(arg)
            , fmt_std_complex_printer<CharT, FloatT, decltype(arg)::float_notation()> >
    {
        return {preview, fp, arg};
    }
};

} // namespace strf


//--------------------------------------------------------------------------------
// 5 // Test
//--------------------------------------------------------------------------------

template <typename T> struct is_float32: std::false_type {};
template <> struct is_float32<float>: std::true_type {};

int main()
{
    std::complex<double> x{3000, 4000};

    auto punct = strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130);

    auto u16str = strf::to_u16string(x);
    assert(u16str == u"(3000, 4000)");

    // using facets

    u16str = strf::to_u16string.with(complex_form::algebric) (x);
    assert(u16str == u"(3000 + i*4000)");

    u16str = strf::to_u16string.with(complex_form::polar) (x);
    assert(u16str == u"5000\u2220 0.9272952180016122");

    u16str = strf::to_u16string.with(complex_form::algebric, punct) (x);
    assert(u16str == u"(3\u02D9" u"000 + i*4\u02D9" u"000)");

    auto str = strf::to_string
        .with( punct, strf::uppercase, strf::iso_8859_3<char>() )
        (std::complex<double>{1.5e+10, 2.5e+10});
    assert(str == "(1\xA9""5E+10, 2\xA9""5E+10)");

    str = strf::to_string
        .with( complex_form::algebric, punct, strf::uppercase, strf::iso_8859_3<char>() )
        (std::complex<double>{1.5e+10, 2.5e+10});
    assert(str == "(1\xA9""5E+10 + i*2\xA9""5E+10)");

    str = strf::to_string
        .with( complex_form::algebric, punct, strf::uppercase, strf::iso_8859_3<char>() )
        (std::complex<double>{1.5e+4, 2.5});

    str = strf::to_string
        .with(complex_form::polar, punct, strf::uppercase, strf::iso_8859_3<char>() )
        (std::complex<double>{1.5e+10, 2.5});
        assert(str == "1\xA9""5E+10? 1\xA9""6666666666666666E-10");

    str = strf::to_string.with( complex_form::algebric
                              , strf::constrain<is_float32>(punct)
                              , strf::constrain<is_float32>(strf::uppercase)
                              , strf::iso_8859_3<char>() )
        (std::complex<double>{1.5e+10, 2.5e+10});
    assert(str == "(1.5e+10 + i*2.5e+10)");

    // using format functions

    str = strf::to_string(*strf::right(x, 30, '_'));
    assert(str == "________________(3000., 4000.)");

    str = strf::to_string(*strf::right(x, 30, '_').algebric());
    assert(str == "_____________(3000. + i*4000.)");

    u16str = strf::to_u16string(*strf::right(x, 30, '_').polar());
    assert(u16str == u"_____5000.\u2220 0.9272952180016122");


    str = strf::to_string(*strf::left(x, 30, '_'));
    assert(str == "(3000., 4000.)________________");

    str = strf::to_string(*strf::left(x, 30, '_').algebric());
    assert(str == "(3000. + i*4000.)_____________");

    u16str = strf::to_u16string(*strf::left(x, 30, '_').polar());
    assert(u16str == u"5000.\u2220 0.9272952180016122_____");


    str = strf::to_string(*strf::center(x, 30, '_'));

    assert(str == "________(3000., 4000.)________");
    str = strf::to_string(*strf::center(x, 30, '_').algebric());

    assert(str == "______(3000. + i*4000.)_______");

    u16str = strf::to_u16string(*strf::center(x, 30, '_').polar());
    assert(u16str == u"__5000.\u2220 0.9272952180016122___");


    str = strf::to_string(+*strf::split(x, 30, '_'));
    assert(str == "(+3000.       ,        +4000.)");

    str = strf::to_string(+*strf::split(x, 30, '_').algebric());
    assert(str == "(+3000.       +      i*+4000.)");

    u16str = strf::to_u16string(+*strf::split(x, 30, '_').polar());
    assert(u16str == u"+5000.  \u2220  +0.9272952180016122");


    str = strf::to_string(*strf::split(x, 0, '_'));
    assert(str == "(3000., 4000.)");

    str = strf::to_string(*strf::split(x, 0, '_').algebric());
    assert(str == "(3000. + i*4000.)");

    u16str = strf::to_u16string(*strf::split(x, 0, '_').polar());
    assert(u16str == u"5000.\u2220 0.9272952180016122");

    // using format functions and facets

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::right(x, 40, '_').sci());
    assert(str == "________________________(3\251E+03, 4\251E+03)");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::right(x, 40, '_').sci().algebric());
    assert(str == "_____________________(3\251E+03 + i*4\251E+03)");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::right(x, 40, '_').sci().polar());
    assert(str == "___________5\251E+03? 9\251272952180016122E-01");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::left(x, 40, '_').sci());
    assert(str == "(3\251E+03, 4\251E+03)________________________");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::left(x, 40, '_').sci().algebric());
    assert(str == "(3\251E+03 + i*4\251E+03)_____________________");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::left(x, 40, '_').sci().polar());
    assert(str == "5\251E+03? 9\251272952180016122E-01___________");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::center(x, 40, '_').sci());
    assert(str == "____________(3\251E+03, 4\251E+03)____________");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::center(x, 40, '_').sci().algebric());
    assert(str == "__________(3\251E+03 + i*4\251E+03)___________");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::center(x, 40, '_').sci().polar());
    assert(str == "_____5\251E+03? 9\251272952180016122E-01______");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::split(x, 40, '_').sci());
    assert(str == "(3\251E+03            ,             4\251E+03)");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::split(x, 40, '_').sci().algebric());
    assert(str == "(3\251E+03            +           i*4\251E+03)");

    str = strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::split(x, 40, '_').sci().polar());
    assert(str == "5\251E+03      ?      9\251272952180016122E-01");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::right(x, 40, '_').sci());
    assert(str == "________________________(3\251E+03, 4\251E+03)");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::right(x, 40, '_').sci().algebric());
    assert(str == "_____________________(3\251E+03 + i*4\251E+03)");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::right(x, 40, '_').sci().polar());
    assert(str == "___________5\251E+03? 9\251272952180016122E-01");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::left(x, 40, '_').sci());
    assert(str == "(3\251E+03, 4\251E+03)________________________");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::left(x, 40, '_').sci().algebric());
    assert(str == "(3\251E+03 + i*4\251E+03)_____________________");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::left(x, 40, '_').sci().polar());
    assert(str == "5\251E+03? 9\251272952180016122E-01___________");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::center(x, 40, '_').sci());
    assert(str == "____________(3\251E+03, 4\251E+03)____________");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::center(x, 40, '_').sci().algebric());
    assert(str == "__________(3\251E+03 + i*4\251E+03)___________");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::center(x, 40, '_').sci().polar());
    assert(str == "_____5\251E+03? 9\251272952180016122E-01______");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::split(x, 40, '_').sci());
    assert(str == "(3\251E+03            ,             4\251E+03)");

    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::split(x, 40, '_').sci().algebric());

    assert(str == "(3\251E+03            +           i*4\251E+03)");
    str =strf::to_string
        .with(strf::iso_8859_3<char>(), punct, strf::uppercase)
        (* strf::split(x, 40, '_').sci().polar());

    assert(str == "5\251E+03      ?      9\251272952180016122E-01");

    // preview
    {
        strf::print_size_and_width_preview pp{strf::width_max};
        strf::preview<char>(pp, strf::pack(), *strf::fmt(x));
        assert(pp.accumulated_size() == 14);
        assert(strf::width_max - pp.remaining_width() == 14);
    }
    {
        strf::print_size_and_width_preview pp{strf::width_max};
        strf::preview<char>(pp, strf::pack(), *strf::fmt(x).algebric());
        assert(pp.accumulated_size() == 17);
        assert(strf::width_max - pp.remaining_width() == 17);
    }
    {
        strf::print_size_and_width_preview pp{strf::width_max};
        strf::preview<char>(pp, strf::pack(), *strf::fmt(x).polar());
        assert(pp.accumulated_size() == 27);
        assert(strf::width_max - pp.remaining_width() == 25);
    }

    (void)str;
    (void)u16str;
    return 0;
}
