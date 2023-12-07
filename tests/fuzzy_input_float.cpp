//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "boost/current_function.hpp"

void on_strf_assert_failed(const char* file, int line, const char* func, const char* expr);
#define STRF_ASSERT(EXPR)                 \
    do {                                  \
        if (!(EXPR)) {                    \
            on_strf_assert_failed(__FILE__, __LINE__, BOOST_CURRENT_FUNCTION, #EXPR); \
        } \
    } while(false);


#define _CRT_SECURE_NO_WARNINGS // NOLINT(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#include <strf/to_cfile.hpp>
#include <ryu/ryu.h>
#include "test_utils/test_recycling.hpp"
#include <random>
#include <ctime>

#if defined(_WIN32)
#define LOCALE_NAME(lang, region) #lang "-" #region
#else
#define LOCALE_NAME(lang, region) #lang "_" #region
#endif

namespace {

#if __cpp_exceptions
class StrAssertionFailed: public std::exception
{
public:
    StrAssertionFailed() = default;
};
#endif

template <typename T>
T clone(const T& x) {
    return x;
}

template <typename FloatT>
struct floating_point_traits;

// template <>
// struct floating_point_traits<float>
// {
//     using uint_equiv = unsigned;
//     static constexpr int exponent_bits_size = 8;
//     static constexpr int mantissa_bits_size = 23;
//     static constexpr uint_equiv mantissa_bits_mask = 0x7FFFFF;
// };

template <>
struct floating_point_traits<double>
{
    using uint_equiv = std::uint64_t;
    static constexpr unsigned exponent_bits_size = 11;
    static constexpr unsigned mantissa_bits_size = 52;
    static constexpr uint_equiv mantissa_bits_mask = 0xFFFFFFFFFFFFFULL;
};

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
inline FloatT make_float
    ( typename helper::uint_equiv ieee_exponent
    , typename helper::uint_equiv ieee_mantissa
    , bool negative = false )
{
    const typename helper::uint_equiv sign = negative;
    auto v = (sign << (helper::mantissa_bits_size + helper::exponent_bits_size))
           | (ieee_exponent << helper::mantissa_bits_size)
           | (ieee_mantissa & helper::mantissa_bits_mask);

    return strf::detail::bit_cast<FloatT>(v);
}

struct measure_and_print_result {
    int count = 0;
    int predicted_size = 0;
    int predicted_width = 0;
};

template <typename Arg>
measure_and_print_result measure_and_print(char* buff, std::ptrdiff_t buff_size, const Arg& arg) {

    const strf::width_t initial_width = (strf::width_t::max)();
    strf::full_premeasurements pre(initial_width);
    auto printer_input = strf::make_printer_input<char>(&pre, strf::pack(), arg);
    using printer_type = typename decltype(printer_input)::printer_type;
    const printer_type printer{printer_input};
    strf::cstr_destination dst{buff, buff_size};
    printer.print_to(dst);
    auto *end = dst.finish().ptr;

    measure_and_print_result result;
    result.count = static_cast<int>(end - buff);
    result.predicted_size = static_cast<int>(pre.accumulated_ssize());
    result.predicted_width = (initial_width - pre.remaining_width()).round();
    return result;
}

template <std::ptrdiff_t BuffSize, typename Arg>
measure_and_print_result measure_and_print(char(&buff)[BuffSize], const Arg& arg) {
    return measure_and_print(buff, BuffSize, arg);
}

template <typename Arg>
int strf_print_and_check
    ( const char* src_filename
    , int src_line
    , char* buff
    , std::ptrdiff_t buff_size
    , const Arg& arg)
{
    auto r = measure_and_print(buff, buff_size, arg);
    const bool t1 = r.count == r.predicted_size;
    const bool t2 = r.count == r.predicted_size;
    if (!t1 || !t2) {
        ++ test_utils::test_err_count();
        to(test_utils::test_messages_destination())('\n');
        test_utils::print_test_message_header(src_filename, src_line);
        if (!t1) {
            to(test_utils::test_messages_destination())
                ( "\n    Predicted size ", r.predicted_size
                , ", but obtained ", r.count);
        }
        if (!t2) {
            to(test_utils::test_messages_destination())
                ( "\n    Predicted width ", r.predicted_width
                , ", but obtained ", r.count);
        }
    }
    return r.count;
}

template <std::ptrdiff_t BuffSize, typename Arg>
int strf_print_and_check
    ( const char* src_filename
    , int src_line
    , char(&buff)[BuffSize]
    , const Arg& arg)
{
    return strf_print_and_check(src_filename, src_line, buff, BuffSize, arg);
}

#define STRF_PRINT(BUFF, BUFF_SIZE, ARG) \
    strf_print_and_check(__FILE__, __LINE__, BUFF, BUFF_SIZE, ARG)

template <typename StrfArg, typename... SprintfArgs>
void test_vs_sprintf
    ( const char* src_filename
    , int src_line
    , const StrfArg& strf_arg
    , const char* sprintf_fmt
    , SprintfArgs... args)
{
    char strf_buff[500];
    char sprintf_buff[500];
    auto strf_result = measure_and_print(strf_buff, strf_arg);

    const bool t1 = strf_result.count == strf_result.predicted_size;
    const bool t2 = strf_result.count == strf_result.predicted_width;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    int sprintf_len = sprintf(sprintf_buff, sprintf_fmt, args...);
    const bool t3 = strf_result.count == sprintf_len;
    const bool t4 = t3 && strf::detail::str_equal
        (strf_buff, sprintf_buff, strf::detail::safe_cast_size_t(sprintf_len));

    if (!t1 || !t2 || !t3 || !t4) {
        ++ test_utils::test_err_count();
        to(test_utils::test_messages_destination())('\n');
        test_utils::print_test_message_header(src_filename, src_line);
        if (!t1) {
            to(test_utils::test_messages_destination())
                ( "\n    Predicted size ", strf_result.predicted_size
                , ", but obtained ", strf_result.count);
        }
        if (!t2) {
            to(test_utils::test_messages_destination())
                ( "\n    Predicted width ", strf_result.predicted_width
                , ", but obtained ", strf_result.count);
        }
        if (!t4) {
            to(test_utils::test_messages_destination())
                ( "\n     strf printed    : \"", strf_buff
                , "\" (", strf_result.count, " characters)"
                , "\n     sprintf printed : \"", sprintf_buff
                , "\" (", sprintf_len, " characters)" );
        }
    }
    TEST_RECYCLING(sprintf_buff) (strf_arg);
}

#define TEST_VS_SPRINTF(STRF_ARG, SPRINTF_FMT, ...) \
    { auto err_count = test_utils::test_err_count(); \
    ; test_vs_sprintf(__FILE__, __LINE__, STRF_ARG, SPRINTF_FMT, __VA_ARGS__) \
    ; if (err_count != test_utils::test_err_count()) return; }

template <typename Arg>
void test_general_without_precision(const Arg& arg, const char* arg_description)
{
    TEST_SCOPE_DESCRIPTION("\n arg : ", arg_description);

    char buff_sci[100];
    char buff_fixed[500];
    char buff_gen[500];

    auto len_gen   = STRF_PRINT(buff_gen,   sizeof(buff_gen),   strf::gen(clone(arg)) );
    auto len_fixed = STRF_PRINT(buff_fixed, sizeof(buff_fixed), strf::fixed(clone(arg)) );
    auto len_sci   = STRF_PRINT(buff_sci,   sizeof(buff_sci),   strf::sci(clone(arg)) );

    strf::float_notation expected_form = strf::float_notation::fixed;
    std::ptrdiff_t expected_size = sizeof(buff_fixed);
    char* expected = buff_fixed;
    auto expected_len = len_fixed;
    if (len_sci < len_fixed)  {
        expected_form = strf::float_notation::scientific;
        expected_size = sizeof(buff_sci);
        expected = buff_sci;
        expected_len = len_sci;
    }

    TEST_EQ(len_gen, expected_len);
    TEST_CSTR_EQ(buff_gen, expected);

    TEST_RECYCLING(buff_gen)   (strf::gen(clone(arg)));
    TEST_RECYCLING(buff_fixed) (strf::fixed(clone(arg)));
    TEST_RECYCLING(buff_sci)   (strf::sci(clone(arg)));

    {
        len_gen = STRF_PRINT(buff_gen, sizeof(buff_gen), strf::gen(clone(arg)).pad0(50));

        expected_len = STRF_PRINT
            ( expected, expected_size
            , strf::fmt(clone(arg)).set_float_notation(expected_form).pad0(50) );

        TEST_EQ(len_gen, expected_len);
        TEST_CSTR_EQ(buff_gen, expected);
        TEST_RECYCLING(buff_gen) (strf::gen(clone(arg)).pad0(50));
    }
    {
        len_gen   = STRF_PRINT(buff_gen, sizeof(buff_gen), strf::gen(clone(arg)) > 50);

        expected_len = STRF_PRINT
            ( expected, expected_size
            , strf::fmt(clone(arg)).set_float_notation(expected_form) > 50);

        TEST_EQ(len_gen, expected_len);
        TEST_CSTR_EQ(buff_gen, expected);
        TEST_RECYCLING(buff_gen) (strf::gen(clone(arg)) > 50);
    }
}

#define TEST_GENERAL_WITHOUT_PRECISION(ARG) test_general_without_precision(ARG, #ARG);

class float64_tester
{
public:

    float64_tester(std::uint32_t ieee_exponent, std::uint64_t ieee_mantissa)
        : bits_exponent(ieee_exponent)
        , bits_mantissa(ieee_mantissa)
        , value(make_float<double>(ieee_exponent, ieee_mantissa))
    {
        auto dec = strf::detail::decode(value);
        m10 = dec.m10;
        e10 = dec.e10;
        digcount = static_cast<int>(strf::detail::count_digits<10>(dec.m10));

        test_identificator_.description_writer()
            .with(strf::lettercase::mixed)
            ( "\nbits_mantissa = ", *strf::hex(bits_mantissa).pad0(13)
            , " bits_exponent = ", bits_exponent
            , "\n m10 = ", m10
            , " e10 = ", e10
            , " value = ", value );
    }

    void run() const
    {
        auto from_ryu = ::ryu_d2d(bits_mantissa, bits_exponent);
        TEST_EQ(from_ryu.exponent, e10);
        TEST_EQ(from_ryu.mantissa, m10);

        if (from_ryu.exponent != e10 || from_ryu.mantissa != m10) {
            return; // no point to test any further
        }

        test_general_notation();
        test_scientic_notation();
        test_fixed_notation();
    }
    void test_general_notation () const
    {
        TEST_GENERAL_WITHOUT_PRECISION(value);
        TEST_GENERAL_WITHOUT_PRECISION(*+strf::fmt(value));
        TEST_GENERAL_WITHOUT_PRECISION(~!strf::fmt(value));

        for (int p = 0; p <= digcount; ++p) {
            if (p == digcount - 1 && (m10 % 10) == 5) {
                // it is implementation-defined how printf rounds on ties
                break;
            }
            TEST_SCOPE_DESCRIPTION("\n precision = ", p);
            TEST_VS_SPRINTF(strf::gen(value, p), "%.*g", p, value);
            TEST_VS_SPRINTF(strf::gen(value, p) > 40, "%40.*g", p, value);
            TEST_VS_SPRINTF(strf::gen(value, p).pad0(40), "%040.*g", p, value);
        }
    }

    void test_scientic_notation () const
    {
        const int max_precision = digcount - 1;

        TEST_VS_SPRINTF( strf::sci(value), "%.*e", max_precision, value);
        TEST_VS_SPRINTF(~strf::sci(value), "% .*e", max_precision, value);
        TEST_VS_SPRINTF(+strf::sci(value), "%+.*e", max_precision, value);

        TEST_VS_SPRINTF(+strf::sci(value) > 40, "%+40.*e", max_precision, value);
        TEST_VS_SPRINTF(+strf::sci(value) < 40, "%-+40.*e", max_precision, value);
        TEST_VS_SPRINTF(+strf::sci(value).pad0(40), "%+040.*e", max_precision, value);

        for (int p = 0; p <= max_precision; ++p) {
            if (p == max_precision - 1 && (m10 % 10) == 5) {
                // it is implementation-defined how printf rounds on ties
                break;
            }
            TEST_SCOPE_DESCRIPTION("\n precision = ", p);
            TEST_VS_SPRINTF(+strf::sci(value, p), "%+.*e", p, value);
            TEST_VS_SPRINTF(+strf::sci(value, p) > 40, "%+40.*e", p, value);
            TEST_VS_SPRINTF(+strf::sci(value, p) < 40, "%-+40.*e", p, value);
            TEST_VS_SPRINTF(+strf::sci(value, p).pad0(40), "%+040.*e", p, value);
        }
    }

    void test_fixed_notation() const
    {
        if (e10 >= 0) {
            // In many cases Strf's output is different from of sprintf even if it correct
            // They are different, but lead to the same value when parsed

            // TEST_VS_SPRINTF(  strf::fixed(value, 0),   "%.0f", value);
            // TEST_VS_SPRINTF(  strf::fixed(value, 5),   "%.5f", value);
            // TEST_VS_SPRINTF(  strf::fixed(value),   "%.0f", value);
            // TEST_VS_SPRINTF(~*strf::fixed(value), "% #.0f", value);

            // TEST_VS_SPRINTF( +*strf::fixed(value) > digcount + 10
            //                , "%+#*.0f", digcount + 10, value);

            // TEST_VS_SPRINTF( +*strf::fixed(value) < digcount + 10
            //                , "%-+#*.0f", digcount + 10, value);

            // TEST_VS_SPRINTF( +*strf::fixed(value).pad0(digcount + 10)
            //                , "%+#0*.0f", digcount + 10, value);

                
        } else {
            const int min_precision = e10 >= -digcount ? 0 : 1 - e10 - digcount;
            const int max_precision = - e10;
            const int total_digcount = e10 >= -digcount ? digcount : - e10;
            const int width = total_digcount + 5;

            for (int p = min_precision; p <= max_precision; ++p) {
                if (p == max_precision - 1 && (m10 % 10) == 5) {
                    // it is implementation-defined how printf rounds on ties
                    break;
                }
                TEST_SCOPE_DESCRIPTION("\n precision = ", p);

                TEST_VS_SPRINTF(  strf::fixed(value, p),   "%.*f", p, value);
                TEST_VS_SPRINTF(~*strf::fixed(value, p), "% #.*f", p, value);

                TEST_VS_SPRINTF( ~*strf::fixed(value, p) > (short)width
                               , "% #*.*f", width, p, value);
                TEST_VS_SPRINTF( ~*strf::fixed(value, p) < (short)width
                               , "%- #*.*f", width, p, value);
                TEST_VS_SPRINTF( ~*strf::fixed(value, p).pad0((short)width)
                               , "% #0*.*f", width, p, value);
            }
        }
        // todo test with punctuation;


    }

private:

    std::uint32_t bits_exponent;
    std::uint64_t bits_mantissa;
    double value;
    std::uint64_t m10;
    std::int32_t e10;
    int digcount;

    test_utils::test_scope test_identificator_;
};

inline STRF_TEST_FUNC void test_exp_and_mantissa
    ( std::uint32_t ieee_exponent, std::uint64_t ieee_mantissa )
{
#if __cpp_exceptions
    try {
        const float64_tester tester(ieee_exponent, ieee_mantissa);
        tester.run();
    } catch (StrAssertionFailed&) {
    }
#else
    const float64_tester tester(ieee_exponent, ieee_mantissa);
    tester.run();
#endif
}

// inline STRF_TEST_FUNC void test_value(double x) {
//     constexpr int m_size = 52; // bits in matissa
//     std::uint64_t bits = strf::detail::to_bits(x);
//     const std::uint32_t exponent = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
//     const std::uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;

//     test_exp_and_mantissa(exponent, mantissa);
// }

void test_mantissa(std::uint64_t mantissa) {
    strf::to(test_utils::test_messages_destination())
        .with(strf::lettercase::mixed)
        ( "\ntesting mantissa = ", *strf::hex(mantissa).pad0(15));
    test_utils::test_messages_destination() .recycle();
    fflush(stdout);
    
    for (std::uint32_t exponent = 0; exponent < 0x7FF; ++exponent) {
        test_exp_and_mantissa(exponent, mantissa);
    }
}

} // unnamed namespace

void on_strf_assert_failed(const char* file, int line, const char* func, const char* expr)
{
    strf::to(test_utils::test_messages_destination())
        ( file, ':', line, ": ", func, ":\n Assertion `", expr, "` failed!\n");
    test_utils::test_scope::print_stack(test_utils::test_messages_destination());
    test_utils::test_messages_destination().recycle();
    (void) fflush(stdout);
    ++test_utils::test_err_count();
#if __cpp_exceptions
    throw StrAssertionFailed();
#else
    std::abort();
#endif
}

int main() {
    strf::narrow_cfile_writer<char, 1024> test_msg_dst(stdout);
    const test_utils::test_messages_destination_guard g(test_msg_dst);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint64_t> distrib{0, 0xFFFFFFFFFFFFF};

    while (true) {
        test_mantissa(distrib(gen));
        (void) fflush(stdout);
    }
    
    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_msg_dst, "\nAll test passed!\n");
    } else {
        strf::to(test_msg_dst) ('\n', err_count, " tests failed!\n");
    }
    return err_count;
}
