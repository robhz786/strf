//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/test_recycling.hpp"

#define TESTF(X) TEST(X)

namespace {

template <typename FloatT>
struct floating_point_traits;

template <>
struct floating_point_traits<float>
{
    using uint_equiv = unsigned;
    static constexpr unsigned exponent_bits_size = 8;
    static constexpr unsigned mantissa_bits_size = 23;
    static constexpr unsigned max_normal_exp = (1U << exponent_bits_size) - 2;
    static constexpr uint_equiv mantissa_bits_mask = 0x7FFFFF;
};

template <>
struct floating_point_traits<double>
{
    using uint_equiv = std::uint64_t;
    static constexpr unsigned exponent_bits_size = 11;
    static constexpr unsigned mantissa_bits_size = 52;
    static constexpr unsigned max_normal_exp = (1U << exponent_bits_size) - 2;
    static constexpr uint_equiv mantissa_bits_mask = 0xfffffffffffffULL;
};

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
inline STRF_HD FloatT make_float
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

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
STRF_HD FloatT float_max() noexcept
{
    return make_float<FloatT>(helper::max_normal_exp, helper::mantissa_bits_mask);
}

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
STRF_HD FloatT make_negative_nan() noexcept
{
    return make_float<FloatT>(helper::max_normal_exp + 1, 1, true);
}

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
STRF_HD FloatT make_nan() noexcept
{
    return make_float<FloatT>(helper::max_normal_exp + 1, 1);
}

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
STRF_HD FloatT make_infinity() noexcept
{
    return make_float<FloatT>(helper::max_normal_exp + 1, 0);
}

#if ! defined (STRF_FREESTANDING)

float read_floating_point(strf::tag<float>, const char* str, char** str_end)
{
    return std::strtof(str, str_end);
}
double read_floating_point(strf::tag<double>, const char* str, char** str_end)
{
    return std::strtod(str, str_end);
}

#endif

template <typename FloatT>
STRF_TEST_FUNC void test_floating_point(FloatT value)
{
    char buff[200];
    auto res = strf::to(buff) (value);
#if ! defined (STRF_FREESTANDING)
    {
        char* end = nullptr;
        auto parsed = read_floating_point(strf::tag<FloatT>{}, buff, &end);
        TEST_EQ(parsed, value);
        TEST_EQ((void*)end, (void*)res.ptr);
    }
#endif
    {
        strf::premeasurements<strf::precalc_size::yes, strf::precalc_width::yes>
            pre{strf::width_max};
        strf::measure<char>(&pre, strf::pack(), value);
        const std::ptrdiff_t content_size = res.ptr - buff;
        TEST_EQ(pre.accumulated_ssize(), content_size);
        auto width = strf::width_max - pre.remaining_width();
        TEST_TRUE(width == strf::width_t(static_cast<std::uint16_t>(content_size)));
    }
}

#if defined(__GNUC__) && (__GNUC__ == 8)
#  define IGNORING_GCC_WARNING_ARRAY_BOUNDS
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
STRF_TEST_FUNC void test_floating_point
    ( typename helper::uint_equiv ieee_exponent
    , typename helper::uint_equiv ieee_mantissa
    , bool negative = false )
{
    ieee_mantissa = ieee_mantissa & helper::mantissa_bits_mask;
    auto value = make_float<FloatT>(ieee_exponent, ieee_mantissa, negative);
    TEST_SCOPE_DESCRIPTION
        ( "ieee_exp=", ieee_exponent, ' '
        , " ieee_mantissa=", strf::hex(ieee_mantissa).p((helper::mantissa_bits_size + 7) / 8)
        , " value=", value );
    test_floating_point(value); // NOLINT(clang-analyzer-core.StackAddressEscape)
}

#if defined(IGNORING_GCC_WARNING_ARRAY_BOUNDS)
#  undef IGNORING_GCC_WARNING_ARRAY_BOUNDS
#  pragma GCC diagnostic pop
#endif

template <typename FloatT>
STRF_TEST_FUNC void test_several_values()
{
    using helper = floating_point_traits<FloatT>;
    constexpr auto u1 = static_cast<typename helper::uint_equiv>(1);
    for(unsigned e = 0; e <= helper::max_normal_exp; ++e) {
        for(unsigned i = 2; i <= helper::mantissa_bits_size; ++i) {
            const unsigned s = helper::mantissa_bits_size - i;
            test_floating_point<FloatT>(e, helper::mantissa_bits_mask << s);
            test_floating_point<FloatT>(e, u1 << s);
        }
        test_floating_point<FloatT>(e, u1 << (helper::mantissa_bits_size - 1));
        test_floating_point<FloatT>(e, 0);
    }
}

template <typename FloatT, typename FPack>
STRF_TEST_FUNC void test_subnormal_values(const FPack& fp)
{
    constexpr auto j = strf::join_right(20, '_');
    const auto nan = make_nan<FloatT>();
    const auto negative_nan = make_negative_nan<FloatT>();
    const auto infinity = make_infinity<FloatT>();

    TESTF("_________________nan")  (fp, j(nan));
    TESTF("_________________inf")  (fp, j(infinity));
    TESTF("________________-inf")  (fp, j(-infinity));

    TESTF("________________-nan")  (fp, j(strf::fmt(negative_nan)));
    TESTF("________________+nan")  (fp, j(+strf::fmt(nan)));
    TESTF("________________+inf")  (fp, j(+strf::fmt(infinity)));
    TESTF("________________+inf")  (fp, j(+strf::fmt(infinity)));
    TESTF("________________-inf")  (fp, j(strf::fmt(-infinity)));
    TESTF("________________-inf")  (fp, j(+strf::fmt(-infinity)));

    TESTF("________________-nan")  (fp, j(strf::fixed(negative_nan)));
    TESTF("_________________nan")  (fp, j(strf::fixed(nan)));
    TESTF("________________+nan")  (fp, j(+strf::fixed(nan)));
    TESTF("_________________inf")  (fp, j(strf::fixed(infinity)));
    TESTF("________________+inf")  (fp, j(+strf::fixed(infinity)));
    TESTF("________________-inf")  (fp, j(strf::fixed(-infinity)));
    TESTF("________________-inf")  (fp, j(+strf::fixed(-infinity)));

    TESTF("________________-nan")  (fp, j(strf::sci(negative_nan)));
    TESTF("_________________nan")  (fp, j(strf::sci(nan)));
    TESTF("________________+nan")  (fp, j(+strf::sci(nan)));
    TESTF("_________________inf")  (fp, j(strf::sci(infinity)));
    TESTF("________________+inf")  (fp, j(+strf::sci(infinity)));
    TESTF("________________-inf")  (fp, j(strf::sci(-infinity)));
    TESTF("________________-inf")  (fp, j(+strf::sci(-infinity)));

    TESTF("_________________nan")  (fp, j(strf::hex(nan)));
    TESTF("________________-nan")  (fp, j(strf::hex(negative_nan)));
    TESTF("________________+nan")  (fp, j(+strf::hex(nan)));
    TESTF("_________________inf")  (fp, j(strf::hex(infinity)));
    TESTF("________________+inf")  (fp, j(+strf::hex(infinity)));
    TESTF("________________-inf")  (fp, j(strf::hex(-infinity)));
    TESTF("________________-inf")  (fp, j(+strf::hex(-infinity)));

    TESTF("___________~~~~~-nan")  (fp, j(strf::right (negative_nan, 9, '~')));
    TESTF("___________+nan~~~~~")  (fp, j(+strf::left  (nan, 9, '~')));
    TESTF("___________~~+nan~~~")  (fp, j(+strf::center(nan, 9, '~')));
    TESTF("___________~~~~~+nan")  (fp, j(+strf::right (nan, 9, '~')));
    TESTF("___________+nan~~~~~")  (fp, j(+strf::left  (nan, 9, '~')));
    TESTF("___________~~+nan~~~")  (fp, j(+strf::center(nan, 9, '~')));


    TESTF("___________~~~~~+inf")  (fp, j(+strf::right (infinity, 9, '~')));
    TESTF("___________+inf~~~~~")  (fp, j(+strf::left  (infinity, 9, '~')));
    TESTF("___________~~+inf~~~")  (fp, j(+strf::center(infinity, 9, '~')));
    TESTF("___________~~~~~+inf")  (fp, j(+strf::right (infinity, 9, '~').hex()));
    TESTF("___________+inf~~~~~")  (fp, j(+strf::left  (infinity, 9, '~').hex()));
    TESTF("___________~~+inf~~~")  (fp, j(+strf::center(infinity, 9, '~').hex()));

    TESTF("___________     -inf")  (fp, j(strf::pad0(-infinity, 9).hex()));
    TESTF("___________     +inf")  (fp, j(+strf::pad0(infinity, 9).hex()));
    TESTF("___________      inf")  (fp, j(strf::pad0(infinity, 9).hex()));

    TESTF("___________     -inf")  (fp, j(strf::pad0(-infinity, 9)));
    TESTF("___________     +inf")  (fp, j(+strf::pad0(infinity, 9)));
    TESTF("___________      inf")  (fp, j(strf::pad0(infinity, 9)));

    TESTF("___________~~~~~+inf")  (fp, j(+strf::right (infinity, 9, '~').pad0(0).hex()));
    TESTF("___________+inf~~~~~")  (fp, j(+strf::left  (infinity, 9, '~').pad0(8).hex()));
    TESTF("___________~~+inf~~~")  (fp, j(+strf::center(infinity, 9, '~').pad0(8).hex()));
    TESTF("__________~~~~~~+inf")  (fp, j(+strf::right (infinity, 9, '~').pad0(10).hex()));
    TESTF("__________+inf~~~~~~")  (fp, j(+strf::left  (infinity, 9, '~').pad0(10).hex()));
    TESTF("__________~~~+inf~~~")  (fp, j(+strf::center(infinity, 9, '~').pad0(10).hex()));

    TESTF("___________~~~~~+inf")  (fp, j(+strf::right (infinity, 9, '~').pad0(9)));
    TESTF("___________+inf~~~~~")  (fp, j(+strf::left  (infinity, 9, '~').pad0(9)));
    TESTF("___________~~+inf~~~")  (fp, j(+strf::center(infinity, 9, '~').pad0(9)));
    TESTF("__________~~~~~~+inf")  (fp, j(+strf::right (infinity, 9, '~').pad0(10)));
    TESTF("__________+inf~~~~~~")  (fp, j(+strf::left  (infinity, 9, '~').pad0(10)));
    TESTF("__________~~~+inf~~~")  (fp, j(+strf::center(infinity, 9, '~').pad0(10)));

    TESTF("___________~~~~~~inf")  (fp, j(strf::right (infinity, 9, '~').fill_sign()));
    TESTF("___________~~~~~~inf")  (fp, j(strf::right (infinity, 9, '~').fill_sign().pad0(8)));
    TESTF("___________~inf~~~~~")  (fp, j(strf::left  (infinity, 9, '~').fill_sign().pad0(8)));
    TESTF("___________~~~inf~~~")  (fp, j(strf::center(infinity, 9, '~').fill_sign().pad0(8)));
    TESTF("__________~~~~~~~inf")  (fp, j(strf::right (infinity, 9, '~').fill_sign().pad0(10)));
    TESTF("__________~inf~~~~~~")  (fp, j(strf::left  (infinity, 9, '~').fill_sign().pad0(10)));
    TESTF("__________~~~~inf~~~")  (fp, j(strf::center(infinity, 9, '~').fill_sign().pad0(10)));

    TESTF("___________      inf")  (fp, j(strf::pad0(infinity, 9).fill_sign()));
    TESTF("________________ inf")  (fp, j(strf::fmt(infinity).fill_sign()));

    TESTF("___________~~~~~~inf")  (fp, j(strf::right (infinity, 9, '~').hex().fill_sign()));
    TESTF("___________~~~~~~inf")  (fp, j(strf::right (infinity, 9, '~').hex().fill_sign().pad0(8)));
    TESTF("___________~inf~~~~~")  (fp, j(strf::left  (infinity, 9, '~').hex().fill_sign().pad0(8)));
    TESTF("___________~~~inf~~~")  (fp, j(strf::center(infinity, 9, '~').hex().fill_sign().pad0(8)));
    TESTF("__________~~~~~~~inf")  (fp, j(strf::right (infinity, 9, '~').hex().fill_sign().pad0(10)));
    TESTF("__________~inf~~~~~~")  (fp, j(strf::left  (infinity, 9, '~').hex().fill_sign().pad0(10)));
    TESTF("__________~~~~inf~~~")  (fp, j(strf::center(infinity, 9, '~').hex().fill_sign().pad0(10)));

    TESTF("___________      inf")  (fp, j(strf::pad0(infinity, 9).hex().fill_sign()));
    TESTF("________________ inf")  (fp, j(strf::fmt(infinity).hex().fill_sign()));

    TESTF("_________________nan")  (fp, strf::lettercase::lower, j(nan));
    TESTF("_________________inf")  (fp, strf::lettercase::lower, j(infinity));
    TESTF("________________-inf")  (fp, strf::lettercase::lower, j(-infinity));
    TESTF("_________________inf")  (fp, strf::lettercase::lower, j(strf::sci(infinity)));
    TESTF("________________-inf")  (fp, strf::lettercase::lower, j(strf::sci(-infinity)));
    TESTF("_________________inf")  (fp, strf::lettercase::lower, j(strf::hex(infinity)));
    TESTF("________________-inf")  (fp, strf::lettercase::lower, j(strf::hex(-infinity)));
    TESTF("_________________nan")  (fp, strf::lettercase::lower, j(strf::hex(nan)));

    TESTF("_________________NaN")  (fp, strf::lettercase::mixed, j(nan));
    TESTF("________________-NaN")  (fp, strf::lettercase::mixed, j(negative_nan));
    TESTF("_________________Inf")  (fp, strf::lettercase::mixed, j(infinity));
    TESTF("________________-Inf")  (fp, strf::lettercase::mixed, j(-infinity));
    TESTF("_________________Inf")  (fp, strf::lettercase::mixed, j(strf::sci(infinity)));
    TESTF("________________-Inf")  (fp, strf::lettercase::mixed, j(strf::sci(-infinity)));
    TESTF("_________________Inf")  (fp, strf::lettercase::mixed, j(strf::hex(infinity)));
    TESTF("________________-Inf")  (fp, strf::lettercase::mixed, j(strf::hex(-infinity)));
    TESTF("_________________NaN")  (fp, strf::lettercase::mixed, j(strf::hex(nan)));

    TESTF("_________________NAN")  (fp, strf::lettercase::upper, j(nan));
    TESTF("________________-NAN")  (fp, strf::lettercase::upper, j(negative_nan));
    TESTF("_________________INF")  (fp, strf::lettercase::upper, j(infinity));
    TESTF("________________-INF")  (fp, strf::lettercase::upper, j(-infinity));
    TESTF("_________________INF")  (fp, strf::lettercase::upper, j(strf::sci(infinity)));
    TESTF("________________-INF")  (fp, strf::lettercase::upper, j(strf::sci(-infinity)));
    TESTF("_________________INF")  (fp, strf::lettercase::upper, j(strf::hex(infinity)));
    TESTF("________________-INF")  (fp, strf::lettercase::upper, j(strf::hex(-infinity)));
    TESTF("_________________NAN")  (fp, strf::lettercase::upper, j(strf::hex(nan)));
}

STRF_TEST_FUNC void basic_tests()
{
    constexpr auto j = strf::join_right(20, '_');

    TESTF("___________________0")  (j(0.0));
    TESTF("__________________-0")  (j(-0.0));
    TESTF("___________________1")  (j(1.0));
    TESTF("__________________-1")  (j(-1.0));
    TESTF("_________________1.5")  (j(1.5));
    TESTF("_____6.103515625e-05")  (j(6.103515625e-05));
    TESTF("____-6.103515625e-10")  (j(-6.103515625e-10));
    TESTF("____6.103515625e-100")  (j(6.103515625e-100));
    TESTF("_____6.103515625e-10")  (j(strf::sci(6.103515625e-10)));
    TESTF("____6.103515625e-100")  (j(strf::sci(6.103515625e-100)));
    TESTF("____6.103515625e+100")  (j(strf::sci(6.103515625e+100)));
    TESTF("_______0.00048828125")  (j(0.00048828125));
    TESTF("______2048.001953125")  (j(2048.001953125));

    TESTF("___________________0") (j(strf::fmt(0.0)));
    TESTF("__________________-0") (j(strf::fmt(-0.0)));
    TESTF("___________________1") (j(strf::fmt(1.0)));
    TESTF("__________________-1") (j(strf::fmt(-1.0)));
    TESTF("_________________1.5") (j(strf::fmt(1.5)));
    TESTF("_________12345678.25") (j(strf::fmt(12345678.25)));

    TESTF("__________________+1") (j(+strf::fmt(1.0)));
    TESTF("__________________-1") (j(+strf::fmt(-1.0)));
    TESTF("__________________1.") (j(*strf::fmt(1.0)));
    TESTF("_________________-1.") (j(*strf::fmt(-1.0)));
    TESTF("_________________+1.")(j(*+strf::fmt(1.0)));
    TESTF("_________________-1.")(j(+*strf::fmt(-1.0)));

    TESTF("_____________+0.0001") (j(+strf::fixed(0.0001)));
    TESTF("_______+0.0001000000") (j(+strf::fixed(0.0001).p(10)));


    TESTF("_______________1e+50")  (strf::lettercase::lower, j(1e+50));
    TESTF("_______________1e+50")  (strf::lettercase::mixed, j(1e+50));
    TESTF("_______________1E+50")  (strf::lettercase::upper, j(1e+50));

    TESTF("______________ 1e+50")  (strf::lettercase::lower, j(strf::sci(1e+50)>6));
    TESTF("______________ 1e+50")  (strf::lettercase::mixed, j(strf::sci(1e+50)>6));
    TESTF("______________ 1E+50")  (strf::lettercase::upper, j(strf::sci(1e+50)>6));

    //----------------------------------------------------------------
    // when precision is not specified, the general format selects the
    // scientific notation if it is shorter than the fixed notation:
    TESTF("_______________10000")  (j(1e+4));
    TESTF("_______________1e+05")  (j(1e+5));
    TESTF("_____________1200000")  (j(1.2e+6));
    TESTF("_____________1.2e+07")  (j(1.2e+7));
    TESTF("_______________0.001")  (j(1e-03));
    TESTF("_______________1e-04")  (j(1e-04));
    TESTF("_____________0.00012")  (j(1.2e-04));
    TESTF("_____________1.2e-05")  (j(1.2e-05));
    TESTF("____________0.000123")  (j(1.23e-04));
    TESTF("____________1.23e-05")  (j(1.23e-05));
    TESTF("_______________10000")  (j(strf::fmt(1e+4)));
    TESTF("______________10000.") (j(*strf::fmt(1e+4)));
    TESTF("_______________1e+05")  (j(strf::fmt(1e+5)));
    TESTF("______________1.e+05") (j(*strf::fmt(1e+5)));
    TESTF("_____________1200000")  (j(strf::fmt(1.2e+6)));
    TESTF("_____________1.2e+06") (j(*strf::fmt(1.2e+6)));
    TESTF("_______________0.001")   (j(strf::fmt(1e-03)));
    TESTF("_______________1e-04")   (j(strf::fmt(1e-04)));
    TESTF("______________0.0001")  (j(*strf::fmt(1e-04)));
    TESTF("______________1.e-05")  (j(*strf::fmt(1e-05)));
    TESTF("_____________0.00012")   (j(strf::fmt(1.2e-04)));
    TESTF("_____________0.00012") (j(*strf::fmt(1.2e-04)));
    TESTF("_____________1.2e-05")  (j(strf::fmt(1.2e-05)));
    TESTF("_____________1.2e-05") (j(*strf::fmt(1.2e-05)));
    TESTF("____________0.000123")  (j(strf::fmt(1.23e-04)));
    TESTF("____________1.23e-05")  (j(strf::fmt(1.23e-05)));
    TESTF("_____6.103515625e-05")    (j(strf::fmt(6.103515625e-05)));
    TESTF("_______0.00048828125")    (j(strf::fmt(0.00048828125)));

    TESTF("_____________1.e+100") (j(*strf::fmt(1e+100)));
    TESTF("_____________1.e-100") (j(*strf::fmt(1e-100)));

    //----------------------------------------------------------------


    //----------------------------------------------------------------
    // when precision is specified in the general format, do as in printf:
    // - The precision specifies the number of significant digits.
    // - scientific notation is used if the resulting exponent is
    //   less than -4 or greater than or equal to the precision.

    TESTF("_____________0.00012")  (j(strf::fmt(0.000123).p(2)));
    TESTF("_____________1.2e-05")  (j(strf::fmt(0.0000123).p(2)));
    TESTF("_______________1e+01") (j(strf::fmt(12.0).p(1)));
    TESTF("_____________1.2e+02") (j(strf::fmt(123.0).p(2)));
    TESTF("_________________120") (j(strf::fmt(120.0).p(4)));
    TESTF("_________________120") (j(strf::fmt(120.1).p(3)));
    TESTF("_______________1e+04") (j(strf::fmt(10000.0).p(4)));
    TESTF("_______________10000") (j(strf::fmt(10000.0).p(5)));
    TESTF("_______________10000") (j(strf::fmt(10000.0).p(6)));
    TESTF("__________6.1035e-05") (j(strf::fmt(6.103515625e-05).p(5)));

    // and if precision is zero, it treated as 1.
    TESTF("_______________1e+01")  (j(strf::fmt(12.0).p(0)));
    TESTF("_______________2e+01")  (j(strf::fmt(15.125).p(0)));

    // and when removing digits, the last digit is rounded.
    TESTF("_____________1.1e+05") (j(strf::fmt(114999.0).p(2)));
    TESTF("_____________1.2e+05") (j(strf::fmt(115000.0).p(2)));
    TESTF("_____________1.2e+05") (j(strf::fmt(125000.0).p(2)));
    TESTF("_____________1.3e+05") (j(strf::fmt(125001.0).p(2)));

    // and the decimal point appears only if followed by
    // a digit, or if operator*() is used.
    TESTF("_______________1e+04")   (j(strf::fmt(10000.0).p(3)));
    TESTF("______________1.e+04")  (j(*strf::fmt(10000.0).p(1)));
    TESTF("________________123.")  (j(*strf::fmt(123.0).p(3)));

    // and trailing zeros are removed, unless operator*() is used.
    TESTF("_____________1.5e+04")   (j(strf::fmt(15000.0).p(3)));
    TESTF("____________1.50e+04")  (j(*strf::fmt(15000.0).p(3)));
    TESTF("_____________1.5e+04")  (j(strf::fmt(15001.0).p(3)));
    TESTF("_____________1.5e+04")  (j(*strf::fmt(15001.0).p(3)));
    TESTF("_________________123")   (j(strf::fmt(123.0).p(5)));
    TESTF("______________123.00")  (j(*strf::fmt(123.0).p(5)));
    TESTF("______________1000.5") (j(!strf::fmt(1000.5).p(6)));

    // test rounding
    TESTF("_________________2.2")  (j(strf::fmt(2.25).p(2)));
    TESTF("_________________2.3")  (j(strf::fmt(2.25000001).p(2)));
    TESTF("________________2.25")  (j(strf::fmt(2.25000001).p(3)));
    TESTF("_____________2.2e+15")  (j(strf::fmt(2.25e+15).p(2)));
    TESTF("____________2.3e+100")  (j(strf::fmt(2.250001e+100).p(2)));
    TESTF("___________2.25e-100")  (j(strf::fmt(2.250001e-100).p(3)));


    //----------------------------------------------------------------
    // strf::fixed

    TESTF("__________________1.")  (j(*strf::fixed(1.0)));
    TESTF("___________________1")   (j(strf::fixed(1.0)));
    TESTF("__________________+1")  (j(+strf::fixed(1.0)));
    TESTF("__________________-1")   (j(strf::fixed(-1.0)));
    TESTF("__________________-1")  (j(+strf::fixed(-1.0)));

    TESTF("___________________1")  (j(strf::fixed(1.0).p(0)));
    TESTF("__________________1.") (j(*strf::fixed(1.0).p(0)));
    TESTF("_________________1.0")  (j(strf::fixed(1.0).p(1)));
    TESTF("________________1.00")  (j(strf::fixed(1.0).p(2)));
    TESTF("______________1.0000")  (j(strf::fixed(1.0).p(4)));
    TESTF("______________1.2500")  (j(strf::fixed(1.25).p(4)));
    TESTF("____________1.001000")  (j(strf::fixed(1.001).p(6)));
    TESTF("_______________0.000")  (j(strf::fixed(1e-30).p(3)));
    TESTF("_______________0.125")  (j(strf::fixed(0.125)));

    // test rounding
    TESTF("_________________2.2")  (j(strf::fixed(2.25).p(1)));
    TESTF("_________________2.3")  (j(strf::fixed(2.25000001).p(1)));
    TESTF("________________2.25")  (j(strf::fixed(2.25000001).p(2)));
    TESTF("______________0.0001")  (j(strf::fixed(0.0000501).p(4)));
    TESTF("______________0.0000")  (j(strf::fixed(0.00004999).p(4)));
    TESTF("_______________0.000")  (j(strf::fixed(0.0000999).p(3)));

    //----------------------------------------------------------------
    // strf::sci

    TESTF("______________0.e+00")  (j(*strf::sci(0.0)));
    TESTF("______________1.e+04")  (j(*strf::sci(1e+4)));
    TESTF("_____________+1.e+04") (j(*+strf::sci(1e+4)));
    TESTF("_____________-1.e+04") (j(*strf::sci(-1e+4)));

    TESTF("_____________1.0e+04")   (j(strf::sci(1e+4).p(1)));
    TESTF("_____________1.0e+04")   (j(strf::sci(1e+4).p(1)));
    TESTF("___________+1.00e+04")  (j(+strf::sci(1e+4).p(2)));
    TESTF("______________1.e+04")  (j(*strf::sci(1e+4).p(0)));
    TESTF("_____________+1.e+04") (j(*+strf::sci(1e+4).p(0)));
    TESTF("______________-1e+04")  (j(+strf::sci(-1e+4).p(0)));
    TESTF("_____________-1.e+04")  (j(*strf::sci(-1e+4).p(0)));

    TESTF("____________1.25e+02") (j(strf::sci(125.0).p(2)));
    TESTF("_____________1.2e+02") (j(strf::sci(125.0).p(1)));
    TESTF("_____________1.2e+02") (j(strf::sci(115.0).p(1)));
    TESTF("_____________1.3e+06") (j(strf::sci(1250001.0).p(1)));
    TESTF("__________8.1928e+03") (j(strf::sci(8192.75).p(4)));
    TESTF("__________8.1922e+03") (j(strf::sci(8192.25).p(4)));
    TESTF("__________1.0242e+03") (j(strf::sci(1024.25).p(4)));
    TESTF("_____________1.7e+01") (j(strf::sci(16.50006103515625).p(1)));
    TESTF("___________1.250e+02") (j(strf::sci(125.0).p(3)));
    TESTF("_________6.25000e-02") (j(strf::sci(0.0625).p(5)));
    TESTF("________8.192750e+03") (j(strf::sci(8192.75).p(6)));

    TESTF("____________2.2e+100") (j(strf::sci(2.25e+100).p(1)));
    TESTF("____________2.3e-100") (j(strf::sci(2.250001e-100).p(1)));
    TESTF("____________2.25e+15") (j(strf::sci(2.250001e+15).p(2)));



    // ---------------------------------------------------------------
    // alignment
    TESTF("_________******-1.25") (j(strf::right(-1.25, 11, '*')));
    TESTF("_________-1.25******") (j(strf::left(-1.25, 11, '*')));
    TESTF("_________***-1.25***") (j(strf::center(-1.25, 11, '*')));

    TESTF("_________-0000001.25") (j(strf::pad0(-1.25, 11)));
    TESTF("_________+0000001.25") (j(+strf::pad0(1.25, 11)));

    TESTF("_______________-1.25") (j(strf::right(-1.25, 5, '*')));
    TESTF("_______________-1.25") (j(strf::left(-1.25, 5, '*')));
    TESTF("_______________-1.25") (j(strf::center(-1.25, 5, '*')));

    TESTF("_____________\xEF\xBF\xBD\xEF\xBF\xBD-1.25")
        (j(strf::right(-1.25, 7, static_cast<char32_t>(0xFFFFFFF))));
    TESTF("_____________-1.25\xEF\xBF\xBD\xEF\xBF\xBD")
        (j(strf::left(-1.25, 7, static_cast<char32_t>(0xFFFFFFF))));
    TESTF("_____________\xEF\xBF\xBD-1.25\xEF\xBF\xBD")
        (j(strf::center(-1.25, 7, static_cast<char32_t>(0xFFFFFFF))));

    //----------------------------------------------------------------
    // pad0
    TESTF("________000000000001") (j(strf::pad0(1.0, 12)));
    TESTF("________+0000000001.") (j(+*strf::pad0(1.0, 12)));
    TESTF("______  +0000000001.") (j(+*strf::pad0(1.0, 12) > 14));
    TESTF("______~~+0000000001.") (j(+*strf::pad0(1.0, 12).fill('~') > 14));
    TESTF("________ 0000000001.") (j(*strf::pad0(1.0, 12).fill_sign()));
    TESTF("______   0000000001.") (j(*strf::pad0(1.0, 12).fill_sign() > 14));
    TESTF("________+00001234.25") (j(+strf::pad0(1234.25, 12)));
    TESTF("________+001234.2500") (j(+*strf::pad0(1234.25, 12).fixed().p(4)));
    TESTF("________00000001e+20") (j(strf::pad0(1e+20, 12)));
    TESTF("________+000001.e+20") (j(+*strf::pad0(1.e+20, 12)));
    TESTF("________ 000001.e+20") (j(*strf::pad0(1.e+20, 12).fill_sign()));
    TESTF("________00001.25e+20") (j(strf::pad0(1.25e+20, 12)));


    //----------------------------------------------------------------
    // fill_sign
    TESTF("__________000001.125") (j(strf::pad0(1.125, 10)));
    TESTF("__________ 00001.125") (j(strf::pad0(1.125, 10).fill_sign()));
    TESTF("_______~~~~1.125~~~~") (j(strf::center(1.125, 13, '~').fill_sign()));
    TESTF("______~~~~~1.125~~~~") (j(strf::center(1.125, 14, '~').fill_sign()));
    TESTF("______~1.125~~~~~~~~") (j(strf::left(1.125, 14, '~').fill_sign()));
    TESTF("______~~~~~~~~~1.125") (j(strf::right(1.125, 14, '~').fill_sign()));

    TESTF("______~~000001.125~~") (j(strf::center(1.125, 14, '~').pad0(10)));
    TESTF("______~~~00001.125~~") (j(strf::center(1.125, 14, '~').pad0(10).fill_sign()));
    TESTF("______~~~~~00001.125") (j(strf::right(1.125, 14, '~').pad0(10).fill_sign()));
    TESTF("______~00001.125~~~~") (j(strf::left(1.125, 14, '~').pad0(10).fill_sign()));

    TESTF("____________1.125000") (j(strf::fixed(1.125, 6)));
    TESTF("___________ 1.125000") (j(strf::fixed(1.125, 6).fill_sign()));
}

STRF_HD double make_normal_double(std::uint64_t digits, int exp = 0)
{
#if defined(STRF_HAS_COUNTL_ZERO)
    auto left_zeros = strf::detail::countl_zero_ll(digits);
#else
    auto left_zeros = strf::detail::slow_countl_zero_ll(digits);
#endif
    digits ^= (1ULL << (63 - left_zeros)); // clear highest non-zero bit
    const auto bits_exp = 1023 + exp;
    STRF_ASSERT(bits_exp >= 0);
    return make_float<double>(static_cast<unsigned>(bits_exp), digits << (left_zeros - 11));
}


STRF_HD double make_subnormal_double(std::uint64_t digits)
{
#if defined(STRF_HAS_COUNTL_ZERO)
    auto left_zeros = strf::detail::countl_zero_ll(digits);
#else
    auto left_zeros = strf::detail::slow_countl_zero_ll(digits);
#endif
    digits ^= (1ULL << (63-left_zeros));
    return make_float<double>(0, digits << (left_zeros - 11));
}



STRF_TEST_FUNC void test_float32()
{
    constexpr auto j = strf::join_right(25, '_');

    TESTF("_______________________+0") (j(+strf::fixed(0.0F)));
    TESTF("___________+1.1754944e-38") (j(+*strf::gen(+1.1754944e-38)));
    TESTF("____________________+1.25") (j(+strf::fixed(1.25F)));
    TESTF("________________+1.250000") (j(+strf::fixed(1.25F).p(6)));
    TESTF("_____________________+0.1") (j(+strf::fixed(0.1F)));
    TESTF("__________________+1.e+20") (j(+*strf::sci(1e+20F)));
    TESTF("____________+1.000000e+20") (j(+strf::sci(1e+20F).p(6)));
    TESTF("____________________+1.25") (j(+strf::gen(1.25F)));
    TESTF("___________________+1.250") (j(+*strf::gen(1.25F).p(4)));

    TESTF("_______________0x1.ffcp+0") (j(strf::hex(make_normal_double(0x1ffcULL))));
    TESTF("__________0x1.8abcdep+127") (j(strf::hex(make_normal_double(0x18abcdeULL, +127))));
    TESTF("_________-0x1.8abcdep-126") (j(strf::hex(-make_normal_double(0x18abcdeULL, -126))));
    auto denorm_min = strf::detail::bit_cast<float>(static_cast<std::uint32_t>(1));
    TESTF("__________0x0.000002p-126") (j(strf::hex(denorm_min)));
    TESTF("__________0x1.fffffep+127") (j(strf::hex(float_max<float>())));
}

STRF_TEST_FUNC void test_hexadecimal()
{
    constexpr auto j = strf::join_right(25, '_');

    TESTF("0x0p+0") (strf::hex(0.0));
    TESTF("___________________0x0p+0") (j(strf::hex(0.0)));
    TESTF("__________________+0x0p+0") (j(+strf::hex(0.0)));
    TESTF("__________________0x0.p+0") (j(*strf::hex(0.0)));
    TESTF("_________________+0x0.p+0") (j(+*strf::hex(0.0)));
    TESTF("_______________0x0.000p+0") (j(strf::hex(0.0).p(3)));
    TESTF("__________________-0x1p-3") (j(strf::hex(-0.125)));
    TESTF("__________________0x1p+11") (j(strf::hex(2048.0)));
    TESTF("__0x1.fffffffffffffp+1023") (j(strf::hex(float_max<double>())));
    auto denorm_min = strf::detail::bit_cast<double>(static_cast<std::uint64_t>(1));
    TESTF("__0x0.0000000000001p-1022") (j(strf::hex(denorm_min)));
    TESTF("________________0x1p-1022") (j(strf::hex(make_normal_double(0x1, -1022))));
    TESTF("_______________0x1.p-1022") (j(*strf::hex(make_normal_double(0x1, -1022))));
    TESTF("____________0x1.000p-1022") (j(*strf::hex(make_normal_double(0x1, -1022)).p(3)));
    TESTF("___________0x0.0009p-1022") (j(strf::hex(make_subnormal_double(0x10009))));

    TESTF("_________________0x1.8p+0") (j(strf::hex(make_normal_double(0x18ULL))));
    TESTF("_________________0x1.cp+0") (j(strf::hex(make_normal_double(0x1cULL))));
    TESTF("_________________0x1.ep+0") (j(strf::hex(make_normal_double(0x1eULL))));
    TESTF("_________________0x1.fp+0") (j(strf::hex(make_normal_double(0x1fULL))));
    TESTF("________________0x1.f8p+0") (j(strf::hex(make_normal_double(0x1f8ULL))));
    TESTF("________________0x1.fcp+0") (j(strf::hex(make_normal_double(0x1fcULL))));
    TESTF("________________0x1.fep+0") (j(strf::hex(make_normal_double(0x1feULL))));
    TESTF("________________0x1.ffp+0") (j(strf::hex(make_normal_double(0x1ffULL))));
    TESTF("_______________0x1.ff8p+0") (j(strf::hex(make_normal_double(0x1ff8ULL))));
    TESTF("_______________0x1.ffcp+0") (j(strf::hex(make_normal_double(0x1ffcULL))));
    TESTF("_______________0x1.ffep+0") (j(strf::hex(make_normal_double(0x1ffeULL))));
    TESTF("_______________0x1.fffp+0") (j(strf::hex(make_normal_double(0x1fffULL))));
    TESTF("______________0x1.fff8p+0") (j(strf::hex(make_normal_double(0x1fff8ULL))));
    TESTF("______________0x1.fffcp+0") (j(strf::hex(make_normal_double(0x1fffcULL))));
    TESTF("______________0x1.fffep+0") (j(strf::hex(make_normal_double(0x1fffeULL))));
    TESTF("______________0x1.ffffp+0") (j(strf::hex(make_normal_double(0x1ffffULL))));
    TESTF("_____________0x1.ffff8p+0") (j(strf::hex(make_normal_double(0x1ffff8ULL))));
    TESTF("_____________0x1.ffffcp+0") (j(strf::hex(make_normal_double(0x1ffffcULL))));
    TESTF("_____________0x1.ffffep+0") (j(strf::hex(make_normal_double(0x1ffffeULL))));
    TESTF("_____________0x1.fffffp+0") (j(strf::hex(make_normal_double(0x1fffffULL))));
    TESTF("____________0x1.fffff8p+0") (j(strf::hex(make_normal_double(0x1fffff8ULL))));
    TESTF("____________0x1.fffffcp+0") (j(strf::hex(make_normal_double(0x1fffffcULL))));
    TESTF("____________0x1.fffffep+0") (j(strf::hex(make_normal_double(0x1fffffeULL))));
    TESTF("____________0x1.ffffffp+0") (j(strf::hex(make_normal_double(0x1ffffffULL))));
    TESTF("___________0x1.ffffff8p+0") (j(strf::hex(make_normal_double(0x1ffffff8ULL))));
    TESTF("___________0x1.ffffffcp+0") (j(strf::hex(make_normal_double(0x1ffffffcULL))));
    TESTF("___________0x1.ffffffep+0") (j(strf::hex(make_normal_double(0x1ffffffeULL))));
    TESTF("___________0x1.fffffffp+0") (j(strf::hex(make_normal_double(0x1fffffffULL))));
    TESTF("__________0x1.fffffff8p+0") (j(strf::hex(make_normal_double(0x1fffffff8ULL))));
    TESTF("__________0x1.fffffffcp+0") (j(strf::hex(make_normal_double(0x1fffffffcULL))));
    TESTF("__________0x1.fffffffep+0") (j(strf::hex(make_normal_double(0x1fffffffeULL))));
    TESTF("__________0x1.ffffffffp+0") (j(strf::hex(make_normal_double(0x1ffffffffULL))));
    TESTF("_________0x1.ffffffff8p+0") (j(strf::hex(make_normal_double(0x1ffffffff8ULL))));
    TESTF("_________0x1.ffffffffcp+0") (j(strf::hex(make_normal_double(0x1ffffffffcULL))));
    TESTF("_________0x1.ffffffffep+0") (j(strf::hex(make_normal_double(0x1ffffffffeULL))));
    TESTF("_________0x1.fffffffffp+0") (j(strf::hex(make_normal_double(0x1fffffffffULL))));
    TESTF("________0x1.fffffffff8p+0") (j(strf::hex(make_normal_double(0x1fffffffff8ULL))));
    TESTF("________0x1.fffffffffcp+0") (j(strf::hex(make_normal_double(0x1fffffffffcULL))));
    TESTF("________0x1.fffffffffep+0") (j(strf::hex(make_normal_double(0x1fffffffffeULL))));
    TESTF("________0x1.ffffffffffp+0") (j(strf::hex(make_normal_double(0x1ffffffffffULL))));
    TESTF("_______0x1.ffffffffff8p+0") (j(strf::hex(make_normal_double(0x1ffffffffff8ULL))));
    TESTF("_______0x1.ffffffffffcp+0") (j(strf::hex(make_normal_double(0x1ffffffffffcULL))));
    TESTF("_______0x1.ffffffffffep+0") (j(strf::hex(make_normal_double(0x1ffffffffffeULL))));
    TESTF("_______0x1.fffffffffffp+0") (j(strf::hex(make_normal_double(0x1fffffffffffULL))));
    TESTF("______0x1.fffffffffff8p+0") (j(strf::hex(make_normal_double(0x1fffffffffff8ULL))));
    TESTF("______0x1.fffffffffffcp+0") (j(strf::hex(make_normal_double(0x1fffffffffffcULL))));
    TESTF("______0x1.fffffffffffep+0") (j(strf::hex(make_normal_double(0x1fffffffffffeULL))));
    TESTF("______0x1.ffffffffffffp+0") (j(strf::hex(make_normal_double(0x1ffffffffffffULL))));
    TESTF("_____0x1.ffffffffffff8p+0") (j(strf::hex(make_normal_double(0x1ffffffffffff8ULL))));
    TESTF("_____0x1.ffffffffffffcp+0") (j(strf::hex(make_normal_double(0x1ffffffffffffcULL))));
    TESTF("_____0x1.ffffffffffffep+0") (j(strf::hex(make_normal_double(0x1ffffffffffffeULL))));
    TESTF("_____0x1.fffffffffffffp+0") (j(strf::hex(make_normal_double(0x1fffffffffffffULL))));
    TESTF("_____0x1.0ffffffffffffp+0") (j(strf::hex(make_normal_double(0x10ffffffffffffULL))));

    TESTF("___________________0x1p+0") (j(strf::hex(make_normal_double(0x112345ULL)).p(0)));
    TESTF("_____________0x1.12345p+0") (j(strf::hex(make_normal_double(0x112345ULL)).p(5)));
    TESTF("____________0x1.123450p+0") (j(strf::hex(make_normal_double(0x112345ULL)).p(6)));
    TESTF("__________________0x1.p+0") (j(*strf::hex(make_normal_double(0x112345ULL)).p(0)));
    TESTF("_________________+0x1.p+0") (j(+*strf::hex(make_normal_double(0x112345ULL)).p(0)));
    TESTF("____________0x0.000p-1022") (j(strf::hex(make_subnormal_double(0x10008ULL)).p(3)));
    TESTF("____________0x0.002p-1022") (j(strf::hex(make_subnormal_double(0x10018ULL)).p(3)));
    TESTF("____________0x0.001p-1022") (j(strf::hex(make_subnormal_double(0x10008000000001ULL)).p(3)));

    TESTF("___________________0X0P+0")  (strf::lettercase::upper, j(strf::hex(0.0)));
    TESTF("_________0X0.ABCDEFP-1022")  (strf::lettercase::upper, j(strf::hex(make_subnormal_double(0x1abcdefULL))));
    TESTF("___________0X1.ABCDEFP+10")  (strf::lettercase::upper, j(strf::hex(make_normal_double(0x1abcdefULL, +10))));

    TESTF("___________________0x0p+0")  (strf::lettercase::mixed, j(strf::hex(0.0)));
    TESTF("_________0x0.ABCDEFp-1022")  (strf::lettercase::mixed, j(strf::hex(make_subnormal_double(0x1abcdefULL))));
    TESTF("___________0x1.ABCDEFp+10")  (strf::lettercase::mixed, j(strf::hex(make_normal_double(0x1abcdefULL, +10))));

    TESTF("___________________0x0p+0")  (strf::lettercase::lower, j(strf::hex(0.0)));
    TESTF("_________0x0.abcdefp-1022")  (strf::lettercase::lower, j(strf::hex(make_subnormal_double(0x1abcdefULL))));
    TESTF("___________0x1.abcdefp+10")  (strf::lettercase::lower, j(strf::hex(make_normal_double(0x1abcdefULL, +10))));

    TESTF("______________-0x1p+0****") (j(strf::left(-1.0, 11, '*').hex()));
    TESTF("______________****-0x1p+0") (j(strf::right(-1.0, 11, '*').hex()));
    TESTF("______________**-0x1p+0**") (j(strf::center(-1.0, 11, '*').hex()));
    TESTF("__________________-0x1p+0") (j(strf::center(-1.0, 7, '*').hex()));

    // pad0
    TESTF("______________-0x00001p+0") (j(strf::pad0(-1.0, 11).hex()));
    TESTF("______________+0x0001.p+0") (j(+*strf::pad0(1.0, 11).hex()));
    TESTF("__________0x001.123450p+0") (j(strf::hex(make_normal_double(0x112345ULL)).p(6).pad0(15)));
    TESTF("______**0x001.123450p+0**") (j(strf::hex(make_normal_double(0x112345ULL)).p(6).pad0(15).fill('*') ^ 19));
    TESTF("__________0x001.123450p+0") (j(strf::hex(make_normal_double(0x112345ULL)).p(6).pad0(15).fill('*') ^ 15));

    // fill_sign
    TESTF("_____________0x001.125p+1") (j(strf::hex(make_normal_double(0x1125ULL, +0x1)).pad0(12)));
    TESTF("______________ 0x1.125p+1") (j(strf::hex(make_normal_double(0x1125ULL, +0x1)).fill_sign()));

    TESTF("_____________ 0x01.125p+1") (j(strf::hex(make_normal_double(0x1125ULL, +0x1)).pad0(12).fill_sign()));
    TESTF("_____________*0x01.125p+1") (j(strf::hex(make_normal_double(0x1125ULL, +0x1)).pad0(12).fill_sign().fill('*')));
    TESTF("_________*****0x01.125p+1") (j(strf::right(make_normal_double(0x1125ULL, +0x1), 16, '*').hex().pad0(12).fill_sign()));
    TESTF("_________***0x01.125p+1**") (j(strf::center(make_normal_double(0x1125ULL, +0x1), 16, '*').hex().pad0(12).fill_sign()));
    TESTF("_________*0x01.125p+1****") (j(strf::left(make_normal_double(0x1125ULL, +0x1), 16, '*').hex().pad0(12).fill_sign()));

    TESTF("_________******0x1.125p+1") (j(strf::right(make_normal_double(0x1125ULL, +0x1), 16, '*').hex().fill_sign()));
    TESTF("_________***0x1.125p+1***") (j(strf::center(make_normal_double(0x1125ULL, +0x1), 16, '*').hex().fill_sign()));
    TESTF("_________*0x1.125p+1*****") (j(strf::left(make_normal_double(0x1125ULL, +0x1), 16, '*').hex().fill_sign()));

}

template <int Base>
struct numpunct_maker {

    STRF_HD explicit numpunct_maker(char32_t sep)
        : separator(sep)
    {
    }

    template <typename... G>
    STRF_HD strf::numpunct<Base> operator() (G... grp) const
    {
        return strf::numpunct<Base>(grp...) .thousands_sep(separator);
    }
    char32_t separator;
};

STRF_TEST_FUNC void test_punctuation()
{
    constexpr auto j = strf::join_right(20, '_');

    {   //alternative punct characters
        auto p = strf::numpunct<10>{3}.thousands_sep(':').decimal_point(',');
        TESTF("_____________1:000,5")  (p, j(strf::punct(1000.5)));
        TESTF("_____________1,5e+50")  (p, j(strf::punct(1.5e+50)));
        TESTF("____________1:000,50")  (p, j(!strf::fixed(1000.5).p(2)));
        TESTF("____________1:000,50")  (p, j(*!strf::fmt(1000.5).p(6)));
        TESTF("_____________1:000,5")  (p, j(!strf::fmt(1000.5).p(6)));
        TESTF("____________1,50e+50")  (p, j(!strf::sci(1.5e+50).p(2)));
        TESTF("__________________1,")  (p, j(*!strf::fmt(1.0)));

        TESTF("_________+1:000:000,")  (p, j(*+!strf::fixed(1000000.0)));
        TESTF("_____+1:000:000,0000")  (p, j(*+!strf::fixed(1000000.0).p(4)));
        TESTF("__+00000001:000:000,")  (p, j(*+!strf::fixed(1000000.0).pad0(18)));

        auto px = strf::numpunct<16>{3}.decimal_point(',');
        TESTF("________0x1,12345p+0")  (px, j(!strf::hex(make_normal_double(0x112345ULL))));
    }
    {   //encoding big punct characters
        auto p = strf::numpunct<10>{3}.thousands_sep(0x10AAAA).decimal_point(0x10FFFF);

        TESTF(u8"_____________1\U0010AAAA" u8"000\U0010FFFF" u8"5") (p, j(strf::punct(1000.5)));
        TESTF(u8"_____________1\U0010FFFF" u8"5e+50")               (p, j(strf::punct(1.5e+50)));
        TESTF(u8"____________1\U0010AAAA" u8"000\U0010FFFF" u8"50") (p, j(!strf::fixed(1000.5).p(2)));
        TESTF(u8"______________1\U0010FFFF" u8"e+50")   (p, j(*!strf::sci(1e+50)));
        TESTF(u8"____________1\U0010FFFF" u8"50e+50")   (p, j(!strf::sci(1.5e+50).p(2)));
        TESTF(u8"__________________1\U0010FFFF")        (p, j(*!strf::fmt(1.0)));
        TESTF(u8"_________________0\U0010FFFF" u8"1")   (p, j(!strf::fixed(0.1)));
        TESTF(u8"_________________0\U0010FFFF" u8"1")   (p, j(strf::punct(0.1)));

        //in hexadecimal
        auto px = strf::numpunct<16>{3}.decimal_point(0x10FFFF);
        TESTF(u8"________0x1\U0010FFFF" u8"12345p+0")
            (px, j(!strf::hex(make_normal_double(0x112345ULL))));
    }
    {   // encoding punct chars in a single-byte encoding
        auto fp = strf::pack
            ( strf::iso_8859_3_t<char>{}
            , strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130) );

        TESTF("_____________1\xFF""000\xA9""5")  (fp, j(strf::punct(1000.5)));
        TESTF("_____________1\xA9""5e+50")       (fp, j(strf::punct(1.5e+50)));
        TESTF("____________1\xFF""000\xA9""50")  (fp, j(!strf::fixed(1000.5).p(2)));
        TESTF("____________1\xA9""50e+50")       (fp, j(!strf::sci(1.5e+50).p(2)));
        TESTF("__________________1\xA9")         (fp, j(*!strf::fmt(1.0)));
        TESTF("_________________0\xA9" "1")      (fp, j(!strf::fixed(0.1)));
        TESTF("_________________0\xA9" "1")      (fp, j(strf::punct(0.1)));

        auto fpx = strf::pack
            ( strf::iso_8859_3_t<char>{}, strf::numpunct<16>(4).decimal_point(0x130) );

        TESTF("________0x1\xA9""12345p+0")  (fpx, j(!strf::hex(make_normal_double(0x112345ULL))));
    }
    {   // invalid punct characters
        // ( thousand separators are omitted  )

        auto p = strf::numpunct<10>{3}.thousands_sep(0xFFFFFF).decimal_point(0xEEEEEE);
        TESTF(u8"______________1000\uFFFD5")         (p, j(strf::punct(1000.5)));
        TESTF(u8"_____________1\uFFFD" u8"5e+50")    (p, j(strf::punct(1.5e+50)));
        TESTF(u8"_____________1000\uFFFD" u8"50")    (p, j(!strf::fixed(1000.5).p(2)));
        TESTF(u8"____________1\uFFFD" u8"50e+50")    (p, j(!strf::sci(1.5e+50).p(2)));
        TESTF(u8"__________________1\uFFFD")         (p, j(*!strf::fmt(1.0)));
        TESTF(u8"_________________0\uFFFD" u8"1")    (p, j(!strf::fixed(0.1)));
        TESTF(u8"_________________0\uFFFD" u8"1")    (p, j(strf::punct(0.1)));

        auto px = strf::numpunct<16>{3}.decimal_point(0xEEEEEE);
        TESTF(u8"________0x1\uFFFD" u8"12345p+0")  (px, j(!strf::hex(make_normal_double(0x112345ULL))));
    }
    {   // invalid punct characters  in a single-byte encoding
        // ( thousand separators are omitted  )
        auto fp = strf::pack
            ( strf::iso_8859_3_t<char>{}
            , strf::numpunct<10>(3).thousands_sep(0xFFF).decimal_point(0xEEE) );

        TESTF("______________1000?5")  (fp, j(strf::punct(1000.5)));
        TESTF("_____________1?5e+50")  (fp, j(strf::punct(1.5e+50)));
        TESTF("_____________1000?50")  (fp, j(!strf::fixed(1000.5).p(2)));
        TESTF("____________1?50e+50")  (fp, j(!strf::sci(1.5e+50).p(2)));
        TESTF("__________________1?")  (fp, j(*!strf::fmt(1.0)));
        TESTF("_________________0?1")  (fp, j(!strf::fixed(0.1)));
        TESTF("_________________0?1")  (fp, j(strf::punct(0.1)));

        auto fpx = strf::pack
            ( strf::iso_8859_3_t<char>{}, strf::numpunct<16>(4).decimal_point(0xEEE) );

        TESTF("________0x1?12345p+0")  (fpx, j(!strf::hex(make_normal_double(0x112345ULL))));
    }

    {   // When the integral part does not have trailing zeros

        TESTF("1:048:576")
            ( strf::numpunct<10>(3).thousands_sep(':')
            , strf::punct(1048576.0));

        TESTF("1:048:576")
            ( strf::numpunct<10>(3).thousands_sep(':')
            , !strf::fixed(1048576.0) );

        TESTF(u8"1\u0ABC" u8"048\u0ABC" u8"576")
            ( strf::numpunct<10>(3).thousands_sep(0xABC)
            , strf::punct(1048576.0));

        TESTF(u8"1\u0ABC" u8"048\u0ABC" u8"576")
            ( strf::numpunct<10>(3).thousands_sep(0xABC)
            , !strf::fixed(1048576.0) );
    }
    {   // variable groups

        using grp = strf::numpunct<10>;

        TESTF("1,00,00,0,00,0")    (grp(1,2,1,2), !strf::fixed(100000000.0));
        TESTF("10,00,00,0,00,0")   (grp(1,2,1,2), !strf::fixed(1000000000.0));

        TESTF("32,10,00,0,00,0")   (grp(1,2,1,2), !strf::fixed(3210000000.0));
        TESTF("43,21,00,0,00,0")   (grp(1,2,1,2), !strf::fixed(4321000000.0));
        TESTF("54,32,10,0,00,0")   (grp(1,2,1,2), !strf::fixed(5432100000.0));
        TESTF("7,65,432,10,0")     (grp(1,2,3,2), !strf::fixed(765432100.0));
        TESTF("7,6,543,21,0,0")    (grp(1,1,2,3,1), !strf::fixed(765432100.0));
        TESTF("7,654,321,00,0")    (grp(1,2,3), !strf::fixed(7654321000.0));

        TESTF("1000,00,0")   (grp(1,2,-1), !strf::fixed(1000000.0));
        TESTF("1234,00,0")   (grp(1,2,-1), !strf::fixed(1234000.0));
        TESTF("1234,50,0")   (grp(1,2,-1), !strf::fixed(1234500.0));
    }
    {   // variable groups and big separator
        const numpunct_maker<10> grp{0xABCD};

        TESTF(u8"1\uABCD" u8"00\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,1,2), !strf::fixed(100000000.0));
        TESTF(u8"10\uABCD" u8"00\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,1,2), !strf::fixed(1000000000.0));

        TESTF(u8"32\uABCD" u8"10\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,1,2), !strf::fixed(3210000000.0));
        TESTF(u8"43\uABCD" u8"21\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,1,2), !strf::fixed(4321000000.0));
        TESTF(u8"54\uABCD" u8"32\uABCD" u8"10\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,1,2), !strf::fixed(5432100000.0));
        TESTF(u8"7\uABCD" u8"65\uABCD" u8"432\uABCD" u8"10\uABCD" u8"0")
              (grp(1,2,3,2), !strf::fixed(765432100.0));
        TESTF(u8"7\uABCD" u8"6\uABCD" u8"543\uABCD" u8"21\uABCD" u8"0\uABCD" u8"0")
              (grp(1,1,2,3,1), !strf::fixed(765432100.0));
        TESTF(u8"7\uABCD" u8"654\uABCD" u8"321\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,3), !strf::fixed(7654321000.0));

        TESTF(u8"1000\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,-1), !strf::fixed(1000000.0));
        TESTF(u8"1234\uABCD" u8"00\uABCD" u8"0")
              (grp(1,2,-1), !strf::fixed(1234000.0));
        TESTF(u8"1234\uABCD" u8"50\uABCD" u8"0")
              (grp(1,2,-1), !strf::fixed(1234500.0));
    }
    {   // when precision is not specified, the general format selects the
        // scientific notation if it is shorter than the fixed notation:
        auto p1 = strf::numpunct<10>{1};

        TESTF("_______________1,0,0")  (p1, j(strf::punct(1e+2)));
        TESTF("_______________1e+03")  (p1, j(strf::punct(1e+3)));
        TESTF("_____________1,2,0,0")  (p1, j(strf::punct(1.2e+3)));
        TESTF("_____________1.2e+04")  (p1, j(strf::punct(1.2e+4)));
        TESTF("______________+1,0,0")  (p1, j(+strf::punct(1e+2)));
        TESTF("______________1,0,0.")  (p1, j(*strf::punct(1e+2)));
        TESTF("_______________1e+03")  (p1, j(strf::punct(1e+3)));
        TESTF("______________1.e+03")  (p1, j(*strf::punct(1e+3)));
        TESTF("_____________1,2,0,0")  (p1, j(strf::punct(1.2e+3)));
        TESTF("_____________1.2e+03")  (p1, j(*strf::punct(1.2e+3)));
        TESTF("_______________1e-04")  (p1, j(strf::punct(1e-4)));
        TESTF("_______________0.001")  (p1, j(strf::punct(1e-3)));
        TESTF("_______________1e-04")  (p1, j(strf::punct(1e-4)));
        TESTF("_______________0.001")  (p1, j(strf::punct(1e-3)));
        TESTF("_______________1e+05")  (strf::numpunct<10>(8), j(strf::punct(1e+5)));
        TESTF("________1.000005e+05")  (p1, j(strf::punct(100000.5)));
        TESTF("_________1,0,0,0,0.5")  (p1, j(strf::punct(10000.5)));
    }
}

STRF_TEST_FUNC void round_up_999()
{
    // When 999... is rounded up becaming 1000...

    // strf::sci
    TESTF("1e-04")      (strf::sci(9.6e-5).p(0));
    TESTF("1.000e-04")  (strf::sci(9.9996e-5).p(3));
    TESTF("1.000e+06")  (strf::sci(9.9996e+5).p(3));
    TESTF("1.000e+100") (strf::sci(9.9996e+99, 3));
    TESTF("1.000e-99")  (strf::sci(9.9996e-100, 3));
    TESTF("1e+100")     (strf::sci(9.5e+99, 0));
    TESTF("1e-99")      (strf::sci(9.5e-100, 0));
    TESTF("+1.e+100")   (+*strf::sci(9.5e+99, 0));
    TESTF("+1.e-99")    (+*strf::sci(9.5e-100, 0));
    TESTF("-1.e+100")   (*strf::sci(-9.5e+99, 0));
    TESTF("-1.e-99")    (*strf::sci(-9.5e-100, 0));

    // strf::fixed
    TESTF("9.9996")    ( strf::fixed(9.9996).p(4));
    TESTF("10.000")    ( strf::fixed(9.9996).p(3));
    TESTF("10")        ( strf::fixed(9.9996).p(0));
    TESTF("10.")       (*strf::fixed(9.9996).p(0));

    TESTF("0.009995")  ( strf::fixed(9.995e-03).p(6));
    TESTF("0.01000")   ( strf::fixed(9.995e-03).p(5));

    TESTF("0.9995")  ( strf::fixed(9.995e-01).p(4));
    TESTF("1.000")   ( strf::fixed(9.995e-01).p(3));

    // strf::gen
    TESTF("0.001")      ( strf::gen(9.9996e-4).p(2));
    TESTF("0.0010")     (*strf::gen(9.9996e-4).p(2));
    TESTF("0.0001000")  (*strf::gen(9.9996e-5).p(4));

    TESTF("1e-05")      ( strf::gen(9.6e-6).p(0));
    TESTF("0.0001")     ( strf::gen(9.6e-5).p(1));
    TESTF("1e-05")      ( strf::gen(9.9996e-6).p(4));
    TESTF("1.000e-05")  (*strf::gen(9.9996e-6).p(4));
    TESTF("1.000e-99")  (*strf::gen(9.9996e-100).p(4));
    TESTF("1.000e-100") (*strf::gen(9.9996e-101).p(4));
    TESTF("1.000e+100") (*strf::gen(9.9996e+99).p(4));
    TESTF("1.000e+101") (*strf::gen(9.9996e+100).p(4));

    TESTF("1.000e+05") (*strf::gen(9.9996e+4).p(4));
    TESTF("1.000e+04") (*strf::gen(9.9996e+3).p(4));
    TESTF("1000.")     (*strf::gen(9.9996e+2).p(4));

    TESTF("9.9996e+05") ( strf::gen(9.9996e+5).p(5));
    TESTF("1e+05")      ( strf::gen(9.9996e+4).p(4));
    TESTF("1e+04")      ( strf::gen(9.9996e+3).p(4));
    TESTF("1000")       ( strf::gen(9.9996e+2).p(4));
    TESTF("9.9996e+05") (*strf::gen(9.9996e+5).p(5));
    TESTF("1.000e+05")  (*strf::gen(9.9996e+4).p(4));
    TESTF("1.000e+04")  (*strf::gen(9.9996e+3).p(4));
    TESTF("1000.")      (*strf::gen(9.9996e+2).p(4));

    TESTF("99996")     ( strf::gen(99996.0).p(5));
    TESTF("1e+05")     ( strf::gen(99996.0).p(4));

    TESTF("10")        ( strf::gen(9.996).p(3));
    TESTF("10.")       (*strf::gen(9.9996).p(2));
    TESTF("10.0")      (*strf::gen(9.996).p(3));
    TESTF("10")        ( strf::gen(9.9996).p(2));
    TESTF("1e+01")     ( strf::gen(9.6).p(1));
    TESTF("1.e+01")    (*strf::gen(9.6).p(1));
}

STRF_TEST_FUNC void other_tests()
{
    TEST_RECYCLING("0.13326196335449228") (0.13326196335449228);

    {
        auto px = strf::numpunct<16>{3}.decimal_point(0x10FFFF);        
        constexpr auto j = strf::join_right(20, '_');

        TEST_RECYCLING(u8"________0x1\U0010FFFF" u8"12345p+0")
            (px, j(!strf::hex(make_normal_double(0x112345ULL))));
    }
}

STRF_TEST_FUNC void test_input_float()
{
#if defined(__GNUC__) && (__GNUC__ == 8)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

    {
        TEST_SCOPE_DESCRIPTION("subnormal float 32");
        {
            TEST_SCOPE_DESCRIPTION("without punctuation");
            test_subnormal_values<float>(strf::pack());
        }
        {
            TEST_SCOPE_DESCRIPTION("with punctuation");
            test_subnormal_values<float>(strf::no_grouping<10>{});
        }
    }
    {
        TEST_SCOPE_DESCRIPTION("subnormal double");
        {
            TEST_SCOPE_DESCRIPTION("without punctuation");
            test_subnormal_values<double>(strf::pack());
        }
        {
            TEST_SCOPE_DESCRIPTION("with punctuation");
            test_subnormal_values<double>(strf::no_grouping<10>{});
        }
    }

    basic_tests();
    test_float32();
    round_up_999();

#if defined(__GNUC__) && (__GNUC__ == 8)
#  pragma GCC diagnostic pop
#endif

    test_hexadecimal();
    test_several_values<float>();
    test_several_values<double>();
    test_punctuation();

    other_tests();
}

} // unnamed namespace

REGISTER_STRF_TEST(test_input_float)

