//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace {

template <typename FloatT>
struct floating_point_traits;

template <>
struct floating_point_traits<float>
{
    using uint_equiv = unsigned;
    static constexpr int exponent_bits_size = 8;
    static constexpr int mantissa_bits_size = 23;
    static constexpr unsigned max_normal_exp = (1 << exponent_bits_size) - 2;
    static constexpr uint_equiv mantissa_bits_mask = 0x7FFFFF;
};

template <>
struct floating_point_traits<double>
{
    using uint_equiv = std::uint64_t;
    static constexpr int exponent_bits_size = 11;
    static constexpr int mantissa_bits_size = 52;
    static constexpr unsigned max_normal_exp = (1 << exponent_bits_size) - 2;
    static constexpr uint_equiv mantissa_bits_mask = 0xFFFFFFFFFFFFFULL;
};

template <typename FloatT, typename helper = floating_point_traits<FloatT>>
inline STRF_HD FloatT make_float
    ( typename helper::uint_equiv ieee_exponent
    , typename helper::uint_equiv ieee_mantissa
    , bool negative = false )
{
    typename helper::uint_equiv sign = negative;
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
        strf::preprinting<strf::precalc_size::yes, strf::precalc_width::yes> p{strf::width_max};
        strf::precalculate<char>(p, strf::pack(), value);
        std::size_t content_size = res.ptr - buff;
        TEST_EQ(p.accumulated_size(), content_size);
        auto width = strf::width_max - p.remaining_width();
        TEST_TRUE(width == strf::width_t(static_cast<std::uint16_t>(content_size)));
    }
}

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
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
    FloatT value = make_float<FloatT>(ieee_exponent, ieee_mantissa, negative);
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
            unsigned s = helper::mantissa_bits_size - i;
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

    TEST("_________________nan").with(fp)  (j(nan));
    TEST("_________________inf").with(fp)  (j(infinity));
    TEST("________________-inf").with(fp)  (j(-infinity));

    TEST("________________-nan").with(fp)  (j(strf::fmt(negative_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fmt(nan)));
    TEST("________________+inf").with(fp)  (j(+strf::fmt(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::fmt(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::fmt(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::fmt(-infinity)));

    TEST("________________-nan").with(fp)  (j(strf::fixed(negative_nan)));
    TEST("_________________nan").with(fp)  (j(strf::fixed(nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fixed(nan)));
    TEST("_________________inf").with(fp)  (j(strf::fixed(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::fixed(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::fixed(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::fixed(-infinity)));

    TEST("________________-nan").with(fp)  (j(strf::sci(negative_nan)));
    TEST("_________________nan").with(fp)  (j(strf::sci(nan)));
    TEST("________________+nan").with(fp)  (j(+strf::sci(nan)));
    TEST("_________________inf").with(fp)  (j(strf::sci(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::sci(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::sci(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::sci(-infinity)));

    TEST("_________________nan").with(fp)  (j(strf::hex(nan)));
    TEST("________________-nan").with(fp)  (j(strf::hex(negative_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::hex(nan)));
    TEST("_________________inf").with(fp)  (j(strf::hex(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::hex(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::hex(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::hex(-infinity)));

    TEST("___________~~~~~-nan").with(fp) (j(strf::right (negative_nan, 9, '~')));
    TEST("___________+nan~~~~~").with(fp) (j(+strf::left  (nan, 9, '~')));
    TEST("___________~~+nan~~~").with(fp) (j(+strf::center(nan, 9, '~')));
    TEST("___________~~~~~+nan").with(fp) (j(+strf::right (nan, 9, '~')));
    TEST("___________+nan~~~~~").with(fp) (j(+strf::left  (nan, 9, '~')));
    TEST("___________~~+nan~~~").with(fp) (j(+strf::center(nan, 9, '~')));


    TEST("___________~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~')));
    TEST("___________+inf~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~')));
    TEST("___________~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~')));
    TEST("___________~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~').hex()));
    TEST("___________+inf~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~').hex()));
    TEST("___________~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~').hex()));

    TEST("___________     -inf").with(fp) (j(strf::pad0(-infinity, 9).hex()));
    TEST("___________     +inf").with(fp) (j(+strf::pad0(infinity, 9).hex()));
    TEST("___________      inf").with(fp) (j(strf::pad0(infinity, 9).hex()));

    TEST("___________     -inf").with(fp) (j(strf::pad0(-infinity, 9)));
    TEST("___________     +inf").with(fp) (j(+strf::pad0(infinity, 9)));
    TEST("___________      inf").with(fp) (j(strf::pad0(infinity, 9)));

    TEST("___________~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~').pad0(0).hex()));
    TEST("___________+inf~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~').pad0(8).hex()));
    TEST("___________~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~').pad0(8).hex()));
    TEST("__________~~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~').pad0(10).hex()));
    TEST("__________+inf~~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~').pad0(10).hex()));
    TEST("__________~~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~').pad0(10).hex()));

    TEST("___________~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~').pad0(9)));
    TEST("___________+inf~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~').pad0(9)));
    TEST("___________~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~').pad0(9)));
    TEST("__________~~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~').pad0(10)));
    TEST("__________+inf~~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~').pad0(10)));
    TEST("__________~~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~').pad0(10)));

    TEST("___________~~~~~~inf").with(fp) (j(strf::right (infinity, 9, '~').fill_sign()));
    TEST("___________~~~~~~inf").with(fp) (j(strf::right (infinity, 9, '~').fill_sign().pad0(8)));
    TEST("___________~inf~~~~~").with(fp) (j(strf::left  (infinity, 9, '~').fill_sign().pad0(8)));
    TEST("___________~~~inf~~~").with(fp) (j(strf::center(infinity, 9, '~').fill_sign().pad0(8)));
    TEST("__________~~~~~~~inf").with(fp) (j(strf::right (infinity, 9, '~').fill_sign().pad0(10)));
    TEST("__________~inf~~~~~~").with(fp) (j(strf::left  (infinity, 9, '~').fill_sign().pad0(10)));
    TEST("__________~~~~inf~~~").with(fp) (j(strf::center(infinity, 9, '~').fill_sign().pad0(10)));

    TEST("___________      inf").with(fp) (j(strf::pad0(infinity, 9).fill_sign()));
    TEST("________________ inf").with(fp) (j(strf::fmt(infinity).fill_sign()));

    TEST("___________~~~~~~inf").with(fp) (j(strf::right (infinity, 9, '~').hex().fill_sign()));
    TEST("___________~~~~~~inf").with(fp) (j(strf::right (infinity, 9, '~').hex().fill_sign().pad0(8)));
    TEST("___________~inf~~~~~").with(fp) (j(strf::left  (infinity, 9, '~').hex().fill_sign().pad0(8)));
    TEST("___________~~~inf~~~").with(fp) (j(strf::center(infinity, 9, '~').hex().fill_sign().pad0(8)));
    TEST("__________~~~~~~~inf").with(fp) (j(strf::right (infinity, 9, '~').hex().fill_sign().pad0(10)));
    TEST("__________~inf~~~~~~").with(fp) (j(strf::left  (infinity, 9, '~').hex().fill_sign().pad0(10)));
    TEST("__________~~~~inf~~~").with(fp) (j(strf::center(infinity, 9, '~').hex().fill_sign().pad0(10)));

    TEST("___________      inf").with(fp) (j(strf::pad0(infinity, 9).hex().fill_sign()));
    TEST("________________ inf").with(fp) (j(strf::fmt(infinity).hex().fill_sign()));

    TEST("_________________nan").with(fp, strf::lettercase::lower)  (j(nan));
    TEST("_________________inf").with(fp, strf::lettercase::lower)  (j(infinity));
    TEST("________________-inf").with(fp, strf::lettercase::lower)  (j(-infinity));
    TEST("_________________inf").with(fp, strf::lettercase::lower)  (j(strf::sci(infinity)));
    TEST("________________-inf").with(fp, strf::lettercase::lower)  (j(strf::sci(-infinity)));
    TEST("_________________inf").with(fp, strf::lettercase::lower)  (j(strf::hex(infinity)));
    TEST("________________-inf").with(fp, strf::lettercase::lower)  (j(strf::hex(-infinity)));
    TEST("_________________nan").with(fp, strf::lettercase::lower)  (j(strf::hex(nan)));

    TEST("_________________NaN").with(fp, strf::lettercase::mixed)  (j(nan));
    TEST("________________-NaN").with(fp, strf::lettercase::mixed)  (j(negative_nan));
    TEST("_________________Inf").with(fp, strf::lettercase::mixed)  (j(infinity));
    TEST("________________-Inf").with(fp, strf::lettercase::mixed)  (j(-infinity));
    TEST("_________________Inf").with(fp, strf::lettercase::mixed)  (j(strf::sci(infinity)));
    TEST("________________-Inf").with(fp, strf::lettercase::mixed)  (j(strf::sci(-infinity)));
    TEST("_________________Inf").with(fp, strf::lettercase::mixed)  (j(strf::hex(infinity)));
    TEST("________________-Inf").with(fp, strf::lettercase::mixed)  (j(strf::hex(-infinity)));
    TEST("_________________NaN").with(fp, strf::lettercase::mixed)  (j(strf::hex(nan)));

    TEST("_________________NAN").with(fp, strf::lettercase::upper)  (j(nan));
    TEST("________________-NAN").with(fp, strf::lettercase::upper)  (j(negative_nan));
    TEST("_________________INF").with(fp, strf::lettercase::upper)  (j(infinity));
    TEST("________________-INF").with(fp, strf::lettercase::upper)  (j(-infinity));
    TEST("_________________INF").with(fp, strf::lettercase::upper)  (j(strf::sci(infinity)));
    TEST("________________-INF").with(fp, strf::lettercase::upper)  (j(strf::sci(-infinity)));
    TEST("_________________INF").with(fp, strf::lettercase::upper)  (j(strf::hex(infinity)));
    TEST("________________-INF").with(fp, strf::lettercase::upper)  (j(strf::hex(-infinity)));
    TEST("_________________NAN").with(fp, strf::lettercase::upper)  (j(strf::hex(nan)));
}

STRF_TEST_FUNC void basic_tests()
{
    constexpr auto j = strf::join_right(20, '_');

    TEST("___________________0")  (j(0.0));
    TEST("__________________-0")  (j(-0.0));
    TEST("___________________1")  (j(1.0));
    TEST("__________________-1")  (j(-1.0));
    TEST("_________________1.5")  (j(1.5));
    TEST("_____6.103515625e-05")  (j(6.103515625e-05));
    TEST("____-6.103515625e-10")  (j(-6.103515625e-10));
    TEST("____6.103515625e-100")  (j(6.103515625e-100));
    TEST("_____6.103515625e-10")  (j(strf::sci(6.103515625e-10)));
    TEST("____6.103515625e-100")  (j(strf::sci(6.103515625e-100)));
    TEST("____6.103515625e+100")  (j(strf::sci(6.103515625e+100)));
    TEST("_______0.00048828125")  (j(0.00048828125));
    TEST("______2048.001953125")  (j(2048.001953125));

    TEST("___________________0") (j(strf::fmt(0.0)));
    TEST("__________________-0") (j(strf::fmt(-0.0)));
    TEST("___________________1") (j(strf::fmt(1.0)));
    TEST("__________________-1") (j(strf::fmt(-1.0)));
    TEST("_________________1.5") (j(strf::fmt(1.5)));
    TEST("_________12345678.25") (j(strf::fmt(12345678.25)));

    TEST("__________________+1") (j(+strf::fmt(1.0)));
    TEST("__________________-1") (j(+strf::fmt(-1.0)));
    TEST("__________________1.") (j(*strf::fmt(1.0)));
    TEST("_________________-1.") (j(*strf::fmt(-1.0)));
    TEST("_________________+1.")(j(*+strf::fmt(1.0)));
    TEST("_________________-1.")(j(+*strf::fmt(-1.0)));

    TEST("_____________+0.0001") (j(+strf::fixed(0.0001)));
    TEST("_______+0.0001000000") (j(+strf::fixed(0.0001).p(10)));


    TEST("_______________1e+50").with(strf::lettercase::lower)  (j(1e+50));
    TEST("_______________1e+50").with(strf::lettercase::mixed)  (j(1e+50));
    TEST("_______________1E+50").with(strf::lettercase::upper)  (j(1e+50));

    TEST("______________ 1e+50").with(strf::lettercase::lower)  (j(strf::sci(1e+50)>6));
    TEST("______________ 1e+50").with(strf::lettercase::mixed)  (j(strf::sci(1e+50)>6));
    TEST("______________ 1E+50").with(strf::lettercase::upper)  (j(strf::sci(1e+50)>6));

    //----------------------------------------------------------------
    // when precision is not specified, the general format selects the
    // scientific notation if it is shorter than the fixed notation:
    TEST("_______________10000")  (j(1e+4));
    TEST("_______________1e+05")  (j(1e+5));
    TEST("_____________1200000")  (j(1.2e+6));
    TEST("_____________1.2e+07")  (j(1.2e+7));
    TEST("_______________0.001")  (j(1e-03));
    TEST("_______________1e-04")  (j(1e-04));
    TEST("_____________0.00012")  (j(1.2e-04));
    TEST("_____________1.2e-05")  (j(1.2e-05));
    TEST("____________0.000123")  (j(1.23e-04));
    TEST("____________1.23e-05")  (j(1.23e-05));
    TEST("_______________10000")  (j(strf::fmt(1e+4)));
    TEST("______________10000.") (j(*strf::fmt(1e+4)));
    TEST("_______________1e+05")  (j(strf::fmt(1e+5)));
    TEST("______________1.e+05") (j(*strf::fmt(1e+5)));
    TEST("_____________1200000")  (j(strf::fmt(1.2e+6)));
    TEST("_____________1.2e+06") (j(*strf::fmt(1.2e+6)));
    TEST("_______________0.001")   (j(strf::fmt(1e-03)));
    TEST("_______________1e-04")   (j(strf::fmt(1e-04)));
    TEST("______________0.0001")  (j(*strf::fmt(1e-04)));
    TEST("______________1.e-05")  (j(*strf::fmt(1e-05)));
    TEST("_____________0.00012")   (j(strf::fmt(1.2e-04)));
    TEST("_____________0.00012") (j(*strf::fmt(1.2e-04)));
    TEST("_____________1.2e-05")  (j(strf::fmt(1.2e-05)));
    TEST("_____________1.2e-05") (j(*strf::fmt(1.2e-05)));
    TEST("____________0.000123")  (j(strf::fmt(1.23e-04)));
    TEST("____________1.23e-05")  (j(strf::fmt(1.23e-05)));
    TEST("_____6.103515625e-05")    (j(strf::fmt(6.103515625e-05)));
    TEST("_______0.00048828125")    (j(strf::fmt(0.00048828125)));

    TEST("_____________1.e+100") (j(*strf::fmt(1e+100)));
    TEST("_____________1.e-100") (j(*strf::fmt(1e-100)));

    //----------------------------------------------------------------


    //----------------------------------------------------------------
    // when precision is specified in the general format, do as in printf:
    // - The precision specifies the number of significant digits.
    // - scientific notation is used if the resulting exponent is
    //   less than -4 or greater than or equal to the precision.

    TEST("_____________0.00012")  (j(strf::fmt(0.000123).p(2)));
    TEST("_____________1.2e-05")  (j(strf::fmt(0.0000123).p(2)));
    TEST("_______________1e+01") (j(strf::fmt(12.0).p(1)));
    TEST("_____________1.2e+02") (j(strf::fmt(123.0).p(2)));
    TEST("_________________120") (j(strf::fmt(120.0).p(4)));
    TEST("_________________120") (j(strf::fmt(120.1).p(3)));
    TEST("_______________1e+04") (j(strf::fmt(10000.0).p(4)));
    TEST("_______________10000") (j(strf::fmt(10000.0).p(5)));
    TEST("_______________10000") (j(strf::fmt(10000.0).p(6)));
    TEST("__________6.1035e-05") (j(strf::fmt(6.103515625e-05).p(5)));

    // and if precision is zero, it treated as 1.
    TEST("_______________1e+01")  (j(strf::fmt(12.0).p(0)));
    TEST("_______________2e+01")  (j(strf::fmt(15.125).p(0)));

    // and when removing digits, the last digit is rounded.
    TEST("_____________1.1e+05") (j(strf::fmt(114999.0).p(2)));
    TEST("_____________1.2e+05") (j(strf::fmt(115000.0).p(2)));
    TEST("_____________1.2e+05") (j(strf::fmt(125000.0).p(2)));
    TEST("_____________1.3e+05") (j(strf::fmt(125001.0).p(2)));

    // and the decimal point appears only if followed by
    // a digit, or if operator*() is used.
    TEST("_______________1e+04")   (j(strf::fmt(10000.0).p(3)));
    TEST("______________1.e+04")  (j(*strf::fmt(10000.0).p(1)));
    TEST("________________123.")  (j(*strf::fmt(123.0).p(3)));

    // and trailing zeros are removed, unless operator*() is used.
    TEST("_____________1.5e+04")   (j(strf::fmt(15000.0).p(3)));
    TEST("____________1.50e+04")  (j(*strf::fmt(15000.0).p(3)));
    TEST("_____________1.5e+04")  (j(strf::fmt(15001.0).p(3)));
    TEST("_____________1.5e+04")  (j(*strf::fmt(15001.0).p(3)));
    TEST("_________________123")   (j(strf::fmt(123.0).p(5)));
    TEST("______________123.00")  (j(*strf::fmt(123.0).p(5)));
    TEST("______________1000.5") (j(!strf::fmt(1000.5).p(6)));

    // test rounding
    TEST("_________________2.2")  (j(strf::fmt(2.25).p(2)));
    TEST("_________________2.3")  (j(strf::fmt(2.25000001).p(2)));
    TEST("________________2.25")  (j(strf::fmt(2.25000001).p(3)));
    TEST("_____________2.2e+15")  (j(strf::fmt(2.25e+15).p(2)));
    TEST("____________2.3e+100")  (j(strf::fmt(2.250001e+100).p(2)));
    TEST("___________2.25e-100")  (j(strf::fmt(2.250001e-100).p(3)));


    //----------------------------------------------------------------
    // strf::fixed

    TEST("__________________1.")  (j(*strf::fixed(1.0)));
    TEST("___________________1")   (j(strf::fixed(1.0)));
    TEST("__________________+1")  (j(+strf::fixed(1.0)));
    TEST("__________________-1")   (j(strf::fixed(-1.0)));
    TEST("__________________-1")  (j(+strf::fixed(-1.0)));

    TEST("___________________1")  (j(strf::fixed(1.0).p(0)));
    TEST("__________________1.") (j(*strf::fixed(1.0).p(0)));
    TEST("_________________1.0")  (j(strf::fixed(1.0).p(1)));
    TEST("________________1.00")  (j(strf::fixed(1.0).p(2)));
    TEST("______________1.0000")  (j(strf::fixed(1.0).p(4)));
    TEST("______________1.2500")  (j(strf::fixed(1.25).p(4)));
    TEST("____________1.001000")  (j(strf::fixed(1.001).p(6)));
    TEST("_______________0.000")  (j(strf::fixed(1e-30).p(3)));
    TEST("_______________0.125")  (j(strf::fixed(0.125)));

    // test rounding
    TEST("_________________2.2")  (j(strf::fixed(2.25).p(1)));
    TEST("_________________2.3")  (j(strf::fixed(2.25000001).p(1)));
    TEST("________________2.25")  (j(strf::fixed(2.25000001).p(2)));
    TEST("______________0.0001")  (j(strf::fixed(0.0000501).p(4)));
    TEST("______________0.0000")  (j(strf::fixed(0.00004999).p(4)));
    TEST("_______________0.000")  (j(strf::fixed(0.0000999).p(3)));

    //----------------------------------------------------------------
    // strf::sci

    TEST("______________0.e+00")  (j(*strf::sci(0.0)));
    TEST("______________1.e+04")  (j(*strf::sci(1e+4)));
    TEST("_____________+1.e+04") (j(*+strf::sci(1e+4)));
    TEST("_____________-1.e+04") (j(*strf::sci(-1e+4)));

    TEST("_____________1.0e+04")   (j(strf::sci(1e+4).p(1)));
    TEST("_____________1.0e+04")   (j(strf::sci(1e+4).p(1)));
    TEST("___________+1.00e+04")  (j(+strf::sci(1e+4).p(2)));
    TEST("______________1.e+04")  (j(*strf::sci(1e+4).p(0)));
    TEST("_____________+1.e+04") (j(*+strf::sci(1e+4).p(0)));
    TEST("______________-1e+04")  (j(+strf::sci(-1e+4).p(0)));
    TEST("_____________-1.e+04")  (j(*strf::sci(-1e+4).p(0)));

    TEST("____________1.25e+02") (j(strf::sci(125.0).p(2)));
    TEST("_____________1.2e+02") (j(strf::sci(125.0).p(1)));
    TEST("_____________1.2e+02") (j(strf::sci(115.0).p(1)));
    TEST("_____________1.3e+06") (j(strf::sci(1250001.0).p(1)));
    TEST("__________8.1928e+03") (j(strf::sci(8192.75).p(4)));
    TEST("__________8.1922e+03") (j(strf::sci(8192.25).p(4)));
    TEST("__________1.0242e+03") (j(strf::sci(1024.25).p(4)));
    TEST("_____________1.7e+01") (j(strf::sci(16.50006103515625).p(1)));
    TEST("___________1.250e+02") (j(strf::sci(125.0).p(3)));
    TEST("_________6.25000e-02") (j(strf::sci(0.0625).p(5)));
    TEST("________8.192750e+03") (j(strf::sci(8192.75).p(6)));

    TEST("____________2.2e+100") (j(strf::sci(2.25e+100).p(1)));
    TEST("____________2.3e-100") (j(strf::sci(2.250001e-100).p(1)));
    TEST("____________2.25e+15") (j(strf::sci(2.250001e+15).p(2)));



    // ---------------------------------------------------------------
    // alignment
    TEST("_________******-1.25") (j(strf::right(-1.25, 11, '*')));
    TEST("_________-1.25******") (j(strf::left(-1.25, 11, '*')));
    TEST("_________***-1.25***") (j(strf::center(-1.25, 11, '*')));

    TEST("_________-0000001.25") (j(strf::pad0(-1.25, 11)));
    TEST("_________+0000001.25") (j(+strf::pad0(1.25, 11)));

    TEST("_______________-1.25") (j(strf::right(-1.25, 5, '*')));
    TEST("_______________-1.25") (j(strf::left(-1.25, 5, '*')));
    TEST("_______________-1.25") (j(strf::center(-1.25, 5, '*')));

    TEST("_____________\xEF\xBF\xBD\xEF\xBF\xBD-1.25")
        (j(strf::right(-1.25, 7, static_cast<char32_t>(0xFFFFFFF))));
    TEST("_____________-1.25\xEF\xBF\xBD\xEF\xBF\xBD")
        (j(strf::left(-1.25, 7, static_cast<char32_t>(0xFFFFFFF))));
    TEST("_____________\xEF\xBF\xBD-1.25\xEF\xBF\xBD")
        (j(strf::center(-1.25, 7, static_cast<char32_t>(0xFFFFFFF))));

    //----------------------------------------------------------------
    // pad0
    TEST("________000000000001") (j(strf::pad0(1.0, 12)));
    TEST("________+0000000001.") (j(+*strf::pad0(1.0, 12)));
    TEST("______  +0000000001.") (j(+*strf::pad0(1.0, 12) > 14));
    TEST("______~~+0000000001.") (j(+*strf::pad0(1.0, 12).fill('~') > 14));
    TEST("________ 0000000001.") (j(*strf::pad0(1.0, 12).fill_sign()));
    TEST("______   0000000001.") (j(*strf::pad0(1.0, 12).fill_sign() > 14));
    TEST("________+00001234.25") (j(+strf::pad0(1234.25, 12)));
    TEST("________+001234.2500") (j(+*strf::pad0(1234.25, 12).fixed().p(4)));
    TEST("________00000001e+20") (j(strf::pad0(1e+20, 12)));
    TEST("________+000001.e+20") (j(+*strf::pad0(1.e+20, 12)));
    TEST("________ 000001.e+20") (j(*strf::pad0(1.e+20, 12).fill_sign()));
    TEST("________00001.25e+20") (j(strf::pad0(1.25e+20, 12)));


    //----------------------------------------------------------------
    // fill_sign
    TEST("__________000001.125") (j(strf::pad0(1.125, 10)));
    TEST("__________ 00001.125") (j(strf::pad0(1.125, 10).fill_sign()));
    TEST("_______~~~~1.125~~~~") (j(strf::center(1.125, 13, '~').fill_sign()));
    TEST("______~~~~~1.125~~~~") (j(strf::center(1.125, 14, '~').fill_sign()));
    TEST("______~1.125~~~~~~~~") (j(strf::left(1.125, 14, '~').fill_sign()));
    TEST("______~~~~~~~~~1.125") (j(strf::right(1.125, 14, '~').fill_sign()));

    TEST("______~~000001.125~~") (j(strf::center(1.125, 14, '~').pad0(10)));
    TEST("______~~~00001.125~~") (j(strf::center(1.125, 14, '~').pad0(10).fill_sign()));
    TEST("______~~~~~00001.125") (j(strf::right(1.125, 14, '~').pad0(10).fill_sign()));
    TEST("______~00001.125~~~~") (j(strf::left(1.125, 14, '~').pad0(10).fill_sign()));

    TEST("____________1.125000") (j(strf::fixed(1.125, 6)));
    TEST("___________ 1.125000") (j(strf::fixed(1.125, 6).fill_sign()));
}

STRF_TEST_FUNC void test_float32()
{
    constexpr auto j = strf::join_right(25, '_');
    TEST("_______________________+0") (j(+strf::fixed(0.0F)));
    TEST("___________+1.1754944e-38") (j(+*strf::gen(+1.1754944e-38)));
    TEST("____________________+1.25") (j(+strf::fixed(1.25F)));
    TEST("________________+1.250000") (j(+strf::fixed(1.25F).p(6)));
    TEST("_____________________+0.1") (j(+strf::fixed(0.1F)));
    TEST("__________________+1.e+20") (j(+*strf::sci(1e+20F)));
    TEST("____________+1.000000e+20") (j(+strf::sci(1e+20F).p(6)));
    TEST("____________________+1.25") (j(+strf::gen(1.25F)));
    TEST("___________________+1.250") (j(+*strf::gen(1.25F).p(4)));

    TEST("_______________0x1.ffcp+0") (j(strf::hex(0x1.ffcp+0F)));
    TEST("__________0x1.8abcdep+127") (j(strf::hex(0x1.8abcdecp+127F)));
    TEST("_________-0x1.8abcdep-126") (j(strf::hex(-0x1.8abcdecp-126F)));
    float denorm_min = strf::detail::bit_cast<float>(static_cast<std::uint32_t>(1));
    TEST("__________0x0.000002p-126") (j(strf::hex(denorm_min)));
    TEST("__________0x1.fffffep+127") (j(strf::hex(float_max<float>())));
}

STRF_TEST_FUNC void test_hexadecimal()
{
    constexpr auto j = strf::join_right(25, '_');

    TEST("0x0p+0") (strf::hex(0.0));
    TEST("___________________0x0p+0") (j(strf::hex(0.0)));
    TEST("__________________+0x0p+0") (j(+strf::hex(0.0)));
    TEST("__________________0x0.p+0") (j(*strf::hex(0.0)));
    TEST("_________________+0x0.p+0") (j(+*strf::hex(0.0)));
    TEST("_______________0x0.000p+0") (j(strf::hex(0.0).p(3)));
    TEST("__________________-0x1p-3") (j(strf::hex(-0.125)));
    TEST("__________________0x1p+11") (j(strf::hex(2048.0)));
    TEST("__0x1.fffffffffffffp+1023") (j(strf::hex(float_max<double>())));
    auto denorm_min = strf::detail::bit_cast<double>(static_cast<std::uint64_t>(1));
    TEST("__0x0.0000000000001p-1022") (j(strf::hex(denorm_min)));
    TEST("________________0x1p-1022") (j(strf::hex(0x1p-1022)));
    TEST("_______________0x1.p-1022") (j(*strf::hex(0x1p-1022)));
    TEST("____________0x1.000p-1022") (j(*strf::hex(0x1p-1022).p(3)));
    TEST("___________0x0.0009p-1022") (j(strf::hex(0x0.0009p-1022)));

    TEST("_________________0x1.8p+0") (j(strf::hex(0x1.8p+0)));
    TEST("_________________0x1.cp+0") (j(strf::hex(0x1.cp+0)));
    TEST("_________________0x1.ep+0") (j(strf::hex(0x1.ep+0)));
    TEST("_________________0x1.fp+0") (j(strf::hex(0x1.fp+0)));
    TEST("________________0x1.f8p+0") (j(strf::hex(0x1.f8p+0)));
    TEST("________________0x1.fcp+0") (j(strf::hex(0x1.fcp+0)));
    TEST("________________0x1.fep+0") (j(strf::hex(0x1.fep+0)));
    TEST("________________0x1.ffp+0") (j(strf::hex(0x1.ffp+0)));
    TEST("_______________0x1.ff8p+0") (j(strf::hex(0x1.ff8p+0)));
    TEST("_______________0x1.ffcp+0") (j(strf::hex(0x1.ffcp+0)));
    TEST("_______________0x1.ffep+0") (j(strf::hex(0x1.ffep+0)));
    TEST("_______________0x1.fffp+0") (j(strf::hex(0x1.fffp+0)));
    TEST("______________0x1.fff8p+0") (j(strf::hex(0x1.fff8p+0)));
    TEST("______________0x1.fffcp+0") (j(strf::hex(0x1.fffcp+0)));
    TEST("______________0x1.fffep+0") (j(strf::hex(0x1.fffep+0)));
    TEST("______________0x1.ffffp+0") (j(strf::hex(0x1.ffffp+0)));
    TEST("_____________0x1.ffff8p+0") (j(strf::hex(0x1.ffff8p+0)));
    TEST("_____________0x1.ffffcp+0") (j(strf::hex(0x1.ffffcp+0)));
    TEST("_____________0x1.ffffep+0") (j(strf::hex(0x1.ffffep+0)));
    TEST("_____________0x1.fffffp+0") (j(strf::hex(0x1.fffffp+0)));
    TEST("____________0x1.fffff8p+0") (j(strf::hex(0x1.fffff8p+0)));
    TEST("____________0x1.fffffcp+0") (j(strf::hex(0x1.fffffcp+0)));
    TEST("____________0x1.fffffep+0") (j(strf::hex(0x1.fffffep+0)));
    TEST("____________0x1.ffffffp+0") (j(strf::hex(0x1.ffffffp+0)));
    TEST("___________0x1.ffffff8p+0") (j(strf::hex(0x1.ffffff8p+0)));
    TEST("___________0x1.ffffffcp+0") (j(strf::hex(0x1.ffffffcp+0)));
    TEST("___________0x1.ffffffep+0") (j(strf::hex(0x1.ffffffep+0)));
    TEST("___________0x1.fffffffp+0") (j(strf::hex(0x1.fffffffp+0)));
    TEST("__________0x1.fffffff8p+0") (j(strf::hex(0x1.fffffff8p+0)));
    TEST("__________0x1.fffffffcp+0") (j(strf::hex(0x1.fffffffcp+0)));
    TEST("__________0x1.fffffffep+0") (j(strf::hex(0x1.fffffffep+0)));
    TEST("__________0x1.ffffffffp+0") (j(strf::hex(0x1.ffffffffp+0)));
    TEST("_________0x1.ffffffff8p+0") (j(strf::hex(0x1.ffffffff8p+0)));
    TEST("_________0x1.ffffffffcp+0") (j(strf::hex(0x1.ffffffffcp+0)));
    TEST("_________0x1.ffffffffep+0") (j(strf::hex(0x1.ffffffffep+0)));
    TEST("_________0x1.fffffffffp+0") (j(strf::hex(0x1.fffffffffp+0)));
    TEST("________0x1.fffffffff8p+0") (j(strf::hex(0x1.fffffffff8p+0)));
    TEST("________0x1.fffffffffcp+0") (j(strf::hex(0x1.fffffffffcp+0)));
    TEST("________0x1.fffffffffep+0") (j(strf::hex(0x1.fffffffffep+0)));
    TEST("________0x1.ffffffffffp+0") (j(strf::hex(0x1.ffffffffffp+0)));
    TEST("_______0x1.ffffffffff8p+0") (j(strf::hex(0x1.ffffffffff8p+0)));
    TEST("_______0x1.ffffffffffcp+0") (j(strf::hex(0x1.ffffffffffcp+0)));
    TEST("_______0x1.ffffffffffep+0") (j(strf::hex(0x1.ffffffffffep+0)));
    TEST("_______0x1.fffffffffffp+0") (j(strf::hex(0x1.fffffffffffp+0)));
    TEST("______0x1.fffffffffff8p+0") (j(strf::hex(0x1.fffffffffff8p+0)));
    TEST("______0x1.fffffffffffcp+0") (j(strf::hex(0x1.fffffffffffcp+0)));
    TEST("______0x1.fffffffffffep+0") (j(strf::hex(0x1.fffffffffffep+0)));
    TEST("______0x1.ffffffffffffp+0") (j(strf::hex(0x1.ffffffffffffp+0)));
    TEST("_____0x1.ffffffffffff8p+0") (j(strf::hex(0x1.ffffffffffff8p+0)));
    TEST("_____0x1.ffffffffffffcp+0") (j(strf::hex(0x1.ffffffffffffcp+0)));
    TEST("_____0x1.ffffffffffffep+0") (j(strf::hex(0x1.ffffffffffffep+0)));
    TEST("_____0x1.fffffffffffffp+0") (j(strf::hex(0x1.fffffffffffffp+0)));

    TEST("___________________0x1p+0") (j(strf::hex(0x1.12345p+0).p(0)));
    TEST("_____________0x1.12345p+0") (j(strf::hex(0x1.12345p+0).p(5)));
    TEST("____________0x1.123450p+0") (j(strf::hex(0x1.12345p+0).p(6)));
    TEST("__________________0x1.p+0") (j(*strf::hex(0x1.12345p+0).p(0)));
    TEST("_________________+0x1.p+0") (j(+*strf::hex(0x1.12345p+0).p(0)));
    TEST("____________0x0.000p-1022") (j(strf::hex(0x0.0008p-1022).p(3)));
    TEST("____________0x0.002p-1022") (j(strf::hex(0x0.0018p-1022).p(3)));
    TEST("____________0x0.001p-1022") (j(strf::hex(0x0.0008000000001p-1022).p(3)));

    TEST("___________________0X0P+0").with(strf::lettercase::upper) (j(strf::hex(0.0)));
    TEST("_________0X0.ABCDEFP-1022").with(strf::lettercase::upper) (j(strf::hex(0x0.abcdefp-1022)));
    TEST("___________0X1.ABCDEFP+10").with(strf::lettercase::upper) (j(strf::hex(0x1.abcdefp+10)));

    TEST("___________________0x0p+0").with(strf::lettercase::mixed) (j(strf::hex(0.0)));
    TEST("_________0x0.ABCDEFp-1022").with(strf::lettercase::mixed) (j(strf::hex(0x0.abcdefp-1022)));
    TEST("___________0x1.ABCDEFp+10").with(strf::lettercase::mixed) (j(strf::hex(0x1.abcdefp+10)));

    TEST("___________________0x0p+0").with(strf::lettercase::lower) (j(strf::hex(0.0)));
    TEST("_________0x0.abcdefp-1022").with(strf::lettercase::lower) (j(strf::hex(0x0.abcdefp-1022)));
    TEST("___________0x1.abcdefp+10").with(strf::lettercase::lower) (j(strf::hex(0x1.abcdefp+10)));

    TEST("______________-0x1p+0****") (j(strf::left(-1.0, 11, '*').hex()));
    TEST("______________****-0x1p+0") (j(strf::right(-1.0, 11, '*').hex()));
    TEST("______________**-0x1p+0**") (j(strf::center(-1.0, 11, '*').hex()));
    TEST("__________________-0x1p+0") (j(strf::center(-1.0, 7, '*').hex()));

    // pad0
    TEST("______________-0x00001p+0") (j(strf::pad0(-1.0, 11).hex()));
    TEST("______________+0x0001.p+0") (j(+*strf::pad0(1.0, 11).hex()));
    TEST("__________0x001.123450p+0") (j(strf::hex(0x1.12345p+0).p(6).pad0(15)));
    TEST("______**0x001.123450p+0**") (j(strf::hex(0x1.12345p+0).p(6).pad0(15).fill('*') ^ 19));
    TEST("__________0x001.123450p+0") (j(strf::hex(0x1.12345p+0).p(6).pad0(15).fill('*') ^ 15));

    // fill_sign
    TEST("_____________0x001.125p+1") (j(strf::hex(0x1.125p+1).pad0(12)));
    TEST("______________ 0x1.125p+1") (j(strf::hex(0x1.125p+1).fill_sign()));

    TEST("_____________ 0x01.125p+1") (j(strf::hex(0x1.125p+1).pad0(12).fill_sign()));
    TEST("_____________*0x01.125p+1") (j(strf::hex(0x1.125p+1).pad0(12).fill_sign().fill('*')));
    TEST("_________*****0x01.125p+1") (j(strf::right(0x1.125p+1, 16, '*').hex().pad0(12).fill_sign()));
    TEST("_________***0x01.125p+1**") (j(strf::center(0x1.125p+1, 16, '*').hex().pad0(12).fill_sign()));
    TEST("_________*0x01.125p+1****") (j(strf::left(0x1.125p+1, 16, '*').hex().pad0(12).fill_sign()));

    TEST("_________******0x1.125p+1") (j(strf::right(0x1.125p+1, 16, '*').hex().fill_sign()));
    TEST("_________***0x1.125p+1***") (j(strf::center(0x1.125p+1, 16, '*').hex().fill_sign()));
    TEST("_________*0x1.125p+1*****") (j(strf::left(0x1.125p+1, 16, '*').hex().fill_sign()));

}

template <int Base>
struct numpunct_maker {

    STRF_HD numpunct_maker(char32_t sep)
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
        TEST("_____________1:000,5").with(p) (j(strf::punct(1000.5)));
        TEST("_____________1,5e+50").with(p) (j(strf::punct(1.5e+50)));
        TEST("____________1:000,50").with(p) (j(!strf::fixed(1000.5).p(2)));
        TEST("____________1:000,50").with(p) (j(*!strf::fmt(1000.5).p(6)));
        TEST("_____________1:000,5").with(p) (j(!strf::fmt(1000.5).p(6)));
        TEST("____________1,50e+50").with(p) (j(!strf::sci(1.5e+50).p(2)));
        TEST("__________________1,").with(p) (j(*!strf::fmt(1.0)));

        TEST("_________+1:000:000,").with(p) (j(*+!strf::fixed(1000000.0)));
        TEST("_____+1:000:000,0000").with(p) (j(*+!strf::fixed(1000000.0).p(4)));
        TEST("__+00000001:000:000,").with(p) (j(*+!strf::fixed(1000000.0).pad0(18)));

        auto px = strf::numpunct<16>{3}.decimal_point(',');
        TEST("________0x1,12345p+0").with(px) (j(!strf::hex(0x1.12345p+0)));
    }
    {   //encoding big punct characters
        auto p = strf::numpunct<10>{3}.thousands_sep(0x10AAAA).decimal_point(0x10FFFF);

        TEST(u8"_____________1\U0010AAAA" u8"000\U0010FFFF" u8"5").with(p)
            (j(strf::punct(1000.5)));
        TEST(u8"_____________1\U0010FFFF" u8"5e+50")              .with(p)
            (j(strf::punct(1.5e+50)));
        TEST(u8"____________1\U0010AAAA" u8"000\U0010FFFF" u8"50").with(p)
            (j(!strf::fixed(1000.5).p(2)));
        TEST(u8"______________1\U0010FFFF" u8"e+50") .with(p) (j(*!strf::sci(1e+50)));
        TEST(u8"____________1\U0010FFFF" u8"50e+50") .with(p) (j(!strf::sci(1.5e+50).p(2)));
        TEST(u8"__________________1\U0010FFFF")      .with(p) (j(*!strf::fmt(1.0)));
        TEST(u8"_________________0\U0010FFFF" u8"1") .with(p) (j(!strf::fixed(0.1)));
        TEST(u8"_________________0\U0010FFFF" u8"1") .with(p) (j(strf::punct(0.1)));

        //in hexadecimal
        auto px = strf::numpunct<16>{3}.decimal_point(0x10FFFF);
        TEST(u8"________0x1\U0010FFFF" u8"12345p+0").with(px)
            (j(!strf::hex(0x1.12345p+0)));
    }
    {   // encoding punct chars in a single-byte encoding
        auto fp = strf::pack
            ( strf::iso_8859_3_t<char>{}
            , strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130) );

        TEST("_____________1\xFF""000\xA9""5").with(fp) (j(strf::punct(1000.5)));
        TEST("_____________1\xA9""5e+50")     .with(fp) (j(strf::punct(1.5e+50)));
        TEST("____________1\xFF""000\xA9""50").with(fp) (j(!strf::fixed(1000.5).p(2)));
        TEST("____________1\xA9""50e+50")     .with(fp) (j(!strf::sci(1.5e+50).p(2)));
        TEST("__________________1\xA9")       .with(fp) (j(*!strf::fmt(1.0)));
        TEST("_________________0\xA9" "1")    .with(fp) (j(!strf::fixed(0.1)));
        TEST("_________________0\xA9" "1")    .with(fp) (j(strf::punct(0.1)));

        auto fpx = strf::pack
            ( strf::iso_8859_3_t<char>{}, strf::numpunct<16>(4).decimal_point(0x130) );

        TEST("________0x1\xA9""12345p+0").with(fpx) (j(!strf::hex(0x1.12345p+0)));
    }
    {   // invalid punct characters
        // ( thousand separators are omitted  )

        auto p = strf::numpunct<10>{3}.thousands_sep(0xFFFFFF).decimal_point(0xEEEEEE);
        TEST(u8"______________1000\uFFFD5")       .with(p) (j(strf::punct(1000.5)));
        TEST(u8"_____________1\uFFFD" u8"5e+50")  .with(p) (j(strf::punct(1.5e+50)));
        TEST(u8"_____________1000\uFFFD" u8"50")  .with(p) (j(!strf::fixed(1000.5).p(2)));
        TEST(u8"____________1\uFFFD" u8"50e+50")  .with(p) (j(!strf::sci(1.5e+50).p(2)));
        TEST(u8"__________________1\uFFFD")       .with(p) (j(*!strf::fmt(1.0)));
        TEST(u8"_________________0\uFFFD" u8"1")  .with(p) (j(!strf::fixed(0.1)));
        TEST(u8"_________________0\uFFFD" u8"1")  .with(p) (j(strf::punct(0.1)));

        auto px = strf::numpunct<16>{3}.decimal_point(0xEEEEEE);
        TEST(u8"________0x1\uFFFD" u8"12345p+0").with(px) (j(!strf::hex(0x1.12345p+0)));
    }
    {   // invalid punct characters  in a single-byte encoding
        // ( thousand separators are omitted  )
        auto fp = strf::pack
            ( strf::iso_8859_3_t<char>{}
            , strf::numpunct<10>(3).thousands_sep(0xFFF).decimal_point(0xEEE) );

        TEST("______________1000?5").with(fp) (j(strf::punct(1000.5)));
        TEST("_____________1?5e+50").with(fp) (j(strf::punct(1.5e+50)));
        TEST("_____________1000?50").with(fp) (j(!strf::fixed(1000.5).p(2)));
        TEST("____________1?50e+50").with(fp) (j(!strf::sci(1.5e+50).p(2)));
        TEST("__________________1?").with(fp) (j(*!strf::fmt(1.0)));
        TEST("_________________0?1").with(fp) (j(!strf::fixed(0.1)));
        TEST("_________________0?1").with(fp) (j(strf::punct(0.1)));

        auto fpx = strf::pack
            ( strf::iso_8859_3_t<char>{}, strf::numpunct<16>(4).decimal_point(0xEEE) );

        TEST("________0x1?12345p+0").with(fpx) (j(!strf::hex(0x1.12345p+0)));
    }

    {   // When the integral part does not have trailing zeros

        TEST("1:048:576") .with(strf::numpunct<10>(3).thousands_sep(':'))
            (strf::punct(1048576.0));

        TEST("1:048:576") .with(strf::numpunct<10>(3).thousands_sep(':'))
            (!strf::fixed(1048576.0));

        TEST(u8"1\u0ABC" u8"048\u0ABC" u8"576")
            .with(strf::numpunct<10>(3).thousands_sep(0xABC))
            (strf::punct(1048576.0));

        TEST(u8"1\u0ABC" u8"048\u0ABC" u8"576")
            .with(strf::numpunct<10>(3).thousands_sep(0xABC))
            (!strf::fixed(1048576.0));
    }
    {   // variable groups

        using grp = strf::numpunct<10>;

        TEST("1,00,00,0,00,0")  .with(grp(1,2,1,2)) (!strf::fixed(100000000.0));
        TEST("10,00,00,0,00,0") .with(grp(1,2,1,2)) (!strf::fixed(1000000000.0));

        TEST("32,10,00,0,00,0") .with(grp(1,2,1,2)) (!strf::fixed(3210000000.0));
        TEST("43,21,00,0,00,0") .with(grp(1,2,1,2)) (!strf::fixed(4321000000.0));
        TEST("54,32,10,0,00,0") .with(grp(1,2,1,2)) (!strf::fixed(5432100000.0));
        TEST("7,65,432,10,0")   .with(grp(1,2,3,2)) (!strf::fixed(765432100.0));
        TEST("7,6,543,21,0,0")  .with(grp(1,1,2,3,1)) (!strf::fixed(765432100.0));
        TEST("7,654,321,00,0")  .with(grp(1,2,3)) (!strf::fixed(7654321000.0));

        TEST("1000,00,0") .with(grp(1,2,-1)) (!strf::fixed(1000000.0));
        TEST("1234,00,0") .with(grp(1,2,-1)) (!strf::fixed(1234000.0));
        TEST("1234,50,0") .with(grp(1,2,-1)) (!strf::fixed(1234500.0));
    }
    {   // variable groups and big separator
        numpunct_maker<10> grp{0xABCD};

        TEST(u8"1\uABCD" u8"00\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (!strf::fixed(100000000.0));
        TEST(u8"10\uABCD" u8"00\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (!strf::fixed(1000000000.0));

        TEST(u8"32\uABCD" u8"10\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (!strf::fixed(3210000000.0));
        TEST(u8"43\uABCD" u8"21\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (!strf::fixed(4321000000.0));
        TEST(u8"54\uABCD" u8"32\uABCD" u8"10\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (!strf::fixed(5432100000.0));
        TEST(u8"7\uABCD" u8"65\uABCD" u8"432\uABCD" u8"10\uABCD" u8"0")
              .with(grp(1,2,3,2)) (!strf::fixed(765432100.0));
        TEST(u8"7\uABCD" u8"6\uABCD" u8"543\uABCD" u8"21\uABCD" u8"0\uABCD" u8"0")
              .with(grp(1,1,2,3,1)) (!strf::fixed(765432100.0));
        TEST(u8"7\uABCD" u8"654\uABCD" u8"321\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,3)) (!strf::fixed(7654321000.0));

        TEST(u8"1000\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,-1)) (!strf::fixed(1000000.0));
        TEST(u8"1234\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,-1)) (!strf::fixed(1234000.0));
        TEST(u8"1234\uABCD" u8"50\uABCD" u8"0")
              .with(grp(1,2,-1)) (!strf::fixed(1234500.0));
    }
    {   // when precision is not specified, the general format selects the
        // scientific notation if it is shorter than the fixed notation:
        auto p1 = strf::numpunct<10>{1};

        TEST("_______________1,0,0").with(p1)  (j(strf::punct(1e+2)));
        TEST("_______________1e+03").with(p1)  (j(strf::punct(1e+3)));
        TEST("_____________1,2,0,0").with(p1)  (j(strf::punct(1.2e+3)));
        TEST("_____________1.2e+04").with(p1)  (j(strf::punct(1.2e+4)));
        TEST("______________+1,0,0").with(p1)  (j(+strf::punct(1e+2)));
        TEST("______________1,0,0.").with(p1) (j(*strf::punct(1e+2)));
        TEST("_______________1e+03").with(p1)  (j(strf::punct(1e+3)));
        TEST("______________1.e+03").with(p1) (j(*strf::punct(1e+3)));
        TEST("_____________1,2,0,0").with(p1)  (j(strf::punct(1.2e+3)));
        TEST("_____________1.2e+03").with(p1) (j(*strf::punct(1.2e+3)));
        TEST("_______________1e-04").with(p1)  (j(strf::punct(1e-4)));
        TEST("_______________0.001").with(p1)  (j(strf::punct(1e-3)));
        TEST("_______________1e-04").with(p1)  (j(strf::punct(1e-4)));
        TEST("_______________0.001").with(p1)  (j(strf::punct(1e-3)));
        TEST("_______________1e+05").with(strf::numpunct<10>(8))(j(strf::punct(1e+5)));
        TEST("________1.000005e+05").with(p1) (j(strf::punct(100000.5)));
        TEST("_________1,0,0,0,0.5").with(p1) (j(strf::punct(10000.5)));
    }
}

STRF_TEST_FUNC void round_up_999()
{
    // When 999... is rounded up becaming 1000...

    // strf::sci
    TEST("1e-04")      (strf::sci(9.6e-5).p(0));
    TEST("1.000e-04")  (strf::sci(9.9996e-5).p(3));
    TEST("1.000e+06")  (strf::sci(9.9996e+5).p(3));
    TEST("1.000e+100") (strf::sci(9.9996e+99, 3));
    TEST("1.000e-99")  (strf::sci(9.9996e-100, 3));
    TEST("1e+100")     (strf::sci(9.5e+99, 0));
    TEST("1e-99")      (strf::sci(9.5e-100, 0));
    TEST("+1.e+100")   (+*strf::sci(9.5e+99, 0));
    TEST("+1.e-99")    (+*strf::sci(9.5e-100, 0));
    TEST("-1.e+100")   (*strf::sci(-9.5e+99, 0));
    TEST("-1.e-99")    (*strf::sci(-9.5e-100, 0));

    // strf::fixed
    TEST("9.9996")    ( strf::fixed(9.9996).p(4));
    TEST("10.000")    ( strf::fixed(9.9996).p(3));
    TEST("10")        ( strf::fixed(9.9996).p(0));
    TEST("10.")       (*strf::fixed(9.9996).p(0));

    TEST("0.009995")  ( strf::fixed(9.995e-03).p(6));
    TEST("0.01000")   ( strf::fixed(9.995e-03).p(5));

    TEST("0.9995")  ( strf::fixed(9.995e-01).p(4));
    TEST("1.000")   ( strf::fixed(9.995e-01).p(3));

    // strf::gen
    TEST("0.001")      ( strf::gen(9.9996e-4).p(2));
    TEST("0.0010")     (*strf::gen(9.9996e-4).p(2));
    TEST("0.0001000")  (*strf::gen(9.9996e-5).p(4));

    TEST("1e-05")      ( strf::gen(9.6e-6).p(0));
    TEST("0.0001")     ( strf::gen(9.6e-5).p(1));
    TEST("1e-05")      ( strf::gen(9.9996e-6).p(4));
    TEST("1.000e-05")  (*strf::gen(9.9996e-6).p(4));
    TEST("1.000e-99")  (*strf::gen(9.9996e-100).p(4));
    TEST("1.000e-100") (*strf::gen(9.9996e-101).p(4));
    TEST("1.000e+100") (*strf::gen(9.9996e+99).p(4));
    TEST("1.000e+101") (*strf::gen(9.9996e+100).p(4));

    TEST("1.000e+05") (*strf::gen(9.9996e+4).p(4));
    TEST("1.000e+04") (*strf::gen(9.9996e+3).p(4));
    TEST("1000.")     (*strf::gen(9.9996e+2).p(4));

    TEST("9.9996e+05") ( strf::gen(9.9996e+5).p(5));
    TEST("1e+05")      ( strf::gen(9.9996e+4).p(4));
    TEST("1e+04")      ( strf::gen(9.9996e+3).p(4));
    TEST("1000")       ( strf::gen(9.9996e+2).p(4));
    TEST("9.9996e+05") (*strf::gen(9.9996e+5).p(5));
    TEST("1.000e+05")  (*strf::gen(9.9996e+4).p(4));
    TEST("1.000e+04")  (*strf::gen(9.9996e+3).p(4));
    TEST("1000.")      (*strf::gen(9.9996e+2).p(4));

    TEST("99996")     ( strf::gen(99996.0).p(5));
    TEST("1e+05")     ( strf::gen(99996.0).p(4));

    TEST("10")        ( strf::gen(9.996).p(3));
    TEST("10.")       (*strf::gen(9.9996).p(2));
    TEST("10.0")      (*strf::gen(9.996).p(3));
    TEST("10")        ( strf::gen(9.9996).p(2));
    TEST("1e+01")     ( strf::gen(9.6).p(1));
    TEST("1.e+01")    (*strf::gen(9.6).p(1));
}

} // unnamed namespace

STRF_TEST_FUNC void test_input_float()
{
#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
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

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
#  pragma GCC diagnostic pop
#endif

    test_hexadecimal();
    test_several_values<float>();
    test_several_values<double>();
    test_punctuation();
}

REGISTER_STRF_TEST(test_input_float);

