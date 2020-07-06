//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <limits>
#include <vector>
#include <cstdlib>

template <typename FloatT, typename FPack>
void test_subnormal_values(const FPack& fp)
{
    constexpr auto j = strf::join_right(20, '_');
    constexpr auto quiet_nan = std::numeric_limits<FloatT>::quiet_NaN();
    constexpr auto signaling_nan = std::numeric_limits<FloatT>::signaling_NaN();
    constexpr auto infinity = std::numeric_limits<FloatT>::infinity();

    TEST("_________________nan").with(fp)  (j(quiet_nan));
    TEST("_________________nan").with(fp)  (j(signaling_nan));
    TEST("_________________inf").with(fp)  (j(infinity));
    TEST("________________-inf").with(fp)  (j(-infinity));

    TEST("________________-nan").with(fp)  (j(strf::fmt(-quiet_nan)));
    TEST("_________________nan").with(fp)  (j(strf::fmt(signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fmt(signaling_nan)));
    TEST("________________+inf").with(fp)  (j(+strf::fmt(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::fmt(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::fmt(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::fmt(-infinity)));

    TEST("________________-nan").with(fp)  (j(strf::fixed(-quiet_nan)));
    TEST("_________________nan").with(fp)  (j(strf::fixed(signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fixed(quiet_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fixed(signaling_nan)));
    TEST("_________________inf").with(fp)  (j(strf::fixed(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::fixed(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::fixed(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::fixed(-infinity)));

    TEST("________________-nan").with(fp)  (j(strf::sci(-quiet_nan)));
    TEST("_________________nan").with(fp)  (j(strf::sci(signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::sci(quiet_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::sci(signaling_nan)));
    TEST("_________________inf").with(fp)  (j(strf::sci(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::sci(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::sci(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::sci(-infinity)));

    TEST("_________________nan").with(fp)  (j(strf::hex(quiet_nan)));
    TEST("________________-nan").with(fp)  (j(strf::hex(-signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::hex(quiet_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::hex(signaling_nan)));
    TEST("_________________inf").with(fp)  (j(strf::hex(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::hex(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::hex(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::hex(-infinity)));

    TEST("___________~~~~~-nan").with(fp) (j(strf::right (-quiet_nan, 9, '~')));
    TEST("___________+nan~~~~~").with(fp) (j(+strf::left  (quiet_nan, 9, '~')));
    TEST("___________~~+nan~~~").with(fp) (j(+strf::center(quiet_nan, 9, '~')));
    TEST("___________+~~~~~nan").with(fp) (j(+strf::split (quiet_nan, 9, '~')));
    TEST("___________~~~~~+nan").with(fp) (j(+strf::right (quiet_nan, 9, '~')));
    TEST("___________+nan~~~~~").with(fp) (j(+strf::left  (quiet_nan, 9, '~')));
    TEST("___________~~+nan~~~").with(fp) (j(+strf::center(quiet_nan, 9, '~')));
    TEST("___________+~~~~~nan").with(fp) (j(+strf::split (quiet_nan, 9, '~')));

    TEST("___________~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~')));
    TEST("___________+inf~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~')));
    TEST("___________~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~')));
    TEST("___________+~~~~~inf").with(fp) (j(+strf::split (infinity, 9, '~')));
    TEST("___________~~~~~+inf").with(fp) (j(+strf::right (infinity, 9, '~').hex()));
    TEST("___________+inf~~~~~").with(fp) (j(+strf::left  (infinity, 9, '~').hex()));
    TEST("___________~~+inf~~~").with(fp) (j(+strf::center(infinity, 9, '~').hex()));
    TEST("___________+~~~~~inf").with(fp) (j(+strf::split (infinity, 9, '~').hex()));

    TEST("_________________nan").with(fp, strf::lowercase)  (j(quiet_nan));
    TEST("_________________inf").with(fp, strf::lowercase)  (j(infinity));
    TEST("________________-inf").with(fp, strf::lowercase)  (j(-infinity));
    TEST("_________________inf").with(fp, strf::lowercase)  (j(strf::sci(infinity)));
    TEST("________________-inf").with(fp, strf::lowercase)  (j(strf::sci(-infinity)));
    TEST("_________________inf").with(fp, strf::lowercase)  (j(strf::hex(infinity)));
    TEST("________________-inf").with(fp, strf::lowercase)  (j(strf::hex(-infinity)));
    TEST("_________________nan").with(fp, strf::lowercase)  (j(strf::hex(quiet_nan)));

    TEST("_________________NaN").with(fp, strf::mixedcase)  (j(quiet_nan));
    TEST("________________-NaN").with(fp, strf::mixedcase)  (j(-quiet_nan));
    TEST("_________________Inf").with(fp, strf::mixedcase)  (j(infinity));
    TEST("________________-Inf").with(fp, strf::mixedcase)  (j(-infinity));
    TEST("_________________Inf").with(fp, strf::mixedcase)  (j(strf::sci(infinity)));
    TEST("________________-Inf").with(fp, strf::mixedcase)  (j(strf::sci(-infinity)));
    TEST("_________________Inf").with(fp, strf::mixedcase)  (j(strf::hex(infinity)));
    TEST("________________-Inf").with(fp, strf::mixedcase)  (j(strf::hex(-infinity)));
    TEST("_________________NaN").with(fp, strf::mixedcase)  (j(strf::hex(quiet_nan)));

    TEST("_________________NAN").with(fp, strf::uppercase)  (j(quiet_nan));
    TEST("________________-NAN").with(fp, strf::uppercase)  (j(-quiet_nan));
    TEST("_________________INF").with(fp, strf::uppercase)  (j(infinity));
    TEST("________________-INF").with(fp, strf::uppercase)  (j(-infinity));
    TEST("_________________INF").with(fp, strf::uppercase)  (j(strf::sci(infinity)));
    TEST("________________-INF").with(fp, strf::uppercase)  (j(strf::sci(-infinity)));
    TEST("_________________INF").with(fp, strf::uppercase)  (j(strf::hex(infinity)));
    TEST("________________-INF").with(fp, strf::uppercase)  (j(strf::hex(-infinity)));
    TEST("_________________NAN").with(fp, strf::uppercase)  (j(strf::hex(quiet_nan)));
}

const char* replace_decimal_point
    ( const char* str
    , char32_t replacement )
{
    static char buff[200];
    auto it = std::strchr(str, '.');
    if (it == nullptr)
    {
        return str;
    }
    std::size_t pos = it - str;
    strf::detail::simple_string_view<char> str1{str, pos};
    strf::detail::simple_string_view<char> str2{str + pos + 1};
    char32_t replacement_str[2] = {replacement, U'\0'};
    strf::to(buff)(str1, strf::conv(replacement_str), str2);
    return buff;
}


template <typename FPack>
void basic_tests(const FPack& fp)
{
    constexpr auto j = strf::join_right(20, '_');

    auto punct = strf::get_facet<strf::numpunct_c<10>, double>(fp);
    auto decimal_point = punct.decimal_point();

    auto rd = [decimal_point](const char* str)
    {
        return replace_decimal_point(str, decimal_point);
    };

    TEST(rd("___________________0")).with(fp)  (j(0.0));
    TEST(rd("__________________-0")).with(fp)  (j(-0.0));
    TEST(rd("___________________1")).with(fp)  (j(1.0));
    TEST(rd("__________________-1")).with(fp)  (j(-1.0));
    TEST(rd("_________________1.5")).with(fp)  (j(1.5));
    TEST(rd("_____6.103515625e-05")).with(fp)  (j(6.103515625e-05));
    TEST(rd("____-6.103515625e-10")).with(fp)  (j(-6.103515625e-10));
    TEST(rd("____6.103515625e-100")).with(fp)  (j(6.103515625e-100));
    TEST(rd("_____6.103515625e-10")).with(fp)  (j(strf::sci(6.103515625e-10)));
    TEST(rd("____6.103515625e-100")).with(fp)  (j(strf::sci(6.103515625e-100)));
    TEST(rd("_______0.00048828125")).with(fp)  (j(0.00048828125));
    TEST(rd("______2048.001953125")).with(fp)  (j(2048.001953125));

    TEST(rd("___________________0")).with(fp) (j(strf::fmt(0.0)));
    TEST(rd("__________________-0")).with(fp) (j(strf::fmt(-0.0)));
    TEST(rd("___________________1")).with(fp) (j(strf::fmt(1.0)));
    TEST(rd("__________________-1")).with(fp) (j(strf::fmt(-1.0)));
    TEST(rd("_________________1.5")).with(fp) (j(strf::fmt(1.5)));
    TEST(rd("_________12345678.25")).with(fp) (j(strf::fmt(12345678.25)));

    TEST(rd("__________________+1")).with(fp) (j(+strf::fmt(1.0)));
    TEST(rd("__________________-1")).with(fp) (j(+strf::fmt(-1.0)));
    TEST(rd("__________________1.")).with(fp) (j(*strf::fmt(1.0)));
    TEST(rd("_________________-1.")).with(fp) (j(*strf::fmt(-1.0)));
    TEST(rd("_________________+1.")).with(fp)(j(*+strf::fmt(1.0)));
    TEST(rd("_________________-1.")).with(fp)(j(+*strf::fmt(-1.0)));

    TEST(rd("_____________+0.0001")).with(fp) (j(+strf::fixed(0.0001)));
    TEST(rd("_______+0.0001000000")).with(fp) (j(+strf::fixed(0.0001).p(10)));


    TEST(rd("_______________1e+50")).with(fp, strf::lowercase)  (j(1e+50));
    TEST(rd("_______________1e+50")).with(fp, strf::mixedcase)  (j(1e+50));
    TEST(rd("_______________1E+50")).with(fp, strf::uppercase)  (j(1e+50));

    TEST(rd("______________ 1e+50")).with(fp, strf::lowercase)  (j(strf::sci(1e+50)>6));
    TEST(rd("______________ 1e+50")).with(fp, strf::mixedcase)  (j(strf::sci(1e+50)>6));
    TEST(rd("______________ 1E+50")).with(fp, strf::uppercase)  (j(strf::sci(1e+50)>6));

    //----------------------------------------------------------------
    // when precision is not specified, the general format selects the
    // scientific notation if it is shorter than the fixed notation:
    TEST(rd("_______________10000")).with(fp)  (j(1e+4));
    TEST(rd("_______________1e+05")).with(fp)  (j(1e+5));
    TEST(rd("_____________1200000")).with(fp)  (j(1.2e+6));
    TEST(rd("_____________1.2e+07")).with(fp)  (j(1.2e+7));
    TEST(rd("_______________0.001")).with(fp)  (j(1e-03));
    TEST(rd("_______________1e-04")).with(fp)  (j(1e-04));
    TEST(rd("_____________0.00012")).with(fp)  (j(1.2e-04));
    TEST(rd("_____________1.2e-05")).with(fp)  (j(1.2e-05));
    TEST(rd("____________0.000123")).with(fp)  (j(1.23e-04));
    TEST(rd("____________1.23e-05")).with(fp)  (j(1.23e-05));
    TEST(rd("_______________10000")).with(fp)  (j(strf::fmt(1e+4)));
    TEST(rd("______________10000.")).with(fp) (j(*strf::fmt(1e+4)));
    TEST(rd("_______________1e+05")).with(fp)  (j(strf::fmt(1e+5)));
    TEST(rd("______________1.e+05")).with(fp) (j(*strf::fmt(1e+5)));
    TEST(rd("_____________1200000")).with(fp)  (j(strf::fmt(1.2e+6)));
    TEST(rd("_____________1.2e+06")).with(fp) (j(*strf::fmt(1.2e+6)));
    TEST(rd("_______________0.001")).with(fp)   (j(strf::fmt(1e-03)));
    TEST(rd("_______________1e-04")).with(fp)   (j(strf::fmt(1e-04)));
    TEST(rd("______________0.0001")).with(fp)  (j(*strf::fmt(1e-04)));
    TEST(rd("______________1.e-05")).with(fp)  (j(*strf::fmt(1e-05)));
    TEST(rd("_____________0.00012")).with(fp)   (j(strf::fmt(1.2e-04)));
    TEST(rd("_____________0.00012")).with(fp) (j(*strf::fmt(1.2e-04)));
    TEST(rd("_____________1.2e-05")).with(fp)  (j(strf::fmt(1.2e-05)));
    TEST(rd("_____________1.2e-05")).with(fp) (j(*strf::fmt(1.2e-05)));
    TEST(rd("____________0.000123")).with(fp)  (j(strf::fmt(1.23e-04)));
    TEST(rd("____________1.23e-05")).with(fp)  (j(strf::fmt(1.23e-05)));
    TEST(rd("_____6.103515625e-05")).with(fp)    (j(strf::fmt(6.103515625e-05)));
    TEST(rd("_______0.00048828125")).with(fp)    (j(strf::fmt(0.00048828125)));

    //----------------------------------------------------------------


    //----------------------------------------------------------------
    // when precision is specified in the general format,
    // do as in printf:
    // - The precision specifies the number of significant digits.
    // - scientific notation is used if the resulting exponent is
    //   less than -4 or greater than or equal to the precision.
    //TEST(rd("______________0.0001")).with(fp)  (j(strf::fmt(1e-4).p());
    //TEST(rd("_______________1e-05")).with(fp)  (j(strf::fmt(1e-5)));
    TEST(rd("_______________1e+01")).with(fp) (j(strf::fmt(12.0).p(1)));
    TEST(rd("_____________1.2e+02")).with(fp) (j(strf::fmt(123.0).p(2)));
    TEST(rd("_________________123")).with(fp) (j(strf::fmt(123.0).p(4)));
    TEST(rd("_______________1e+04")).with(fp) (j(strf::fmt(10000.0).p(4)));
    TEST(rd("_______________10000")).with(fp) (j(strf::fmt(10000.0).p(5)));
    TEST(rd("__________6.1035e-05")).with(fp) (j(strf::fmt(6.103515625e-05).p(5)));

    // and if precision is zero, it treated as 1.
    TEST(rd("_______________1e+01")).with(fp)  (j(strf::fmt(12.0).p(0)));
    TEST(rd("_______________2e+01")).with(fp)  (j(strf::fmt(15.125).p(0)));

    // and when removing digits, the last digit is rounded.
    TEST(rd("_____________1.1e+05")).with(fp) (j(strf::fmt(114999.0).p(2)));
    TEST(rd("_____________1.2e+05")).with(fp) (j(strf::fmt(115000.0).p(2)));
    TEST(rd("_____________1.2e+05")).with(fp) (j(strf::fmt(125000.0).p(2)));
    TEST(rd("_____________1.3e+05")).with(fp) (j(strf::fmt(125001.0).p(2)));

    // and the decimal point appears only if followed by
    // a digit, or if operator*() is used.
    TEST(rd("_______________1e+04")).with(fp)   (j(strf::fmt(10000.0).p(3)));
    TEST(rd("______________1.e+04")).with(fp)  (j(*strf::fmt(10000.0).p(1)));
    TEST(rd("________________123.")).with(fp)  (j(*strf::fmt(123.0).p(3)));

    // and trailing zeros are removed, unless operator*() is used.
    TEST(rd("_____________1.5e+04")).with(fp)   (j(strf::fmt(15000.0).p(3)));
    TEST(rd("____________1.50e+04")).with(fp)  (j(*strf::fmt(15000.0).p(3)));
    TEST(rd("_____________1.5e+04")).with(fp)  (j(strf::fmt(15001.0).p(3)));
    TEST(rd("_____________1.5e+04")).with(fp)  (j(*strf::fmt(15001.0).p(3)));
    TEST(rd("_________________123")).with(fp)   (j(strf::fmt(123.0).p(5)));
    TEST(rd("______________123.00")).with(fp)  (j(*strf::fmt(123.0).p(5)));
    //----------------------------------------------------------------

    // force fixed notation

    TEST(rd("__________________1.")).with(fp)  (j(*strf::fixed(1.0)));
    TEST(rd("___________________1")).with(fp)   (j(strf::fixed(1.0)));
    TEST(rd("__________________+1")).with(fp)  (j(+strf::fixed(1.0)));
    TEST(rd("__________________-1")).with(fp)   (j(strf::fixed(-1.0)));
    TEST(rd("__________________-1")).with(fp)  (j(+strf::fixed(-1.0)));

    TEST(rd("___________________1")).with(fp)  (j(strf::fixed(1.0).p(0)));
    TEST(rd("__________________1.")).with(fp) (j(*strf::fixed(1.0).p(0)));
    TEST(rd("_________________1.0")).with(fp)  (j(strf::fixed(1.0).p(1)));
    TEST(rd("________________1.00")).with(fp)  (j(strf::fixed(1.0).p(2)));
    TEST(rd("______________1.0000")).with(fp)  (j(strf::fixed(1.0).p(4)));
    TEST(rd("______________1.2500")).with(fp)  (j(strf::fixed(1.25).p(4)));
    TEST(rd("_______________0.125")).with(fp)  (j(strf::fixed(0.125)));

    TEST(rd("______________0.e+00")).with(fp)  (j(*strf::sci(0.0)));
    TEST(rd("______________1.e+04")).with(fp)  (j(*strf::sci(1e+4)));
    TEST(rd("_____________+1.e+04")).with(fp) (j(*+strf::sci(1e+4)));
    TEST(rd("_____________-1.e+04")).with(fp) (j(*strf::sci(-1e+4)));

    TEST(rd("_____________1.0e+04")).with(fp)   (j(strf::sci(1e+4).p(1)));
    TEST(rd("_____________1.0e+04")).with(fp)   (j(strf::sci(1e+4).p(1)));
    TEST(rd("___________+1.00e+04")).with(fp)  (j(+strf::sci(1e+4).p(2)));
    TEST(rd("______________1.e+04")).with(fp)  (j(*strf::sci(1e+4).p(0)));
    TEST(rd("_____________+1.e+04")).with(fp) (j(*+strf::sci(1e+4).p(0)));
    TEST(rd("______________-1e+04")).with(fp)  (j(+strf::sci(-1e+4).p(0)));
    TEST(rd("_____________-1.e+04")).with(fp)  (j(*strf::sci(-1e+4).p(0)));

    TEST(rd("____________1.25e+02")).with(fp) (j(strf::sci(125.0).p(2)));
    TEST(rd("_____________1.2e+02")).with(fp) (j(strf::sci(125.0).p(1)));
    TEST(rd("_____________1.2e+02")).with(fp) (j(strf::sci(115.0).p(1)));
    TEST(rd("_____________1.3e+06")).with(fp) (j(strf::sci(1250001.0).p(1)));
    TEST(rd("__________8.1928e+03")).with(fp) (j(strf::sci(8192.75).p(4)));
    TEST(rd("__________8.1922e+03")).with(fp) (j(strf::sci(8192.25).p(4)));
    TEST(rd("__________1.0242e+03")).with(fp) (j(strf::sci(1024.25).p(4)));
    TEST(rd("_____________1.7e+01")).with(fp) (j(strf::sci(16.50006103515625).p(1)));
    TEST(rd("___________1.250e+02")).with(fp) (j(strf::sci(125.0).p(3)));
    TEST(rd("_________6.25000e-02")).with(fp) (j(strf::sci(0.0625).p(5)));
    TEST(rd("________8.192750e+03")).with(fp) (j(strf::sci(8192.75).p(6)));

    TEST(rd("_________******-1.25")).with(fp) (j(strf::right(-1.25, 11, '*')));
    TEST(rd("_________-1.25******")).with(fp) (j(strf::left(-1.25, 11, '*')));
    TEST(rd("_________***-1.25***")).with(fp) (j(strf::center(-1.25, 11, '*')));
    TEST(rd("_________-******1.25")).with(fp) (j(strf::split(-1.25, 11, '*')));
    TEST(rd("_________+******1.25")).with(fp) (j(+strf::split(1.25, 11, '*')));

    TEST(rd("_______________-1.25")).with(fp) (j(strf::right(-1.25, 5, '*')));
    TEST(rd("_______________-1.25")).with(fp) (j(strf::left(-1.25, 5, '*')));
    TEST(rd("_______________-1.25")).with(fp) (j(strf::center(-1.25, 5, '*')));
    TEST(rd("_______________-1.25")).with(fp) (j(strf::split(-1.25, 5, '*')));
    TEST(rd("_______________+1.25")).with(fp) (j(+strf::split(1.25, 5, '*')));

    TEST(rd("_____________\xEF\xBF\xBD\xEF\xBF\xBD-1.25")).with(fp)
        (j(strf::right(-1.25, 7, 0xFFFFFFF)));
    TEST(rd("_____________-1.25\xEF\xBF\xBD\xEF\xBF\xBD")).with(fp)
        (j(strf::left(-1.25, 7, 0xFFFFFFF)));
    TEST(rd("_____________\xEF\xBF\xBD-1.25\xEF\xBF\xBD")).with(fp)
        (j(strf::center(-1.25, 7, 0xFFFFFFF)));
    TEST(rd("_____________-\xEF\xBF\xBD\xEF\xBF\xBD""1.25")).with(fp)
        (j(strf::split(-1.25, 7, 0xFFFFFFF)));
    TEST(rd("_____________+\xEF\xBF\xBD\xEF\xBF\xBD""1.25")).with(fp)
        (j(+strf::split(1.25, 7, 0xFFFFFFF)));
}


void test_hexadecimal()
{
    constexpr auto j = strf::join_right(20, '_');

    TEST("______________0x0p+0") (j(strf::hex(0.0)));
    TEST("_____________0x0.p+0") (j(*strf::hex(0.0)));
    TEST("__________0x0.000p+0") (j(strf::hex(0.0).p(3)));
    TEST("_____________-0x1p-3") (j(strf::hex(-0.125)));
    TEST("_____________0x1p+11") (j(strf::hex(2048.0)));
    TEST("__0x1.fffffffffffffp+1023")
        ( strf::join_right(25, '_') (strf::hex((std::numeric_limits<double>::max)())));
    TEST("___________0x1p-1022") (j(strf::hex(0x1p-1022)));
    TEST("__________0x1.p-1022") (j(*strf::hex(0x1p-1022)));
    TEST("_______0x1.000p-1022") (j(*strf::hex(0x1p-1022).p(3)));
    TEST("______0x0.0009p-1022") (j(strf::hex(0x0.0009p-1022)));

    TEST("______________0x1p+0") (j(strf::hex(0x1.12345p+0).p(0)));
    TEST("________0x1.12345p+0") (j(strf::hex(0x1.12345p+0).p(5)));
    TEST("_______0x1.123450p+0") (j(strf::hex(0x1.12345p+0).p(6)));
    TEST("_____________0x1.p+0") (j(*strf::hex(0x1.12345p+0).p(0)));
    TEST("_______0x0.000p-1022") (j(strf::hex(0x0.0008p-1022).p(3)));
    TEST("_______0x0.002p-1022") (j(strf::hex(0x0.0018p-1022).p(3)));
    TEST("_______0x0.001p-1022") (j(strf::hex(0x0.0008000000001p-1022).p(3)));

    TEST("______________0X0P+0").with(strf::uppercase) (j(strf::hex(0.0)));
    TEST("____0X0.ABCDEFP-1022").with(strf::uppercase) (j(strf::hex(0x0.abcdefp-1022)));
    TEST("______0X1.ABCDEFP+10").with(strf::uppercase) (j(strf::hex(0x1.abcdefp+10)));

    TEST("______________0x0p+0").with(strf::mixedcase) (j(strf::hex(0.0)));
    TEST("____0x0.ABCDEFp-1022").with(strf::mixedcase) (j(strf::hex(0x0.abcdefp-1022)));
    TEST("______0x1.ABCDEFp+10").with(strf::mixedcase) (j(strf::hex(0x1.abcdefp+10)));

    TEST("______________0x0p+0").with(strf::lowercase) (j(strf::hex(0.0)));
    TEST("____0x0.abcdefp-1022").with(strf::lowercase) (j(strf::hex(0x0.abcdefp-1022)));
    TEST("______0x1.abcdefp+10").with(strf::lowercase) (j(strf::hex(0x1.abcdefp+10)));

    TEST("_________-0x1p+0****") (j(strf::left(-1.0, 11, '*').hex()));
    TEST("_________****-0x1p+0") (j(strf::right(-1.0, 11, '*').hex()));
    TEST("_________-****0x1p+0") (j(strf::split(-1.0, 11, '*').hex()));
    TEST("_________**-0x1p+0**") (j(strf::center(-1.0, 11, '*').hex()));
    TEST("_____________-0x1p+0") (j(strf::center(-1.0, 7, '*').hex()));
}

double make_double(std::uint64_t ieee_exponent, std::uint64_t ieee_mantissa)
{
    std::uint64_t v = (ieee_exponent << 52) | (ieee_mantissa & 0xFFFFFFFFFFFFFull);
    double d;
    std::memcpy(&d, &v, 8);
    return d;
}
float make_float(std::uint32_t ieee_exponent, std::uint32_t ieee_mantissa)
{
    std::uint32_t v = (ieee_exponent << 23) | (ieee_mantissa & 0x7FFFFF);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

std::vector<double> generate_double_samples()
{
    std::vector<double> samples;
    constexpr int ieee_exp_size = 11;
    constexpr int ieee_m_size = 52;
    constexpr unsigned max_normal_exp = (1 << ieee_exp_size) - 2;

    for(unsigned e = 0; e <= max_normal_exp; ++e) {
        for(unsigned i = 2; i <= ieee_m_size; ++i) {
            unsigned s = ieee_m_size - i;
            samples.push_back(make_double(e, 0xFFFFFFFFFFFFFull << s));
            samples.push_back(make_double(e, 1ull << s));
        }
        samples.push_back(make_double(e, 1ull << (ieee_m_size - 1)));
        samples.push_back(make_double(e, 0));
    }
    return samples;
}

std::vector<float> generate_float_samples()
{
    std::vector<float> samples;
    constexpr int ieee_exp_size = 8;
    constexpr int ieee_m_size = 23;
    constexpr unsigned max_normal_exp = (1 << ieee_exp_size) - 2;

    for(unsigned e = 0; e < max_normal_exp; ++e) {
        for(unsigned i = 2; i <= ieee_m_size; ++i) {
            unsigned s = ieee_m_size - i;
            samples.push_back(make_float(e, 0x7FFFFFul << s));
            samples.push_back(make_float(e, 1 << s));
        }
        samples.push_back(make_float(e, 1 << (ieee_m_size - 1)));
        samples.push_back(make_float(e, 0));
    }
    return samples;
}

void test_input_float()
{
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

    {
        TEST_SCOPE_DESCRIPTION("default facets");
        basic_tests(strf::pack());
    }
    {
        TEST_SCOPE_DESCRIPTION("with small decimal point");
        basic_tests(strf::pack( strf::numpunct<10>(-1).decimal_point('*')
                              , strf::numpunct<16>(-1).decimal_point('*') ));
    }

    test_hexadecimal();

    {
        auto vec = generate_double_samples();
        char buff[64];
        for (const auto d: vec)
        {
            (void) strf::to(buff) (d);
            auto parsed = std::strtod(buff, nullptr);
            TEST_EQ(parsed, d);
        }
    }

    {
        auto vec = generate_float_samples();
        char buff[64];
        for (const float f: vec)
        {
            (void) strf::to(buff) (f);
            auto parsed = std::strtof(buff, nullptr);
            TEST_EQ(parsed, f);
        }
    }

    // ================================================================================
    // With punctuation
    // ================================================================================

    constexpr auto j = strf::join_right(20, '_');

    {   //alternative punct characters
        auto p = strf::numpunct<10>{3}.thousands_sep(':').decimal_point(',');
        TEST("_____________1:000,5").with(p) (j(1000.5));
        TEST("_____________1,5e+50").with(p) (j(1.5e+50));
        TEST("____________1:000,50").with(p) (j(strf::fixed(1000.5).p(2)));
        TEST("____________1,50e+50").with(p) (j(strf::sci(1.5e+50).p(2)));
        TEST("__________________1,").with(p) (j(*strf::fmt(1.0)));
        TEST("_________+1:000:000,").with(p) (j(*+strf::fixed(1000000.0)));
        TEST("_____+1:000:000,0000").with(p) (j(*+strf::fixed(1000000.0).p(4)));

        auto px = strf::numpunct<16>{3}.decimal_point(',');
        TEST("________0x1,12345p+0").with(px) (j(strf::hex(0x1.12345p+0)));
    }
    {   //encoding big punct characters
        auto p = strf::numpunct<10>{3}.thousands_sep(0x10AAAA).decimal_point(0x10FFFF);

        TEST(u8"_____________1\U0010AAAA" u8"000\U0010FFFF" u8"5").with(p) (j(1000.5));
        TEST(u8"_____________1\U0010FFFF" u8"5e+50")              .with(p) (j(1.5e+50));
        TEST(u8"____________1\U0010AAAA" u8"000\U0010FFFF" u8"50").with(p)
            (j(strf::fixed(1000.5).p(2)));
        TEST(u8"____________1\U0010FFFF" u8"50e+50") .with(p) (j(strf::sci(1.5e+50).p(2)));
        TEST(u8"__________________1\U0010FFFF")      .with(p) (j(*strf::fmt(1.0)));
        TEST(u8"_________________0\U0010FFFF" u8"1") .with(p) (j(strf::fixed(0.1)));
        TEST(u8"_________________0\U0010FFFF" u8"1") .with(p) (j(0.1));

        //in hexadecimal
        auto px = strf::numpunct<16>{3}.decimal_point(0x10FFFF);
        TEST(u8"________0x1\U0010FFFF" u8"12345p+0").with(px)
            (j(strf::hex(0x1.12345p+0)));
    }
    {   // encoding punct chars in a single-byte encoding
        auto fp = strf::pack
            ( strf::iso_8859_3<char>()
            , strf::numpunct<10>(3).thousands_sep(0x2D9).decimal_point(0x130) );

        TEST("_____________1\xFF""000\xA9""5").with(fp) (j(1000.5));
        TEST("_____________1\xA9""5e+50")     .with(fp) (j(1.5e+50));
        TEST("____________1\xFF""000\xA9""50").with(fp) (j(strf::fixed(1000.5).p(2)));
        TEST("____________1\xA9""50e+50")     .with(fp) (j(strf::sci(1.5e+50).p(2)));
        TEST("__________________1\xA9")       .with(fp) (j(*strf::fmt(1.0)));
        TEST("_________________0\xA9" "1")    .with(fp) (j(strf::fixed(0.1)));
        TEST("_________________0\xA9" "1")    .with(fp) (j(0.1));

        auto fpx = strf::pack
            ( strf::iso_8859_3<char>(), strf::numpunct<16>(4).decimal_point(0x130) );

        TEST("________0x1\xA9""12345p+0").with(fpx) (j(strf::hex(0x1.12345p+0)));
    }
    {   // invalid punct characters
        // ( thousand separators are omitted  )

        auto p = strf::numpunct<10>{3}.thousands_sep(0xFFFFFF).decimal_point(0xEEEEEE);
        TEST(u8"______________1000\uFFFD5")       .with(p) (j(1000.5));
        TEST(u8"_____________1\uFFFD" u8"5e+50")  .with(p) (j(1.5e+50));
        TEST(u8"_____________1000\uFFFD" u8"50")  .with(p) (j(strf::fixed(1000.5).p(2)));
        TEST(u8"____________1\uFFFD" u8"50e+50")  .with(p) (j(strf::sci(1.5e+50).p(2)));
        TEST(u8"__________________1\uFFFD")       .with(p) (j(*strf::fmt(1.0)));
        TEST(u8"_________________0\uFFFD" u8"1")  .with(p) (j(strf::fixed(0.1)));
        TEST(u8"_________________0\uFFFD" u8"1")  .with(p) (j(0.1));

        auto px = strf::numpunct<16>{3}.decimal_point(0xEEEEEE);
        TEST(u8"________0x1\uFFFD" u8"12345p+0").with(px) (j(strf::hex(0x1.12345p+0)));
    }
    {   // invalid punct characters  in a single-byte encoding
        // ( thousand separators are omitted  )
        auto fp = strf::pack
            ( strf::iso_8859_3<char>()
            , strf::numpunct<10>(3).thousands_sep(0xFFF).decimal_point(0xEEE) );

        TEST("______________1000?5").with(fp) (j(1000.5));
        TEST("_____________1?5e+50").with(fp) (j(1.5e+50));
        TEST("_____________1000?50").with(fp) (j(strf::fixed(1000.5).p(2)));
        TEST("____________1?50e+50").with(fp) (j(strf::sci(1.5e+50).p(2)));
        TEST("__________________1?").with(fp) (j(*strf::fmt(1.0)));
        TEST("_________________0?1").with(fp) (j(strf::fixed(0.1)));
        TEST("_________________0?1").with(fp) (j(0.1));

        auto fpx = strf::pack
            ( strf::iso_8859_3<char>(), strf::numpunct<16>(4).decimal_point(0xEEE) );

        TEST("________0x1?12345p+0").with(fpx) (j(strf::hex(0x1.12345p+0)));
    }

    {   // When the integral part does not have trailing zeros

        TEST("1:048:576") .with(strf::numpunct<10>(3).thousands_sep(':'))
            (1048576.0);

        TEST("1:048:576") .with(strf::numpunct<10>(3).thousands_sep(':'))
            (strf::fixed(1048576.0));

        TEST(u8"1\u0ABC" u8"048\u0ABC" u8"576")
            .with(strf::numpunct<10>(3).thousands_sep(0xABC))
            (1048576.0);

        TEST(u8"1\u0ABC" u8"048\u0ABC" u8"576")
            .with(strf::numpunct<10>(3).thousands_sep(0xABC))
            (strf::fixed(1048576.0));
    }
    {   // variable groups
        auto grp = [](auto... grps) { return strf::numpunct<10>{grps...}; };

        TEST("1,00,00,0,00,0")  .with(grp(1,2,1,2)) (strf::fixed(100000000.0));
        TEST("10,00,00,0,00,0") .with(grp(1,2,1,2)) (strf::fixed(1000000000.0));

        TEST("32,10,00,0,00,0") .with(grp(1,2,1,2)) (strf::fixed(3210000000.0));
        TEST("43,21,00,0,00,0") .with(grp(1,2,1,2)) (strf::fixed(4321000000.0));
        TEST("54,32,10,0,00,0") .with(grp(1,2,1,2)) (strf::fixed(5432100000.0));
        TEST("7,65,432,10,0")   .with(grp(1,2,3,2)) (strf::fixed(765432100.0));
        TEST("7,6,543,21,0,0")  .with(grp(1,1,2,3,1)) (strf::fixed(765432100.0));
        TEST("7,654,321,00,0")  .with(grp(1,2,3)) (strf::fixed(7654321000.0));

        TEST("1000,00,0") .with(grp(1,2,-1)) (strf::fixed(1000000.0));
        TEST("1234,00,0") .with(grp(1,2,-1)) (strf::fixed(1234000.0));
        TEST("1234,50,0") .with(grp(1,2,-1)) (strf::fixed(1234500.0));
    }
    {   // variable groups and big separator
        auto grp = [](auto... grps)
            { return strf::numpunct<10>{grps...}.thousands_sep(0xABCD); };

        TEST(u8"1\uABCD" u8"00\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (strf::fixed(100000000.0));
        TEST(u8"10\uABCD" u8"00\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (strf::fixed(1000000000.0));

        TEST(u8"32\uABCD" u8"10\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (strf::fixed(3210000000.0));
        TEST(u8"43\uABCD" u8"21\uABCD" u8"00\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (strf::fixed(4321000000.0));
        TEST(u8"54\uABCD" u8"32\uABCD" u8"10\uABCD" u8"0\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,1,2)) (strf::fixed(5432100000.0));
        TEST(u8"7\uABCD" u8"65\uABCD" u8"432\uABCD" u8"10\uABCD" u8"0")
              .with(grp(1,2,3,2)) (strf::fixed(765432100.0));
        TEST(u8"7\uABCD" u8"6\uABCD" u8"543\uABCD" u8"21\uABCD" u8"0\uABCD" u8"0")
              .with(grp(1,1,2,3,1)) (strf::fixed(765432100.0));
        TEST(u8"7\uABCD" u8"654\uABCD" u8"321\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,3)) (strf::fixed(7654321000.0));

        TEST(u8"1000\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,-1)) (strf::fixed(1000000.0));
        TEST(u8"1234\uABCD" u8"00\uABCD" u8"0")
              .with(grp(1,2,-1)) (strf::fixed(1234000.0));
        TEST(u8"1234\uABCD" u8"50\uABCD" u8"0")
              .with(grp(1,2,-1)) (strf::fixed(1234500.0));
    }
    {   // when precision is not specified, the general format selects the
        // scientific notation if it is shorter than the fixed notation:
        auto p1 = strf::numpunct<10>{1};

        TEST("_______________1,0,0").with(p1)  (j(1e+2));
        TEST("_______________1e+03").with(p1)  (j(1e+3));
        TEST("_____________1,2,0,0").with(p1)  (j(1.2e+3));
        TEST("_____________1.2e+04").with(p1)  (j(1.2e+4));
        TEST("_______________1,0,0").with(p1)  (j(strf::fmt(1e+2)));
        TEST("______________1,0,0.").with(p1) (j(*strf::fmt(1e+2)));
        TEST("_______________1e+03").with(p1)  (j(strf::fmt(1e+3)));
        TEST("______________1.e+03").with(p1) (j(*strf::fmt(1e+3)));
        TEST("_____________1,2,0,0").with(p1)  (j(strf::fmt(1.2e+3)));
        TEST("_____________1.2e+03").with(p1) (j(*strf::fmt(1.2e+3)));
        TEST("_______________1e-04").with(p1)  (j(1e-4));
        TEST("_______________0.001").with(p1)  (j(1e-3));
        TEST("_______________1e-04").with(p1)  (j(strf::fmt(1e-4)));
        TEST("_______________0.001").with(p1)  (j(strf::fmt(1e-3)));
        TEST("_______________1e+05").with(strf::numpunct<10>(8))(j(strf::fmt(1e+5)));
        TEST("________1.000005e+05").with(p1) (j(100000.5));
        TEST("_________1,0,0,0,0.5").with(p1) (j(10000.5));
    }
}

