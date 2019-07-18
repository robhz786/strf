//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <limits>
#include "test_utils.hpp"

namespace strf = boost::stringify::v0;

int main()
{
    TEST("nan")    (std::numeric_limits<double>::quiet_NaN());
    TEST("nan")    (std::numeric_limits<double>::signaling_NaN());
    TEST("inf")    (1.0/0.0);
    TEST("-inf")   (-1.0/0.0);

    TEST("nan")    (strf::fmt(std::numeric_limits<double>::quiet_NaN()));
    TEST("nan")    (strf::fmt(std::numeric_limits<double>::signaling_NaN()));
    TEST("inf")    (strf::fmt(1.0/0.0));
    TEST("+inf")  (+strf::fmt(1.0/0.0));
    TEST("-inf")   (strf::fmt(-1.0/0.0));
    TEST("-inf")  (+strf::fmt(-1.0/0.0));

    TEST("nan")    (strf::fixed(std::numeric_limits<double>::quiet_NaN()));
    TEST("nan")    (strf::fixed(std::numeric_limits<double>::signaling_NaN()));
    TEST("inf")    (strf::fixed(1.0/0.0));
    TEST("+inf")  (+strf::fixed(1.0/0.0));
    TEST("-inf")   (strf::fixed(-1.0/0.0));
    TEST("-inf")  (+strf::fixed(-1.0/0.0));

    TEST("nan")    (strf::sci(std::numeric_limits<double>::quiet_NaN()));
    TEST("nan")    (strf::sci(std::numeric_limits<double>::signaling_NaN()));
    TEST("inf")    (strf::sci(1.0/0.0));
    TEST("+inf")  (+strf::sci(1.0/0.0));
    TEST("-inf")   (strf::sci(-1.0/0.0));
    TEST("-inf")  (+strf::sci(-1.0/0.0));

    TEST("0")      (0.0);
    TEST("-0")    (-0.0);
    TEST("1")      (1.0);
    TEST("-1")    (-1.0);
    TEST("1.5")    (1.5);
    TEST("6.103515625e-05") (6.103515625e-05);
    TEST("0.00048828125")   (0.00048828125);
    TEST("2048.001953125")  (2048.001953125);

    auto punct = strf::monotonic_grouping<10>{3}.decimal_point('~').thousands_sep('^');
    TEST("0").facets(punct)      (0.0);
    TEST("-0").facets(punct)    (-0.0);
    TEST("1").facets(punct)      (1.0);
    TEST("-1").facets(punct)    (-1.0);
    TEST("1~5").facets(punct)    (1.5);
    TEST("6~103515625e-05").facets(punct) (6.103515625e-05);
    TEST("0~00048828125").facets(punct)   (0.00048828125);
    TEST("2^048~001953125").facets(punct) (2048.001953125);

    TEST("0")      (strf::fmt(0.0));
    TEST("-0")     (strf::fmt(-0.0));
    TEST("1")      (strf::fmt(1.0));
    TEST("-1")     (strf::fmt(-1.0));
    TEST("1.5")    (strf::fmt(1.5));

    TEST("+1")    (+strf::fmt(1.0));
    TEST("-1")    (+strf::fmt(-1.0));
    TEST("1.")    (~strf::fmt(1.0));
    TEST("-1.")   (~strf::fmt(-1.0));
    TEST("+1.")  (~+strf::fmt(1.0));
    TEST("-1.")  (+~strf::fmt(-1.0));

    //----------------------------------------------------------------
    // when precision is not specified, the general format selects the
    // scientific notation if it is shorter than the decimal notation:
    TEST("10000")     (1e+4);
    TEST("1e+05")     (1e+5);
    TEST("1200000" )  (1.2e+6);
    //TEST("1.2e+06" )  (1.2e+6);
    //TEST("0.001")    (1e-03);
    //TEST("1e-04")    (1e-04);
    //TEST("0.00012")  (1.2e-04);
    //TEST("1.2e-05")  (1.2e-05);
    //TEST("0.000123") (1.23e-04);
    //TEST("1.23e-05") (1.23e-05);
    TEST("10000")     (strf::fmt(1e+4));
    TEST("10000.")   (~strf::fmt(1e+4));
    TEST("1e+05")     (strf::fmt(1e+5));
    TEST("1.e+05")   (~strf::fmt(1e+5));
    TEST("1200000" )  (strf::fmt(1.2e+6));
    TEST("1.2e+06")  (~strf::fmt(1.2e+6));
    //TEST("0.001")     (strf::fmt(1e-03));
    //TEST("1e-04")     (strf::fmt(1e-04));
    //TEST("0.0001")   (~strf::fmt(1e-04));
    //TEST("1.e-05")   (~strf::fmt(1e-05));
    //TEST("0.00012")  (strf::fmt(1.2e-04));
    //TEST("0.00012") (~strf::fmt(1.2e-04));
    //TEST("1.2e-05")  (strf::fmt(1.2e-05));
    //TEST("1.2e-05") (~strf::fmt(1.2e-05));
    //TEST("0.000123") (strf::fmt(1.23e-04));
    //TEST("1.23e-05") (strf::fmt(1.23e-05));
    TEST("6.103515625e-05") (strf::fmt(6.103515625e-05));
    TEST("0.00048828125") (strf::fmt(0.00048828125));
    //----------------------------------------------------------------


    //----------------------------------------------------------------
    // when precision is specified in the general format,
    // do as in printf:
    // - The precision specifies the number of significant digits.
    // - scientific notation is used if the resulting exponent is
    //   less than -4 or greater than or equal to the precision.
    //TEST("0.0001")    (strf::fmt(1e-4).p();
    //TEST("1e-05")     (strf::fmt(1e-5));
    TEST("1e+01")   (strf::fmt(12.0).p(1));
    TEST("1.2e+02")   (strf::fmt(123.0).p(2));
    TEST("123")       (strf::fmt(123.0).p(4));
    TEST("1e+04")     (strf::fmt(10000.0).p(4));
    TEST("10000")     (strf::fmt(10000.0).p(5));

    // and if precision is zero, it treated as 1.
    TEST("1e+01")   (strf::fmt(12.0).p(0));
    TEST("2e+01")   (strf::fmt(15.125).p(0));

    // and when removing digits, the last digit is rounded.
    TEST("1.1e+05") (strf::fmt(114999.0).p(2));
    TEST("1.2e+05") (strf::fmt(115000.0).p(2));
    TEST("1.2e+05") (strf::fmt(125000.0).p(2));
    TEST("1.3e+05") (strf::fmt(125001.0).p(2));

    // and the decimal point appears only if followed by
    // a digit, or if operator~() is used.
    TEST("1e+04")     (strf::fmt(10000.0).p(3));
    TEST("1.e+04")   (~strf::fmt(10000.0).p(1));
    TEST("123.")      (~strf::fmt(123.0).p(3));

    // and trailing zeros are removed, unless operator~() is used.
    TEST("1.5e+04")    (strf::fmt(15000.0).p(3));
    TEST("1.50e+04")  (~strf::fmt(15000.0).p(3));
    TEST("123")       (strf::fmt(123.0).p(5));
    TEST("123.00")    (~strf::fmt(123.0).p(5));
    //----------------------------------------------------------------

/*
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
    TEST("")    (strf::fmt());
*/
    // force decimal notation

    TEST("1.")    (~strf::fixed(1.0));
    TEST("1")      (strf::fixed(1.0));
    TEST("+1")    (+strf::fixed(1.0));
    TEST("-1")     (strf::fixed(-1.0));
    TEST("-1")    (+strf::fixed(-1.0));

    TEST("1")      (strf::fixed(1.0).p(0));
    TEST("1.")    (~strf::fixed(1.0).p(0));
    TEST("1.0")    (strf::fixed(1.0).p(1));
    TEST("1.00")   (strf::fixed(1.0).p(2));
    TEST("1.0000") (strf::fixed(1.0).p(4));
    TEST("0.125")  (strf::fixed(0.125));

    // round when forcing fixed notation
    TEST("1.250")  (strf::fixed(1.25).p(3));
    TEST("1.25")   (strf::fixed(1.25).p(2));
    TEST("1.2")    (strf::fixed(1.25).p(1));
    TEST("1.8")    (strf::fixed(1.75).p(1));
    TEST("1.3")    (strf::fixed(1.25048828125).p(1));
    TEST("1.2505") (strf::fixed(1.25048828125).p(4));


    // force scientific notation

    TEST("0e+00")     (strf::sci(0.0));
    TEST("0.e+00")   (~strf::sci(0.0));
    TEST("+0e+00")   (+strf::sci(0.0));
    TEST("+0.e+00") (+~strf::sci(0.0));

    TEST("1e+04")     (strf::sci(1e+4));
    TEST("+1e+04")   (+strf::sci(1e+4));
    TEST("1.e+04")   (~strf::sci(1e+4));
    TEST("+1.e+04") (~+strf::sci(1e+4));
    TEST("-1e+04")    (strf::sci(-1e+4));
    TEST("-1e+04")   (+strf::sci(-1e+4));
    TEST("-1.e+04")  (~strf::sci(-1e+4));

    TEST("1.0e+04")    (strf::sci(1e+4).p(1));
    TEST("1.0e+04")    (strf::sci(1e+4).p(1));
    TEST("+1.00e+04") (+strf::sci(1e+4).p(2));
    TEST("1.e+04")    (~strf::sci(1e+4).p(0));
    TEST("+1.e+04")  (~+strf::sci(1e+4).p(0));
    TEST("-1e+04")    (+strf::sci(-1e+4).p(0));
    TEST("-1.e+04")   (~strf::sci(-1e+4).p(0));

    // rounding when forcing scientific notation
    TEST("1.25e+02")   (strf::sci(125.0).p(2));
    TEST("1.2e+02")    (strf::sci(125.0).p(1));
    TEST("1.2e+02")    (strf::sci(115.0).p(1));
    TEST("1.3e+06")    (strf::sci(1250001.0).p(1));
    TEST("8.1928e+03") (strf::sci(8192.75).p(4));
    TEST("8.1922e+03") (strf::sci(8192.25).p(4));
    TEST("1.0242e+03") (strf::sci(1024.25).p(4));
    TEST("1.7e+01")    (strf::sci(16.50006103515625).p(1));

    // add trailing zeros if precision requires
    TEST("1.250e+02")    (strf::sci(125.0).p(3));
    TEST("6.25000e-02")  (strf::sci(0.0625).p(5));
    TEST("8.192750e+03") (strf::sci(8192.75).p(6));





    // precision:
    // if precision == (unsigned)-1, which is the default, then
    // prints how many digits are necessary for a parser
    // to fully recover the exact value

    // otherwise when on fixed or scientific notation
    // precision is the number of digits after the radix point

    // on general notation, precision is the number of significant
    // digits. If 0, do like when precision == (unsigned)-1

    //

/*
              when e10 <= -_m10_digcount:  _sci_notation = (_m10_digcount + 2 + (showpoint || _m10_digcount > 1) < -e10)
                1 e-1
                  0.1      : 2 - e10
                  1e-01    : _m10_digcount + 4 + (showpoint || _m10_digcount > 1) + (e10 > 99)
                1 e-3
                  0.001
                  1e-03

                1 e-4
                  0.0001
                  1e-04
                  1.e-04
                1 e-5
                  0.00001  : 2 - e10
                  1e-05    : _m10_digcount + 4 + (showpoint || _m10_digcount > 1) + (e10 > 99)
                  1.e-05

                22 e-2
                  0.22
                  2.2e-01
                22 e-5
                  0.00022  : 2 - e10
                  2.2e-04  : _m10_digcount + 4 + (showpoint || _m10_digcount > 1) + (-e10 > 99)
                333e-5
                  0.00333  : 2 - e10
                  3.33e-04 : _m10_digcount + 4 + (showpoint || _m10_digcount > 1) + (-e10 > 99)


              when e10 >= 0:   _sci_notation = e10 > 4 + (!showpoint && _m10_digcount > 1)
                fixed : _m10_digcount + showpoint + e10;
                sci   : _m10_digcount + 4 + (showpoint || _m10_digcount > 1) + (e10 > 99)

              when -_m10_digcount < e10 < 0  : (_m10_digcount must be greater than 1 )
                22 e-1         _sci_notation = false
                  2.2      : _m10_digcount + 1
                  2.2e-01  : _m10_digcount + 5 + (e10 > 99)
*/
    return boost::report_errors();
}

