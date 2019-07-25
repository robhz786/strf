//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <limits>
#include "test_utils.hpp"

namespace strf = boost::stringify::v0;


template <typename FPack>
void basic_tests(const FPack& fp)
{
    constexpr auto j = strf::join_right(20, '_');
    auto quiet_nan = std::numeric_limits<double>::quiet_NaN();
    auto signaling_nan = std::numeric_limits<double>::signaling_NaN();

    TEST("_________________nan").facets(fp)  (j(quiet_nan));
    TEST("_________________nan").facets(fp)  (j(signaling_nan));
    TEST("_________________inf").facets(fp)  (j(1.0/0.0));
    TEST("________________-inf").facets(fp)  (j(-1.0/0.0));

    TEST("_________________nan").facets(fp)  (j(strf::fmt(quiet_nan)));
    TEST("_________________nan").facets(fp)  (j(strf::fmt(signaling_nan)));
    TEST("_________________inf").facets(fp)  (j(strf::fmt(1.0/0.0)));
    TEST("________________+inf").facets(fp) (j(+strf::fmt(1.0/0.0)));
    TEST("________________-inf").facets(fp)  (j(strf::fmt(-1.0/0.0)));
    TEST("________________-inf").facets(fp) (j(+strf::fmt(-1.0/0.0)));

    TEST("_________________nan").facets(fp)  (j(strf::fixed(quiet_nan)));
    TEST("_________________nan").facets(fp)  (j(strf::fixed(signaling_nan)));
    TEST("_________________inf").facets(fp)  (j(strf::fixed(1.0/0.0)));
    TEST("________________+inf").facets(fp) (j(+strf::fixed(1.0/0.0)));
    TEST("________________-inf").facets(fp)  (j(strf::fixed(-1.0/0.0)));
    TEST("________________-inf").facets(fp) (j(+strf::fixed(-1.0/0.0)));

    TEST("_________________nan").facets(fp)  (j(strf::sci(quiet_nan)));
    TEST("_________________nan").facets(fp)  (j(strf::sci(signaling_nan)));
    TEST("_________________inf").facets(fp)  (j(strf::sci(1.0/0.0)));
    TEST("________________+inf").facets(fp) (j(+strf::sci(1.0/0.0)));
    TEST("________________-inf").facets(fp)  (j(strf::sci(-1.0/0.0)));
    TEST("________________-inf").facets(fp) (j(+strf::sci(-1.0/0.0)));

    TEST("___________________0").facets(fp)  (j(0.0));
    TEST("__________________-0").facets(fp)  (j(-0.0));
    TEST("___________________1").facets(fp)  (j(1.0));
    TEST("__________________-1").facets(fp)  (j(-1.0));
    TEST("_________________1.5").facets(fp)  (j(1.5));
    TEST("_____6.103515625e-05").facets(fp)  (j(6.103515625e-05));
    TEST("_______0.00048828125").facets(fp)  (j(0.00048828125));
    TEST("______2048.001953125").facets(fp)  (j(2048.001953125));


    TEST("___________________0").facets(fp) (j(strf::fmt(0.0)));
    TEST("__________________-0").facets(fp) (j(strf::fmt(-0.0)));
    TEST("___________________1").facets(fp) (j(strf::fmt(1.0)));
    TEST("__________________-1").facets(fp) (j(strf::fmt(-1.0)));
    TEST("_________________1.5").facets(fp) (j(strf::fmt(1.5)));

    TEST("__________________+1").facets(fp) (j(+strf::fmt(1.0)));
    TEST("__________________-1").facets(fp) (j(+strf::fmt(-1.0)));
    TEST("__________________1.").facets(fp) (j(~strf::fmt(1.0)));
    TEST("_________________-1.").facets(fp) (j(~strf::fmt(-1.0)));
    TEST("_________________+1.").facets(fp)(j(~+strf::fmt(1.0)));
    TEST("_________________-1.").facets(fp)(j(+~strf::fmt(-1.0)));

    //----------------------------------------------------------------
    // when precision is not specified, the general format selects the
    // scientific notation if it is shorter than the decimal notation:
    TEST("_______________10000").facets(fp)  (j(1e+4));
    TEST("_______________1e+05").facets(fp)  (j(1e+5));
    TEST("_____________1200000").facets(fp)  (j(1.2e+6));
    //TEST("_____________1.2e+06").facets(fp)  (j(1.2e+6));
    //TEST("_______________0.001").facets(fp)  (j(1e-03));
    //TEST("_______________1e-04").facets(fp)  (j(1e-04));
    //TEST("_____________0.00012").facets(fp)  (j(1.2e-04));
    //TEST("_____________1.2e-05").facets(fp)  (j(1.2e-05));
    //TEST("____________0.000123").facets(fp)  (j(1.23e-04));
    //TEST("____________1.23e-05").facets(fp)  (j(1.23e-05));
    TEST("_______________10000").facets(fp)  (j(strf::fmt(1e+4)));
    TEST("______________10000.").facets(fp) (j(~strf::fmt(1e+4)));
    TEST("_______________1e+05").facets(fp)  (j(strf::fmt(1e+5)));
    TEST("______________1.e+05").facets(fp) (j(~strf::fmt(1e+5)));
    TEST("_____________1200000").facets(fp)  (j(strf::fmt(1.2e+6)));
    TEST("_____________1.2e+06").facets(fp) (j(~strf::fmt(1.2e+6)));
    //TEST("_______________0.001").facets(fp)   (j(strf::fmt(1e-03)));
    //TEST("_______________1e-04").facets(fp)   (j(strf::fmt(1e-04)));
    //TEST("______________0.0001").facets(fp)  (j(~strf::fmt(1e-04)));
    //TEST("______________1.e-05").facets(fp)  (j(~strf::fmt(1e-05)));
    //TEST("_____________0.00012").facets(fp)   (j(strf::fmt(1.2e-04)));
    //TEST("_____________0.00012").facets(fp) (j(~strf::fmt(1.2e-04)));
    //TEST("_____________1.2e-05").facets(fp)  (j(strf::fmt(1.2e-05)));
    //TEST("_____________1.2e-05").facets(fp) (j(~strf::fmt(1.2e-05)));
    //TEST("____________0.000123").facets(fp)  (j(strf::fmt(1.23e-04)));
    //TEST("____________1.23e-05").facets(fp)  (j(strf::fmt(1.23e-05)));
    TEST("_____6.103515625e-05").facets(fp)    (j(strf::fmt(6.103515625e-05)));
    TEST("_______0.00048828125").facets(fp)    (j(strf::fmt(0.00048828125)));

    //----------------------------------------------------------------


    //----------------------------------------------------------------
    // when precision is specified in the general format,
    // do as in printf:
    // - The precision specifies the number of significant digits.
    // - scientific notation is used if the resulting exponent is
    //   less than -4 or greater than or equal to the precision.
    //TEST("______________0.0001").facets(fp)  (j(strf::fmt(1e-4).p());
    //TEST("_______________1e-05").facets(fp)  (j(strf::fmt(1e-5)));
    TEST("_______________1e+01").facets(fp) (j(strf::fmt(12.0).p(1)));
    TEST("_____________1.2e+02").facets(fp) (j(strf::fmt(123.0).p(2)));
    TEST("_________________123").facets(fp) (j(strf::fmt(123.0).p(4)));
    TEST("_______________1e+04").facets(fp) (j(strf::fmt(10000.0).p(4)));
    TEST("_______________10000").facets(fp) (j(strf::fmt(10000.0).p(5)));
    TEST("__________6.1035e-05").facets(fp) (j(strf::fmt(6.103515625e-05).p(5)));

    // and if precision is zero, it treated as 1.
    TEST("_______________1e+01").facets(fp)  (j(strf::fmt(12.0).p(0)));
    TEST("_______________2e+01").facets(fp)  (j(strf::fmt(15.125).p(0)));

    // and when removing digits, the last digit is rounded.
    TEST("_____________1.1e+05").facets(fp) (j(strf::fmt(114999.0).p(2)));
    TEST("_____________1.2e+05").facets(fp) (j(strf::fmt(115000.0).p(2)));
    TEST("_____________1.2e+05").facets(fp) (j(strf::fmt(125000.0).p(2)));
    TEST("_____________1.3e+05").facets(fp) (j(strf::fmt(125001.0).p(2)));

    // and the decimal point appears only if followed by
    // a digit, or if operator~() is used.
    TEST("_______________1e+04").facets(fp)   (j(strf::fmt(10000.0).p(3)));
    TEST("______________1.e+04").facets(fp)  (j(~strf::fmt(10000.0).p(1)));
    TEST("________________123.").facets(fp)  (j(~strf::fmt(123.0).p(3)));

    // and trailing zeros are removed, unless operator~() is used.
    TEST("_____________1.5e+04").facets(fp)   (j(strf::fmt(15000.0).p(3)));
    TEST("____________1.50e+04").facets(fp)  (j(~strf::fmt(15000.0).p(3)));
    TEST("_________________123").facets(fp)   (j(strf::fmt(123.0).p(5)));
    TEST("______________123.00").facets(fp)  (j(~strf::fmt(123.0).p(5)));
    //----------------------------------------------------------------

    // force decimal notation

    TEST("__________________1.").facets(fp)  (j(~strf::fixed(1.0)));
    TEST("___________________1").facets(fp)   (j(strf::fixed(1.0)));
    TEST("__________________+1").facets(fp)  (j(+strf::fixed(1.0)));
    TEST("__________________-1").facets(fp)   (j(strf::fixed(-1.0)));
    TEST("__________________-1").facets(fp)  (j(+strf::fixed(-1.0)));

    TEST("___________________1").facets(fp)  (j(strf::fixed(1.0).p(0)));
    TEST("__________________1.").facets(fp) (j(~strf::fixed(1.0).p(0)));
    TEST("_________________1.0").facets(fp)  (j(strf::fixed(1.0).p(1)));
    TEST("________________1.00").facets(fp)  (j(strf::fixed(1.0).p(2)));
    TEST("______________1.0000").facets(fp)  (j(strf::fixed(1.0).p(4)));
    TEST("_______________0.125").facets(fp)  (j(strf::fixed(0.125)));

    // round when forcing fixed notation
    TEST("_______________1.250").facets(fp) (j(strf::fixed(1.25).p(3)));
    TEST("________________1.25").facets(fp) (j(strf::fixed(1.25).p(2)));
    TEST("_________________1.2").facets(fp) (j(strf::fixed(1.25).p(1)));
    TEST("_________________1.8").facets(fp) (j(strf::fixed(1.75).p(1)));
    TEST("_________________1.3").facets(fp) (j(strf::fixed(1.25048828125).p(1)));
    TEST("______________1.2505").facets(fp) (j(strf::fixed(1.25048828125).p(4)));

    // force scientific notation

    TEST("_______________0e+00").facets(fp)   (j(strf::sci(0.0)));
    TEST("______________0.e+00").facets(fp)  (j(~strf::sci(0.0)));
    TEST("______________+0e+00").facets(fp)  (j(+strf::sci(0.0)));
    TEST("_____________+0.e+00").facets(fp) (j(+~strf::sci(0.0)));

    TEST("_______________1e+04").facets(fp)   (j(strf::sci(1e+4)));
    TEST("______________+1e+04").facets(fp)  (j(+strf::sci(1e+4)));
    TEST("______________1.e+04").facets(fp)  (j(~strf::sci(1e+4)));
    TEST("_____________+1.e+04").facets(fp) (j(~+strf::sci(1e+4)));
    TEST("______________-1e+04").facets(fp)   (j(strf::sci(-1e+4)));
    TEST("______________-1e+04").facets(fp)  (j(+strf::sci(-1e+4)));
    TEST("_____________-1.e+04").facets(fp) (j(~strf::sci(-1e+4)));

    TEST("_____________1.0e+04").facets(fp)   (j(strf::sci(1e+4).p(1)));
    TEST("_____________1.0e+04").facets(fp)   (j(strf::sci(1e+4).p(1)));
    TEST("___________+1.00e+04").facets(fp)  (j(+strf::sci(1e+4).p(2)));
    TEST("______________1.e+04").facets(fp)  (j(~strf::sci(1e+4).p(0)));
    TEST("_____________+1.e+04").facets(fp) (j(~+strf::sci(1e+4).p(0)));
    TEST("______________-1e+04").facets(fp)  (j(+strf::sci(-1e+4).p(0)));
    TEST("_____________-1.e+04").facets(fp)  (j(~strf::sci(-1e+4).p(0)));

    // rounding when forcing scientific notation
    TEST("____________1.25e+02").facets(fp) (j(strf::sci(125.0).p(2)));
    TEST("_____________1.2e+02").facets(fp) (j(strf::sci(125.0).p(1)));
    TEST("_____________1.2e+02").facets(fp) (j(strf::sci(115.0).p(1)));
    TEST("_____________1.3e+06").facets(fp) (j(strf::sci(1250001.0).p(1)));
    TEST("__________8.1928e+03").facets(fp) (j(strf::sci(8192.75).p(4)));
    TEST("__________8.1922e+03").facets(fp) (j(strf::sci(8192.25).p(4)));
    TEST("__________1.0242e+03").facets(fp) (j(strf::sci(1024.25).p(4)));
    TEST("_____________1.7e+01").facets(fp) (j(strf::sci(16.50006103515625).p(1)));

    // add trailing zeros if precision requires
    TEST("___________1.250e+02").facets(fp) (j(strf::sci(125.0).p(3)));
    TEST("_________6.25000e-02").facets(fp) (j(strf::sci(0.0625).p(5)));
    TEST("________8.192750e+03").facets(fp) (j(strf::sci(8192.75).p(6)));

/*
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
    TEST("____________________").facets(fp)    (j(strf::fmt()));
*/
}

int main()
{
    {
        BOOST_TEST_LABEL << "default facets";
        basic_tests(strf::pack());
    }

    {
        BOOST_TEST_LABEL << "with punctuation";
        basic_tests(strf::no_grouping<10>{});
    }

    auto p = strf::monotonic_grouping<10>{3}.decimal_point(',').thousands_sep(':');
    constexpr auto j = strf::join_right(20, '_');

    TEST("_________________1,5").facets(p) (j(1.5));
    TEST("_____6,103515625e-05").facets(p) (j(6.103515625e-05));
    TEST("__________6,1035e-05").facets(p) (j(strf::fmt(6.103515625e-05).p(5)));
    TEST("_________6,10352e-05").facets(p) (j(strf::sci(6.103515625e-05).p(5)));
    TEST("_______0,00048828125").facets(p) (j(0.00048828125));
    TEST("_____2:048,001953125").facets(p) (j(2048.001953125));
    TEST("___________2:048,002").facets(p) (j(strf::fixed(2048.001953125).p(3)));
    TEST("___________1:000:000").facets(p) (j(strf::fixed(1000000.0)));
    TEST("_________+1:000:000,").facets(p) (j(~+strf::fixed(1000000.0)));
    TEST("_____+1:000:000,0000").facets(p) (j(~+strf::fixed(1000000.0).p(4)));

    TEST("___________1:024,125").facets(p) (j(1024.125f));
    TEST("_______+1,024125e+03").facets(p) (j(+strf::sci(1024.125f)));


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

