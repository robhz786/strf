//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <limits>
#include <vector>
#include "test_utils.hpp"
#include <cstdlib>


template <typename FPack>
void basic_tests(const FPack& fp)
{
    constexpr auto j = strf::join_right(20, '_');
    constexpr auto quiet_nan = std::numeric_limits<double>::quiet_NaN();
    constexpr auto signaling_nan = std::numeric_limits<double>::signaling_NaN();
    constexpr auto infinity = std::numeric_limits<double>::infinity();

    TEST("_________________nan").with(fp)  (j(quiet_nan));
    TEST("_________________nan").with(fp)  (j(signaling_nan));
    TEST("_________________inf").with(fp)  (j(infinity));
    TEST("________________-inf").with(fp)  (j(-infinity));

    TEST("_________________nan").with(fp)  (j(strf::fmt(quiet_nan)));
    TEST("_________________nan").with(fp)  (j(strf::fmt(signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fmt(signaling_nan)));
    TEST("________________+inf").with(fp)  (j(+strf::fmt(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::fmt(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::fmt(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::fmt(-infinity)));

    TEST("_________________nan").with(fp)  (j(strf::fixed(quiet_nan)));
    TEST("_________________nan").with(fp)  (j(strf::fixed(signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fixed(quiet_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::fixed(signaling_nan)));
    TEST("_________________inf").with(fp)  (j(strf::fixed(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::fixed(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::fixed(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::fixed(-infinity)));

    TEST("_________________nan").with(fp)  (j(strf::sci(quiet_nan)));
    TEST("_________________nan").with(fp)  (j(strf::sci(signaling_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::sci(quiet_nan)));
    TEST("________________+nan").with(fp)  (j(+strf::sci(signaling_nan)));
    TEST("_________________inf").with(fp)  (j(strf::sci(infinity)));
    TEST("________________+inf").with(fp) (j(+strf::sci(infinity)));
    TEST("________________-inf").with(fp)  (j(strf::sci(-infinity)));
    TEST("________________-inf").with(fp) (j(+strf::sci(-infinity)));

    TEST("___________________0").with(fp)  (j(0.0));
    TEST("__________________-0").with(fp)  (j(-0.0));
    TEST("___________________1").with(fp)  (j(1.0));
    TEST("__________________-1").with(fp)  (j(-1.0));
    TEST("_________________1.5").with(fp)  (j(1.5));
    TEST("_____6.103515625e-05").with(fp)  (j(6.103515625e-05));
    TEST("_______0.00048828125").with(fp)  (j(0.00048828125));
    TEST("______2048.001953125").with(fp)  (j(2048.001953125));

    TEST("___________~~~~~+nan")(j(+strf::right (quiet_nan, 9, '~')));
    TEST("___________+nan~~~~~")(j(+strf::left  (quiet_nan, 9, '~')));
    TEST("___________~~+nan~~~")(j(+strf::center(quiet_nan, 9, '~')));
    TEST("___________+~~~~~nan")(j(+strf::split (quiet_nan, 9, '~')));

    TEST("___________~~~~~+inf")(j(+strf::right (infinity, 9, '~')));
    TEST("___________+inf~~~~~")(j(+strf::left  (infinity, 9, '~')));
    TEST("___________~~+inf~~~")(j(+strf::center(infinity, 9, '~')));
    TEST("___________+~~~~~inf")(j(+strf::split (infinity, 9, '~')));

    TEST("___________~~~~~+1.5")(j(+strf::right (1.5, 9, '~')));
    TEST("___________+1.5~~~~~")(j(+strf::left  (1.5, 9, '~')));
    TEST("___________~~+1.5~~~")(j(+strf::center(1.5, 9, '~')));
    TEST("___________+~~~~~1.5")(j(+strf::split (1.5, 9, '~')));

    TEST("___________________0").with(fp) (j(strf::fmt(0.0)));
    TEST("__________________-0").with(fp) (j(strf::fmt(-0.0)));
    TEST("___________________1").with(fp) (j(strf::fmt(1.0)));
    TEST("__________________-1").with(fp) (j(strf::fmt(-1.0)));
    TEST("_________________1.5").with(fp) (j(strf::fmt(1.5)));

    TEST("__________________+1").with(fp) (j(+strf::fmt(1.0)));
    TEST("__________________-1").with(fp) (j(+strf::fmt(-1.0)));
    TEST("__________________1.").with(fp) (j(~strf::fmt(1.0)));
    TEST("_________________-1.").with(fp) (j(~strf::fmt(-1.0)));
    TEST("_________________+1.").with(fp)(j(~+strf::fmt(1.0)));
    TEST("_________________-1.").with(fp)(j(+~strf::fmt(-1.0)));

    //----------------------------------------------------------------
    // when precision is not specified, the general format selects the
    // scientific notation if it is shorter than the fixed notation:
    TEST("_______________10000").with(fp)  (j(1e+4));
    TEST("_______________1e+05").with(fp)  (j(1e+5));
    TEST("_____________1200000").with(fp)  (j(1.2e+6));
    TEST("_____________1.2e+07").with(fp)  (j(1.2e+7));
    TEST("_______________0.001").with(fp)  (j(1e-03));
    TEST("_______________1e-04").with(fp)  (j(1e-04));
    TEST("_____________0.00012").with(fp)  (j(1.2e-04));
    TEST("_____________1.2e-05").with(fp)  (j(1.2e-05));
    TEST("____________0.000123").with(fp)  (j(1.23e-04));
    TEST("____________1.23e-05").with(fp)  (j(1.23e-05));
    TEST("_______________10000").with(fp)  (j(strf::fmt(1e+4)));
    TEST("______________10000.").with(fp) (j(~strf::fmt(1e+4)));
    TEST("_______________1e+05").with(fp)  (j(strf::fmt(1e+5)));
    TEST("______________1.e+05").with(fp) (j(~strf::fmt(1e+5)));
    TEST("_____________1200000").with(fp)  (j(strf::fmt(1.2e+6)));
    TEST("_____________1.2e+06").with(fp) (j(~strf::fmt(1.2e+6)));
    TEST("_______________0.001").with(fp)   (j(strf::fmt(1e-03)));
    TEST("_______________1e-04").with(fp)   (j(strf::fmt(1e-04)));
    TEST("______________0.0001").with(fp)  (j(~strf::fmt(1e-04)));
    TEST("______________1.e-05").with(fp)  (j(~strf::fmt(1e-05)));
    TEST("_____________0.00012").with(fp)   (j(strf::fmt(1.2e-04)));
    TEST("_____________0.00012").with(fp) (j(~strf::fmt(1.2e-04)));
    TEST("_____________1.2e-05").with(fp)  (j(strf::fmt(1.2e-05)));
    TEST("_____________1.2e-05").with(fp) (j(~strf::fmt(1.2e-05)));
    TEST("____________0.000123").with(fp)  (j(strf::fmt(1.23e-04)));
    TEST("____________1.23e-05").with(fp)  (j(strf::fmt(1.23e-05)));
    TEST("_____6.103515625e-05").with(fp)    (j(strf::fmt(6.103515625e-05)));
    TEST("_______0.00048828125").with(fp)    (j(strf::fmt(0.00048828125)));

    //----------------------------------------------------------------


    //----------------------------------------------------------------
    // when precision is specified in the general format,
    // do as in printf:
    // - The precision specifies the number of significant digits.
    // - scientific notation is used if the resulting exponent is
    //   less than -4 or greater than or equal to the precision.
    //TEST("______________0.0001").with(fp)  (j(strf::fmt(1e-4).p());
    //TEST("_______________1e-05").with(fp)  (j(strf::fmt(1e-5)));
    TEST("_______________1e+01").with(fp) (j(strf::fmt(12.0).p(1)));
    TEST("_____________1.2e+02").with(fp) (j(strf::fmt(123.0).p(2)));
    TEST("_________________123").with(fp) (j(strf::fmt(123.0).p(4)));
    TEST("_______________1e+04").with(fp) (j(strf::fmt(10000.0).p(4)));
    TEST("_______________10000").with(fp) (j(strf::fmt(10000.0).p(5)));
    TEST("__________6.1035e-05").with(fp) (j(strf::fmt(6.103515625e-05).p(5)));

    // and if precision is zero, it treated as 1.
    TEST("_______________1e+01").with(fp)  (j(strf::fmt(12.0).p(0)));
    TEST("_______________2e+01").with(fp)  (j(strf::fmt(15.125).p(0)));

    // and when removing digits, the last digit is rounded.
    TEST("_____________1.1e+05").with(fp) (j(strf::fmt(114999.0).p(2)));
    TEST("_____________1.2e+05").with(fp) (j(strf::fmt(115000.0).p(2)));
    TEST("_____________1.2e+05").with(fp) (j(strf::fmt(125000.0).p(2)));
    TEST("_____________1.3e+05").with(fp) (j(strf::fmt(125001.0).p(2)));

    // and the decimal point appears only if followed by
    // a digit, or if operator~() is used.
    TEST("_______________1e+04").with(fp)   (j(strf::fmt(10000.0).p(3)));
    TEST("______________1.e+04").with(fp)  (j(~strf::fmt(10000.0).p(1)));
    TEST("________________123.").with(fp)  (j(~strf::fmt(123.0).p(3)));

    // and trailing zeros are removed, unless operator~() is used.
    TEST("_____________1.5e+04").with(fp)   (j(strf::fmt(15000.0).p(3)));
    TEST("____________1.50e+04").with(fp)  (j(~strf::fmt(15000.0).p(3)));
    TEST("_________________123").with(fp)   (j(strf::fmt(123.0).p(5)));
    TEST("______________123.00").with(fp)  (j(~strf::fmt(123.0).p(5)));
    //----------------------------------------------------------------

    // force decimal notation

    TEST("__________________1.").with(fp)  (j(~strf::fixed(1.0)));
    TEST("___________________1").with(fp)   (j(strf::fixed(1.0)));
    TEST("__________________+1").with(fp)  (j(+strf::fixed(1.0)));
    TEST("__________________-1").with(fp)   (j(strf::fixed(-1.0)));
    TEST("__________________-1").with(fp)  (j(+strf::fixed(-1.0)));

    TEST("___________________1").with(fp)  (j(strf::fixed(1.0).p(0)));
    TEST("__________________1.").with(fp) (j(~strf::fixed(1.0).p(0)));
    TEST("_________________1.0").with(fp)  (j(strf::fixed(1.0).p(1)));
    TEST("________________1.00").with(fp)  (j(strf::fixed(1.0).p(2)));
    TEST("______________1.0000").with(fp)  (j(strf::fixed(1.0).p(4)));
    TEST("_______________0.125").with(fp)  (j(strf::fixed(0.125)));

    // round when forcing fixed notation
    TEST("_______________1.250").with(fp) (j(strf::fixed(1.25).p(3)));
    TEST("________________1.25").with(fp) (j(strf::fixed(1.25).p(2)));
    TEST("_________________1.2").with(fp) (j(strf::fixed(1.25).p(1)));
    TEST("_________________1.8").with(fp) (j(strf::fixed(1.75).p(1)));
    TEST("_________________1.3").with(fp) (j(strf::fixed(1.25048828125).p(1)));
    TEST("______________1.2505").with(fp) (j(strf::fixed(1.25048828125).p(4)));

    // force scientific notation

    TEST("_______________0e+00").with(fp)   (j(strf::sci(0.0)));
    TEST("______________0.e+00").with(fp)  (j(~strf::sci(0.0)));
    TEST("______________+0e+00").with(fp)  (j(+strf::sci(0.0)));
    TEST("_____________+0.e+00").with(fp) (j(+~strf::sci(0.0)));

    TEST("_______________1e+04").with(fp)   (j(strf::sci(1e+4)));
    TEST("______________+1e+04").with(fp)  (j(+strf::sci(1e+4)));
    TEST("______________1.e+04").with(fp)  (j(~strf::sci(1e+4)));
    TEST("_____________+1.e+04").with(fp) (j(~+strf::sci(1e+4)));
    TEST("______________-1e+04").with(fp)   (j(strf::sci(-1e+4)));
    TEST("______________-1e+04").with(fp)  (j(+strf::sci(-1e+4)));
    TEST("_____________-1.e+04").with(fp) (j(~strf::sci(-1e+4)));

    TEST("_____________1.0e+04").with(fp)   (j(strf::sci(1e+4).p(1)));
    TEST("_____________1.0e+04").with(fp)   (j(strf::sci(1e+4).p(1)));
    TEST("___________+1.00e+04").with(fp)  (j(+strf::sci(1e+4).p(2)));
    TEST("______________1.e+04").with(fp)  (j(~strf::sci(1e+4).p(0)));
    TEST("_____________+1.e+04").with(fp) (j(~+strf::sci(1e+4).p(0)));
    TEST("______________-1e+04").with(fp)  (j(+strf::sci(-1e+4).p(0)));
    TEST("_____________-1.e+04").with(fp)  (j(~strf::sci(-1e+4).p(0)));

    // rounding when forcing scientific notation
    TEST("____________1.25e+02").with(fp) (j(strf::sci(125.0).p(2)));
    TEST("_____________1.2e+02").with(fp) (j(strf::sci(125.0).p(1)));
    TEST("_____________1.2e+02").with(fp) (j(strf::sci(115.0).p(1)));
    TEST("_____________1.3e+06").with(fp) (j(strf::sci(1250001.0).p(1)));
    TEST("__________8.1928e+03").with(fp) (j(strf::sci(8192.75).p(4)));
    TEST("__________8.1922e+03").with(fp) (j(strf::sci(8192.25).p(4)));
    TEST("__________1.0242e+03").with(fp) (j(strf::sci(1024.25).p(4)));
    TEST("_____________1.7e+01").with(fp) (j(strf::sci(16.50006103515625).p(1)));

    // add trailing zeros if precision requires
    TEST("___________1.250e+02").with(fp) (j(strf::sci(125.0).p(3)));
    TEST("_________6.25000e-02").with(fp) (j(strf::sci(0.0625).p(5)));
    TEST("________8.192750e+03").with(fp) (j(strf::sci(8192.75).p(6)));
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

int main()
{
    {
        TEST_SCOPE_DESCRIPTION("default facets");
        basic_tests(strf::pack());
    }

    {
        TEST_SCOPE_DESCRIPTION("with punctuation");
        basic_tests(strf::no_grouping<10>{});
    }

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
    constexpr auto j = strf::join_right(20, '_');
    {
        // check whether it correctly selects the shortest representation
        auto p = strf::monotonic_grouping<10>{1}.thousands_sep(',');
        TEST("_______________1,0,0").with(p) (j(100.0));
        TEST("_______________1e+03").with(p) (j(1000.0));
        TEST("_____________1,0,0,0").with(p) (j(strf::fixed(1000.0)));
        TEST("________1.000005e+05").with(p) (j(100000.5));
        TEST("_________1,0,0,0,0.5").with(p) (j(10000.5));
    }

    {
        auto p = strf::monotonic_grouping<10>{3}.decimal_point(',').thousands_sep(':');

        TEST("_________________1,5").with(p) (j(1.5));
        TEST("_________________0,1").with(p) (j(0.1));
        TEST("_____6,103515625e-05").with(p) (j(6.103515625e-05));
        TEST("__________6,1035e-05").with(p) (j(strf::fmt(6.103515625e-05).p(5)));
        TEST("_________6,10352e-05").with(p) (j(strf::sci(6.103515625e-05).p(5)));
        TEST("_______0,00048828125").with(p) (j(0.00048828125));
        TEST("_____2:048,001953125").with(p) (j(2048.001953125));
        TEST("___________2:048,002").with(p) (j(strf::fixed(2048.001953125).p(3)));
        TEST("___________1:000:000").with(p) (j(strf::fixed(1000000.0)));
        TEST("_________+1:000:000,").with(p) (j(~+strf::fixed(1000000.0)));
        TEST("_____+1:000:000,0000").with(p) (j(~+strf::fixed(1000000.0).p(4)));

        TEST("___________1:024,125").with(p) (j(1024.125f));
        TEST("_______+1,024125e+03").with(p) (j(+strf::sci(1024.125f)));
    }

    return test_finish();
}

