#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

#define TEST_RF(EXPECTED, RF) make_tester((EXPECTED), __FILE__, __LINE__, RF)

#define TEST_ERR(EXPECTED, ERR) make_tester((EXPECTED), __FILE__, __LINE__, ERR)

#define TEST_ERR_RF(EXPECTED, ERR, RF) make_tester((EXPECTED), __FILE__, __LINE__, ERR, RF)

namespace strf = boost::stringify::v0;
int main()
{

    TEST(u"abc\u0200\uD500\U0010FF00")(u8"abc\u0200\uD500\U0010FF00");

    
    const char* str8_with_surr=  "--\xED\xA0\x80--";
    const char* str8_no_surr = u8"--\uFFFE--";
    
    const char16_t  str16_with_surr[] = {u'-', u'-', 0xD800, u'-', u'-', u'\0'};
    const char16_t* str16_no_surr = u"--\uFFFE--";


    (void) str8_with_surr;
    (void) str8_no_surr;
    (void) str16_with_surr;
    (void) str16_no_surr;

    
    TEST(str16_with_surr)
        .facets(strf::to_utf16<char16_t>().keep_surrogates(true))
        .facets(strf::from_utf8())
        .as(str8_with_surr)
        ();

    // TODO ...

    
    return report_errors() || boost::report_errors();
}
