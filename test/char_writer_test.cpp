#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST testf<__LINE__>

int main()
{
    
    TEST( "a") () ( 'a');
    TEST( "a") () (U'a');
    TEST(u"a") () (u'a');
    TEST(u"a") () (U'a');
    TEST(U"a") () (U'a');
    TEST(L"a") () (L'a');
    TEST(L"a") () (U'a');
    
    TEST (u8"\ud7ff")     () (U'\ud7ff');
    TEST (u8"\ue000")     () (U'\ue000');
    TEST (u8"\uffff")     () (U'\uffff');
    TEST (u8"\U00010000") () (U'\U00010000');
    TEST (u8"\U0010ffff") () (U'\U0010ffff');

    TEST (u"\ud7ff")     () (U'\ud7ff');
    TEST (u"\ue000")     () (U'\ue000');
    TEST (u"\uffff")     () (U'\uffff');
    TEST (u"\U00010000") () (U'\U00010000');
    TEST (u"\U0010ffff") () (U'\U0010ffff');

    TEST (L"\ud7ff")     () (U'\ud7ff');
    TEST (L"\ue000")     () (U'\ue000');
    TEST (L"\uffff")     () (U'\uffff');
    TEST (L"\U00010000") () (U'\U00010000');
    TEST (L"\U0010ffff") () (U'\U0010ffff');

    TEST (U"\ud7ff")     () (U'\ud7ff');
    TEST (U"\ue000")     () (U'\ue000');
    TEST (U"\uffff")     () (U'\uffff');
    TEST (U"\U00010000") () (U'\U00010000');
    TEST (U"\U0010ffff") () (U'\U0010ffff');
    
    // invalid codepoints:
    TEST( "") () (static_cast<char32_t>(0x110000));
    TEST(u"") () (static_cast<char32_t>(0xd800));
    TEST(u"") () (static_cast<char32_t>(0xdfff));
   
    return  boost::report_errors();
}













