#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST test<__LINE__>

int main()
{

    /*
    TEST( "abcd", {}, {'a', u'b', U'c', L'd'});
    TEST(u"abcd", {}, {'a', u'b', U'c', L'd'});
    TEST(U"abcd", {}, {'a', u'b', U'c', L'd'});
    TEST(L"abcd", {}, {'a', u'b', U'c', L'd'});

    TEST (u8"\ud7ff\ue000\uffff\U00010000\U0010ffff", {},
          {U'\ud7ff', U'\ue000', U'\uffff', U'\U00010000', U'\U0010ffff'});

    TEST (u"\ud7ff\ue000\uffff\U00010000\U0010ffff", {},
          {U'\ud7ff', U'\ue000', U'\uffff', U'\U00010000', U'\U0010ffff'});

    TEST (L"\ud7ff\ue000\uffff\U00010000\U0010ffff", {},
          {U'\ud7ff', U'\ue000', U'\uffff', U'\U00010000', U'\U0010ffff'});

    TEST (U"\ud7ff\ue000\uffff\U00010000\U0010ffff", {},
          {U'\ud7ff', U'\ue000', U'\uffff', U'\U00010000', U'\U0010ffff'});
    
    // invalid codepoints:
    TEST ( "", {}, {static_cast<char32_t>(0x110000)});
    TEST (u"", {}, {static_cast<char32_t>(0xd800), static_cast<char32_t>(0xdfff)});
    */ 
    return  boost::report_errors();
}













