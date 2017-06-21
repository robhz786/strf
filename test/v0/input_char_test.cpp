//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST testf<__LINE__>

int main()
{
    
    TEST( "a") ( 'a');
    TEST( "a") (U'a');
    TEST(u"a") (u'a');
    TEST(u"a") (U'a');
    TEST(U"a") (U'a');
    TEST(L"a") (L'a');
    TEST(L"a") (U'a');
    TEST(u"  a") ({boost::stringify::v0::join_right(3), {u'a'}});
    
   
    TEST (u8"\ud7ff")     (U'\ud7ff');
    TEST (u8"\ue000")     (U'\ue000');
    TEST (u8"\uffff")     (U'\uffff');
    TEST (u8"\U00010000") (U'\U00010000');
    TEST (u8"\U0010ffff") (U'\U0010ffff');

    TEST (u"\ud7ff")     (U'\ud7ff');
    TEST (u"\ue000")     (U'\ue000');
    TEST (u"\uffff")     (U'\uffff');
    TEST (u"\U00010000") (U'\U00010000');
    TEST (u"\U0010ffff") (U'\U0010ffff');

    TEST (L"\ud7ff")     (U'\ud7ff');
    TEST (L"\ue000")     (U'\ue000');
    TEST (L"\uffff")     (U'\uffff');
    TEST (L"\U00010000") (U'\U00010000');
    TEST (L"\U0010ffff") (U'\U0010ffff');

    TEST (U"\ud7ff")     (U'\ud7ff');
    TEST (U"\ue000")     (U'\ue000');
    TEST (U"\uffff")     (U'\uffff');
    TEST (U"\U00010000") (U'\U00010000');
    TEST (U"\U0010ffff") (U'\U0010ffff');
    
    // invalid codepoints:
    TEST( "") (static_cast<char32_t>(0x110000));
    TEST(u"") (static_cast<char32_t>(0xd800));
    TEST(u"") (static_cast<char32_t>(0xdfff));


    // width and justificafion

    // auto f1 = strf::make_ftuple(strf::fill(U'~'), strf::width(4));
    // TEST( "~~~ab~~~c~") .with(f1) ( 'a', { 'b', "<"}, { 'c', {2, "<"}});
    // TEST( "~~~ab~~~c~") .with(f1) (U'a', {U'b', "<"}, {U'c', {2, "<"}});
    // TEST(u"~~~ab~~~c~") .with(f1) (U'a', {U'b', "<"}, {U'c', {2, "<"}});

    // auto f2 = strf::make_ftuple(strf::fill(U'~'), strf::width(4), strf::internal<>);
    // TEST( "~~~ab~~~c~") .with(f2) ( 'a', { 'b', "<"}, { 'c', {2, "<"}});
    // TEST( "~~~ab~~~c~") .with(f2) (U'a', {U'b', "<"}, {U'c', {2, "<"}});
    // TEST(u"~~~ab~~~c~") .with(f2) (U'a', {U'b', "<"}, {U'c', {2, "<"}});
    
    // auto f3 = strf::make_ftuple(strf::fill(U'~'), strf::width(4), strf::left<>);
    // TEST( "a~~~~~~b~~c") .with(f3) ( 'a', { 'b', ">"}, { 'c', {2, "%"}});
    // TEST( "a~~~~~~b~~c") .with(f3) (U'a', {U'b', ">"}, {U'c', {2, "%"}});
    // TEST(u"a~~~~~~b~~c") .with(f3) (U'a', {U'b', ">"}, {U'c', {2, "%"}});

    // TEST( "abc") .with(strf::width(1)) ( 'a', { 'b', "<"}, { 'c', "%"});
    // TEST( "abc") .with(strf::width(0)) ( 'a', { 'b', "<"}, { 'c', "%"});
    // TEST( "abc") .with(strf::width(1)) (U'a', {U'b', "<"}, {U'c', "%"});
    // TEST( "abc") .with(strf::width(0)) (U'a', {U'b', "<"}, {U'c', "%"});
    // TEST(u"abc") .with(strf::width(1)) (U'a', {U'b', "<"}, {U'c', "%"});
    // TEST(u"abc") .with(strf::width(0)) (U'a', {U'b', "<"}, {U'c', "%"});

    int rc = boost::report_errors();
    return rc;
}













