//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

template <typename T>
struct is_char32: public std::is_same<T, char32_t>
{
};

int main()
{
    namespace strf = boost::stringify::v0;

    // conversion

    TEST (u8"\ud7ff")     .exception(U'\ud7ff');
    TEST (u8"\ue000")     .exception(U'\ue000');
    TEST (u8"\uffff")     .exception(U'\uffff');
    TEST (u8"\U00010000") .exception(U'\U00010000');
    TEST (u8"\U0010ffff") .exception(U'\U0010ffff');

    TEST (u"\ud7ff")     .exception(U'\ud7ff');
    TEST (u"\ue000")     .exception(U'\ue000');
    TEST (u"\uffff")     .exception(U'\uffff');
    TEST (u"\U00010000") .exception(U'\U00010000');
    TEST (u"\U0010ffff") .exception(U'\U0010ffff');

    TEST (L"\ud7ff")     .exception(U'\ud7ff');
    TEST (L"\ue000")     .exception(U'\ue000');
    TEST (L"\uffff")     .exception(U'\uffff');
    TEST (L"\U00010000") .exception(U'\U00010000');
    TEST (L"\U0010ffff") .exception(U'\U0010ffff');

    TEST (U"\ud7ff")     .exception(U'\ud7ff');
    TEST (U"\ue000")     .exception(U'\ue000');
    TEST (U"\uffff")     .exception(U'\uffff');
    TEST (U"\U00010000") .exception(U'\U00010000');
    TEST (U"\U0010ffff") .exception(U'\U0010ffff');

    TEST("a") .exception( U'a' );
    TEST("aaaa") .exception( strf::multi(U'a', 4) );
    TEST("  aa") .exception( strf::multi(U'a', 2) > 4 );

    TEST("    a") .exception( strf::right(U'a', 5) );
    TEST("a    ") .exception( strf::left(U'a', 5)  );
    TEST("aa   ") .exception( strf::multi(U'a', 2) < 5 );

    TEST("....a") .exception( strf::right(U'a', 5, '.')  );
    TEST("a....") .exception( strf::left(U'a', 5, '.')   );
    TEST("..a..") .exception( strf::center(U'a', 5, '.') );

    TEST("...aa") .exception( strf::right(U'a', 5, '.').multi(2)  );
    TEST("aa...") .exception( strf::left(U'a', 5, '.').multi(2)   );
    TEST(".aa..") .exception( strf::center(U'a', 5, '.').multi(2) );

    TEST(".....") .exception( strf::right(U'a', 5, '.').multi(0)  );
    TEST(".....") .exception( strf::left(U'a', 5, '.').multi(0)   );
    TEST(".....") .exception( strf::center(U'a', 5, '.').multi(0) );

    TEST("a")      .exception( strf::join_left(0, '.')(U'a') );
    TEST("   a")   .exception( strf::join_left(1, '.')(strf::right(U'a', 4)) );
    TEST("   a..") .exception( strf::join_left(6, '.')(strf::right(U'a', 4)) );

    TEST("  aa")   .exception( strf::join_left(2, '.')(strf::multi(U'a', 2) > 4) );
    TEST("  aa")   .exception( strf::join_left(2, '.')(strf::multi(U'a', 2) > 4) );
    TEST("  aa")   .exception( strf::join_left(4, '.')(strf::multi(U'a', 2) > 4) );
    TEST("  aa..") .exception( strf::join_left(6, '.')(strf::multi(U'a', 2) > 4) );

    TEST("aaaa")   .exception( strf::join_left(2, '.')(strf::multi(U'a', 4) > 2) );
    TEST("aaaa")   .exception( strf::join_left(4, '.')(strf::multi(U'a', 4) > 2) );
    TEST("aaaa..") .exception( strf::join_left(6, '.')(strf::multi(U'a', 4) > 2) );

    TEST("aaaa")   .exception( strf::join_left(2, '.')(strf::multi(U'a', 4) > 4) );
    TEST("aaaa")   .exception( strf::join_left(4, '.')(strf::multi(U'a', 4) > 4) );
    TEST("aaaa..") .exception( strf::join_left(6, '.')(strf::multi(U'a', 4) > 4) );

    return report_errors() || boost::report_errors();
}













