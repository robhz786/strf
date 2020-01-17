//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "test_utils.hpp"

template <typename T>
struct is_char32: public std::is_same<T, char32_t>
{
};

int main()
{
    // conversion

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

    TEST("a") ( U'a' );
    TEST("aaaa") ( strf::multi(U'a', 4) );
    TEST("  aa") ( strf::multi(U'a', 2) > 4 );

    TEST("    a") ( strf::right(U'a', 5) );
    TEST("a    ") ( strf::left(U'a', 5)  );
    TEST("aa   ") ( strf::multi(U'a', 2) < 5 );

    TEST("....a") ( strf::right(U'a', 5, '.')  );
    TEST("a....") ( strf::left(U'a', 5, '.')   );
    TEST("..a..") ( strf::center(U'a', 5, '.') );

    TEST("...aa") ( strf::right(U'a', 5, '.').multi(2)  );
    TEST("aa...") ( strf::left(U'a', 5, '.').multi(2)   );
    TEST(".aa..") ( strf::center(U'a', 5, '.').multi(2) );

    TEST(".....") ( strf::right(U'a', 5, '.').multi(0)  );
    TEST(".....") ( strf::left(U'a', 5, '.').multi(0)   );
    TEST(".....") ( strf::center(U'a', 5, '.').multi(0) );

    TEST("a")      ( strf::join_left(0, '.')(U'a') );
    TEST("   a")   ( strf::join_left(1, '.')(strf::right(U'a', 4)) );
    TEST("   a..") ( strf::join_left(6, '.')(strf::right(U'a', 4)) );

    TEST("  aa")   ( strf::join_left(2, '.')(strf::multi(U'a', 2) > 4) );
    TEST("  aa")   ( strf::join_left(2, '.')(strf::multi(U'a', 2) > 4) );
    TEST("  aa")   ( strf::join_left(4, '.')(strf::multi(U'a', 2) > 4) );
    TEST("  aa..") ( strf::join_left(6, '.')(strf::multi(U'a', 2) > 4) );

    TEST("aaaa")   ( strf::join_left(2, '.')(strf::multi(U'a', 4) > 2) );
    TEST("aaaa")   ( strf::join_left(4, '.')(strf::multi(U'a', 4) > 2) );
    TEST("aaaa..") ( strf::join_left(6, '.')(strf::multi(U'a', 4) > 2) );

    TEST("aaaa")   ( strf::join_left(2, '.')(strf::multi(U'a', 4) > 4) );
    TEST("aaaa")   ( strf::join_left(4, '.')(strf::multi(U'a', 4) > 4) );
    TEST("aaaa..") ( strf::join_left(6, '.')(strf::multi(U'a', 4) > 4) );

    return test_finish();
}













