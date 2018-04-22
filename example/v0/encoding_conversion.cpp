#include <boost/assert.hpp>
#include <boost/stringify.hpp>

#include <iostream>

void input_ouput_different_char_types()
{
    //[input_output_different_char_types
    namespace strf = boost::stringify::v0;

    auto str   = strf::make_string    ("aaa-", u"bbb-", U"ccc-", L"ddd");
    auto str16 = strf::make_u16string ("aaa-", u"bbb-", U"ccc-", L"ddd");
    auto str32 = strf::make_u32string ("aaa-", u"bbb-", U"ccc-", L"ddd");
    auto wstr  = strf::make_wstring   ("aaa-", u"bbb-", U"ccc-", L"ddd");
    
    BOOST_ASSERT(str.value()   ==  "aaa-bbb-ccc-ddd");
    BOOST_ASSERT(str16.value() == u"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(str32.value() == U"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(wstr.value()  == L"aaa-bbb-ccc-ddd");
    //]
}


// void mtf8 ()
// {

//     //[ mtf8_samples
//     namespace strf = boost::stringify::v0;

//     // from UTF-16 to  Modified UTF-8 (MTF-8)
//     std::u16string str_utf16 {u"---\0---", 7};
//     auto str_mtf8 = strf::make_string
//         .facets(strf::to_mtf8())
//         (str_utf16);

//     BOOST_ASSERT(0 == str_mtf8.compare(0, 8, "---\xC0\x80---", 8));

//     // from Modified UTF-8 (MTF-8) to UTF-8
//     auto str_utf8 = strf::make_string
//         .facets(strf::from_mtf8())
//         (str_mtf8);

//     BOOST_ASSERT(0 == str_utf8.compare(0, 7,"---\0---", 7));
    
//     //]
// }

// void windows_1252 ()
// {
//     //[ windows_1252_samples
//     namespace strf = boost::stringify::v0;

//     // From Windows-1252 to utf8
//     auto str_utf8 = strf::make_string
//         .facets(strf::from_windows_1252())
//         (u8"--\x80--\x99--\x9D--"); // '\x9D' is invalid

//     BOOST_ASSERT(str_utf8 == u8"--\u20AC--\u2122--\uFFFE--");

//     // Back to Windows-1252
//     auto str_win1252 = strf::make_string
//         .facets(strf::to_windows_1252())
//         (str_utf8);

//     BOOST_ASSERT(str_win1252 == "--\x80--\x99--?--");
//     //]
// }


// void sanitise ()
// {
//     //[ sanitise
//     namespace strf = boost::stringify::v0;
//     auto str = strf::make_string("--\x99--", strf::sani("--\x99--"));
//     BOOST_ASSERT(str == u8"--\x99----\uFFFE--");
//     //]
// }

// void error_signal ()
// {
//     //[ error_signal
//     namespace strf = boost::stringify::v0;

//     //
//     // 1) using an alternative character to signal the error
//     //
//     auto str = strf::make_string
//         .facets(strf::to_utf8().on_error(U'\u2639'))
//         ("--\x99--", strf::sani("--\x99--"));
    
//     BOOST_ASSERT(str == u8"--\x99----\u2639--");

    
//     //
//     // 2) using error code
//     //
//     auto err_code = std::make_error_code(std::errc::illegal_byte_sequence);
//     auto xstr = strf::make_string
//         .facets(strf::to_utf8().on_error(err_code))
//         ("--\x99--", strf::sani("--\x99--"));
    
//     BOOST_ASSERT(!xstr && xstr.error() == err_code);

//     //
//     // 3) using exception
//     //
//     struct x
//     {
//         static void thrower_func()
//         {
//             throw std::invalid_argument("encoding error");
//         }
//     };

//     std::string what;

//     try
//     {
//         strf::make_string
//             .facets(strf::to_ascii().on_error(x::thrower_func))
//             ("--\u10000--");
//     }
//     catch(std::invalid_argument& e)
//     {
//         what = e.what();
//     }

//     BOOST_ASSERT(what == "encoding error");
//     //]
// }

// void keeping_surrogates ()
// {
//     //[ keeping_surrogates
//     namespace strf = boost::stringify::v0;
    
//     // from UTF-16 to  Modified UTF-8 (MTF-8), preseving surrogates
//     std::u16string input_utf16 {u"---\0---", 7};
//     input_utf16[1] = 0xD800; // a surrogate character alone
    

//     auto str1 = strf::make_string
//         .facets(strf::to_mtf8().keep_surrogates(false))
//         (input_utf16);

//     auto str2 = strf::make_string
//         .facets(strf::to_mtf8().keep_surrogates(true))
//         (input_utf16);

    
//     BOOST_ASSERT(str1 == u8"-\uFFFE-\xC0\x80---");
//     BOOST_ASSERT(str2 == u8"-\xED\xA0\x80-\xC0\x80---");

//     // now back to UTF-16, preserving the surrogates:
//     auto str3 = strf::make_u16string
//         .facets(strf::to_utf16<char16_t>().keep_surrogates(true))
//         (str2);

//     BOOST_ASSERT(str3 == input_utf16);
//     //]

// }


int main()
{
    input_ouput_different_char_types();
    // mtf8 ();
    // windows_1252();
    // sanitise();
    // error_signal();
    // keeping_surrogates();

    return 0;
}

