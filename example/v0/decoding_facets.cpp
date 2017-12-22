#include <boost/assert.hpp>
#include <boost/stringify.hpp>

void input_ouput_different_char_types()
{
    //[input_ouput_different_char_types
    namespace strf = boost::stringify::v0;

    auto xstr   = strf::make_string()    = {"aaa-", u"bbb-", U"ccc-", L"ddd"};
    auto xstr16 = strf::make_u16string() = {"aaa-", u"bbb-", U"ccc-", L"ddd"};
    auto xstr32 = strf::make_u32string() = {"aaa-", u"bbb-", U"ccc-", L"ddd"};
    auto wxstr  = strf::make_wstring()   = {"aaa-", u"bbb-", U"ccc-", L"ddd"};
    
    BOOST_ASSERT(xstr.value()   ==  "aaa-bbb-ccc-ddd");
    BOOST_ASSERT(xstr16.value() == u"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(xstr32.value() == U"aaa-bbb-ccc-ddd");
    BOOST_ASSERT(wxstr.value()  == L"aaa-bbb-ccc-ddd");
    //]
}

void utf8_decoding_options()
{
    //[utf8_decoding_options
    namespace strf = boost::stringify::v0;

    const char* surr_D800  = "\xED\xA0\x80";
    const char* mtf8_null  = "\xC0\x80";
    const char* overlong_007F = "\xC1\xBF";

    {
        auto str = strf::make_u32string() &= {surr_D800, mtf8_null, overlong_007F};
        BOOST_ASSERT(str == U"\uFFFD\uFFFD\uFFFD");
    }

    {
        auto str = strf::make_u32string
            .with(strf::make_u8decoder().tolerate_overlong())
            &= {surr_D800, mtf8_null, overlong_007F};
    
        BOOST_ASSERT(str[0] == U'\uFFFD');
        BOOST_ASSERT(str[1] == U'\0');
        BOOST_ASSERT(str[2] == U'\u007F');
    }

    {
        auto str = strf::make_u32string
            .with(strf::make_u8decoder().mutf8())
            &= {surr_D800, mtf8_null, overlong_007F};
    
        BOOST_ASSERT(str[0] == U'\uFFFD');
        BOOST_ASSERT(str[1] == U'\0');
        BOOST_ASSERT(str[2] == U'\uFFFD');
    }

    {
        auto str = strf::make_u32string
            .with(strf::make_u8decoder().wtf8())
            &= {surr_D800, mtf8_null, overlong_007F};
    
        BOOST_ASSERT(str[0] == 0xD800);
        BOOST_ASSERT(str[1] == U'\uFFFD');
        BOOST_ASSERT(str[2] == U'\uFFFD');
    }

    {  
        auto str = strf::make_u32string
            .with(strf::make_u8decoder().wtf8().mutf8())
            &= {surr_D800, mtf8_null, overlong_007F};
    
        BOOST_ASSERT(str[0] == 0xD800);
        BOOST_ASSERT(str[1] == U'\0');
        BOOST_ASSERT(str[2] == U'\uFFFD');
    }

    //]
}


int main()
{
    input_ouput_different_char_types();
    utf8_decoding_options();
}
