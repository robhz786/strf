//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

static strf::detail::simple_string_view<char> STRF_TEST_FUNC make_str_0_to_xff(char* str)
{
    for(unsigned i = 0; i < 0x100; ++i)
        str[i] = static_cast<char>(i);
    str[0x100] = '\0';
    return {str, 0x100};
}

template <typename Encoding>
strf::detail::simple_string_view<char> STRF_TEST_FUNC char_0_to_0xff_sanitized(Encoding enc)
{
    static char str[0x100];
    for(unsigned i = 0; i < 0x100; ++i)
    {
        char32_t ch32 = enc.decode_char(static_cast<std::uint8_t>(i));
        unsigned char ch = ( ch32 == (char32_t)-1
                           ? static_cast<unsigned char>('?')
                           : static_cast<unsigned char>(i) );
        str[i] = ch;
    }
    return {str, 0x100};
}

static void STRF_TEST_FUNC remove_fffd
    ( strf::detail::simple_string_view<char32_t> input
    , char32_t* dest ) noexcept
{
    const char32_t* src = input.data();
    const char32_t* src_end = input.end();
    while(src != src_end) {
        if (*src != 0xFFFD) {
            *dest = *src;
            ++dest;
        }
        ++src;
    }
    *dest = 0;
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_TEST_FUNC make_view
    ( const CharT* begin, const CharT* end)
{
    return {begin, end};
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_TEST_FUNC make_view
    ( const CharT* begin, std::size_t size)
{
    return {begin, size};
}

template <typename CharT>
bool STRF_TEST_FUNC operator==
    ( strf::detail::simple_string_view<CharT> str1
    , strf::detail::simple_string_view<CharT> str2 )
{
    if (str1.size() != str2.size())
        return false;

    return strf::detail::str_equal(str1.data(), str2.data(), str1.size());
}

static STRF_TEST_FUNC bool encoding_error_handler_called;

static STRF_TEST_FUNC void encoding_error_handler()
{
    encoding_error_handler_called = true;
}

template <typename Encoding>
void STRF_TEST_FUNC test
    ( Encoding enc
    , strf::detail::simple_string_view<char32_t> decoded_0_to_0xff )
{
    TEST_SCOPE_DESCRIPTION(enc.name());

    static char buff_str_0_to_xff[0x101];
    auto str_0_to_xff = make_str_0_to_xff(buff_str_0_to_xff);

    {
        // to UTF-32
        TEST(decoded_0_to_0xff) (strf::sani(str_0_to_xff, enc));
    }

    char32_t valid_u32input[0x101];
    remove_fffd(decoded_0_to_0xff, valid_u32input);
    char char_buf[0x400];
    {
        // from and back to UTF-32
        auto r = strf::to(char_buf).with(enc) (strf::sani(valid_u32input));
        auto enc_str = make_view(char_buf, r.ptr);

        TEST(valid_u32input) (strf::sani(enc_str, enc));
    }
    {
        // from UTF-8
        char char8_buf[0x400];
        auto r8 = strf::to(char8_buf) (strf::sani(valid_u32input));
        auto u8str =  make_view(char8_buf, r8.ptr);

        auto r = strf::to(char_buf).with(enc) (strf::sani(u8str, strf::utf8<char>()));
        auto enc_str = make_view(char_buf, r.ptr);

        TEST(valid_u32input) (strf::sani(enc_str, enc));

    }

    auto sanitized_0_to_0xff = char_0_to_0xff_sanitized(enc);
    {   // from UTF-8
        char char8_buf[0x400];
        auto r8 = strf::to(char8_buf) (strf::sani(decoded_0_to_0xff));
        auto u8str = make_view(char8_buf, r8.ptr);
        TEST(sanitized_0_to_0xff)
            .with(enc)
            (strf::sani(u8str, strf::utf8<char>()));
    }

    TEST(sanitized_0_to_0xff) .with(enc) (strf::sani(str_0_to_xff));
    TEST("---?+++").with(enc)(strf::sani(u"---\U0010FFFF+++"));


    {
        char buff[10];
        ::encoding_error_handler_called = false;
        strf::to(buff)
            .with(enc, strf::invalid_seq_notifier{encoding_error_handler})
            (strf::sani(u"---\U0010FFFF++"));
        TEST_TRUE(strf::detail::str_equal(buff, "---?++", 6));
        TEST_TRUE(::encoding_error_handler_called);
    }

}

static strf::detail::simple_string_view<char32_t> STRF_TEST_FUNC decoded_0_to_xff_iso_8859_1()
{
    static const char32_t table[0x100] =
        { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
        , 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
        , 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
        , 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
        , 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27
        , 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f
        , 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37
        , 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f
        , 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47
        , 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f
        , 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57
        , 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f
        , 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67
        , 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f
        , 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77
        , 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f
        , 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87
        , 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f
        , 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97
        , 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f
        , 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7
        , 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf
        , 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7
        , 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf
        , 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7
        , 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf
        , 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7
        , 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf
        , 0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7
        , 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef
        , 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7
        , 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff };

    return {table, 0x100};
}

static strf::detail::simple_string_view<char32_t> STRF_TEST_FUNC decoded_0_to_xff_iso_8859_3()
{
    static const char32_t table[0x100] =
        { 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007
        , 0x0008, 0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x000e, 0x000f
        , 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017
        , 0x0018, 0x0019, 0x001a, 0x001b, 0x001c, 0x001d, 0x001e, 0x001f
        , 0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027
        , 0x0028, 0x0029, 0x002a, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f
        , 0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037
        , 0x0038, 0x0039, 0x003a, 0x003b, 0x003c, 0x003d, 0x003e, 0x003f
        , 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047
        , 0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f
        , 0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057
        , 0x0058, 0x0059, 0x005a, 0x005b, 0x005c, 0x005d, 0x005e, 0x005f
        , 0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067
        , 0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f
        , 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077
        , 0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x007e, 0x007f
        , 0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087
        , 0x0088, 0x0089, 0x008a, 0x008b, 0x008c, 0x008d, 0x008e, 0x008f
        , 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097
        , 0x0098, 0x0099, 0x009a, 0x009b, 0x009c, 0x009d, 0x009e, 0x009f
        , 0x00a0, 0x0126, 0x02D8, 0x00A3, 0x00A4, 0xFFFD, 0x0124, 0x00A7
        , 0x00A8, 0x0130, 0x015E, 0x011E, 0x0134, 0x00AD, 0xFFFD, 0x017B
        , 0x00B0, 0x0127, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x0125, 0x00B7
        , 0x00B8, 0x0131, 0x015F, 0x011F, 0x0135, 0x00BD, 0xFFFD, 0x017C
        , 0x00C0, 0x00C1, 0x00C2, 0xFFFD, 0x00C4, 0x010A, 0x0108, 0x00C7
        , 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF
        , 0xFFFD, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x0120, 0x00D6, 0x00D7
        , 0x011C, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x016C, 0x015C, 0x00DF
        , 0x00E0, 0x00E1, 0x00E2, 0xFFFD, 0x00E4, 0x010B, 0x0109, 0x00E7
        , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF
        , 0xFFFD, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x0121, 0x00F6, 0x00F7
        , 0x011D, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x016D, 0x015D, 0x02D9 };

    return {table, 0x100};
}

static strf::detail::simple_string_view<char32_t> STRF_TEST_FUNC decoded_0_to_xff_iso_8859_15()
{
    static const char32_t table[0x100] =
        { 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007
        , 0x0008, 0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x000e, 0x000f
        , 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017
        , 0x0018, 0x0019, 0x001a, 0x001b, 0x001c, 0x001d, 0x001e, 0x001f
        , 0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027
        , 0x0028, 0x0029, 0x002a, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f
        , 0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037
        , 0x0038, 0x0039, 0x003a, 0x003b, 0x003c, 0x003d, 0x003e, 0x003f
        , 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047
        , 0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f
        , 0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057
        , 0x0058, 0x0059, 0x005a, 0x005b, 0x005c, 0x005d, 0x005e, 0x005f
        , 0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067
        , 0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f
        , 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077
        , 0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x007e, 0x007f
        , 0x0080, 0x0081, 0x0082, 0x0083, 0x0084, 0x0085, 0x0086, 0x0087
        , 0x0088, 0x0089, 0x008a, 0x008b, 0x008c, 0x008d, 0x008e, 0x008f
        , 0x0090, 0x0091, 0x0092, 0x0093, 0x0094, 0x0095, 0x0096, 0x0097
        , 0x0098, 0x0099, 0x009a, 0x009b, 0x009c, 0x009d, 0x009e, 0x009f
        , 0x00a0, 0x00a1, 0x00a2, 0x00a3, 0x20ac, 0x00a5, 0x0160, 0x00a7
        , 0x0161, 0x00a9, 0x00aa, 0x00ab, 0x00ac, 0x00ad, 0x00ae, 0x00af
        , 0x00b0, 0x00b1, 0x00b2, 0x00b3, 0x017d, 0x00b5, 0x00b6, 0x00b7
        , 0x017e, 0x00b9, 0x00ba, 0x00bb, 0x0152, 0x0153, 0x0178, 0x00bf
        , 0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00c4, 0x00c5, 0x00c6, 0x00c7
        , 0x00c8, 0x00c9, 0x00ca, 0x00cb, 0x00cc, 0x00cd, 0x00ce, 0x00cf
        , 0x00d0, 0x00d1, 0x00d2, 0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7
        , 0x00d8, 0x00d9, 0x00da, 0x00db, 0x00dc, 0x00dd, 0x00de, 0x00df
        , 0x00e0, 0x00e1, 0x00e2, 0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7
        , 0x00e8, 0x00e9, 0x00ea, 0x00eb, 0x00ec, 0x00ed, 0x00ee, 0x00ef
        , 0x00f0, 0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7
        , 0x00f8, 0x00f9, 0x00fa, 0x00fb, 0x00fc, 0x00fd, 0x00fe, 0x00ff };

    return {table, 0x100};
}

static strf::detail::simple_string_view<char32_t> STRF_TEST_FUNC decoded_0_to_xff_windows_1252()
{
    static const char32_t table[0x100] =
        { 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007
        , 0x0008, 0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x000e, 0x000f
        , 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017
        , 0x0018, 0x0019, 0x001a, 0x001b, 0x001c, 0x001d, 0x001e, 0x001f
        , 0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027
        , 0x0028, 0x0029, 0x002a, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f
        , 0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037
        , 0x0038, 0x0039, 0x003a, 0x003b, 0x003c, 0x003d, 0x003e, 0x003f
        , 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047
        , 0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f
        , 0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057
        , 0x0058, 0x0059, 0x005a, 0x005b, 0x005c, 0x005d, 0x005e, 0x005f
        , 0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067
        , 0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f
        , 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077
        , 0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x007e, 0x007f
        , 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
        , 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F
        , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
        , 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178
        , 0x00a0, 0x00a1, 0x00a2, 0x00a3, 0x00a4, 0x00a5, 0x00a6, 0x00a7
        , 0x00a8, 0x00a9, 0x00aa, 0x00ab, 0x00ac, 0x00ad, 0x00ae, 0x00af
        , 0x00b0, 0x00b1, 0x00b2, 0x00b3, 0x00b4, 0x00b5, 0x00b6, 0x00b7
        , 0x00b8, 0x00b9, 0x00ba, 0x00bb, 0x00bc, 0x00bd, 0x00be, 0x00bf
        , 0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00c4, 0x00c5, 0x00c6, 0x00c7
        , 0x00c8, 0x00c9, 0x00ca, 0x00cb, 0x00cc, 0x00cd, 0x00ce, 0x00cf
        , 0x00d0, 0x00d1, 0x00d2, 0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7
        , 0x00d8, 0x00d9, 0x00da, 0x00db, 0x00dc, 0x00dd, 0x00de, 0x00df
        , 0x00e0, 0x00e1, 0x00e2, 0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7
        , 0x00e8, 0x00e9, 0x00ea, 0x00eb, 0x00ec, 0x00ed, 0x00ee, 0x00ef
        , 0x00f0, 0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7
        , 0x00f8, 0x00f9, 0x00fa, 0x00fb, 0x00fc, 0x00fd, 0x00fe, 0x00ff };

    return {table, 0x100};
}

void STRF_TEST_FUNC test_single_byte_encodings()
{
    test(strf::iso_8859_1<char>(), decoded_0_to_xff_iso_8859_1());
    test(strf::iso_8859_3<char>(), decoded_0_to_xff_iso_8859_3());
    test(strf::iso_8859_15<char>(), decoded_0_to_xff_iso_8859_15());
    test(strf::windows_1252<char>(), decoded_0_to_xff_windows_1252() );
}
