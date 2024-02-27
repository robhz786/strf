# Strf

[![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=main&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/main)
[![codecov](https://codecov.io/gh/robhz786/strf/branch/main/graph/badge.svg?token=d5DIZzYv5O)](https://codecov.io/gh/robhz786/strf)

A C++ text formatting and transcoding library.

Header-only by default, but can be used as a static library as well ( see [details](http://robhz786.github.io/strf/v0.15.3/install.html) ).

## Documentation


* [Learn by examples](#code-samples) ( if you just wanna have an glance )
* [Tutorial](http://robhz786.github.io/strf/v0.15.3/tutorial.html) ( good starting point, if you actually want learn about it )
* [Quick reference](http://robhz786.github.io/strf/v0.15.3/quick_reference.html) ( if you are already using it )
* Full Header references ( if you need to get into the details )
  * [`<strf/destination.hpp>`](http://robhz786.github.io/strf/v0.15.3/destination_hpp.html) is a lightweight and freestanding header that defines the `destination` class template. All other headers depend on this one.
  * [`<strf.hpp>`](http://robhz786.github.io/strf/v0.15.3/strf_hpp.html) is the main header. It can be used in a freestanding environment, whereas the headers bellow can not.
  * [`<strf/iterator.hpp>`](http://robhz786.github.io/strf/v0.15.3/iterator_hpp.html) defines an output iterator adapter for the `destination` class template.
  * [`<strf/to_string.hpp>`](http://robhz786.github.io/strf/v0.15.3/to_string_hpp.html) adds support for writting to `std::basic_string`. It includes `<strf.hpp>`.
  * [`<strf/to_cfile.hpp>`](http://robhz786.github.io/strf/v0.15.3/to_cfile_hpp.html) adds support for writting to `FILE*`. It includes `<strf.hpp>`.
  * [`<strf/to_streambuf.hpp>`](http://robhz786.github.io/strf/v0.15.3/to_streambuf_hpp.html) adds support for writting to `std::basic_streambuf`. It includes `<strf.hpp>`.
  * `strf/all.hpp` include all of the headers above.
* HOWTOs
  * [Integrate strf in your environment](http://robhz786.github.io/strf/v0.15.3/install.html)
  * Extending:
    * [Adding a destination](http://robhz786.github.io/strf/v0.15.3/howto_add_destination.html)
    * [Adding a printable type](http://robhz786.github.io/strf/v0.15.3/howto_add_printable_types.html)
    * [Overriding a printable type](http://robhz786.github.io/strf/v0.15.3/howto_override_printable_types.html)
  * [Using strf on CUDA devices (experimental) ](http://robhz786.github.io/strf/v0.15.3/cuda.html)

* [Benchmarks](http://robhz786.github.io/strf-benchmarks/v0.15.3/results.html) ( versus {fmt} )

## Requirements

From version 0.16 onwards, C++11 is no longer supported. The compiler must support C++14 or above.

Strf has been tested in the following compilers:

* Clang 6.0
* GCC 6.5.0
* Visual Studio 2019 16
* NVCC 11.8

## Code samples

- [Basic examples](#basic-examples)
- [Alignment formatting](#alignment-formatting)
- [Single character argument](#single-character-argument)
- [Numeric Formatting](#numeric-formatting)
- [Numeric Punctuation](#numeric-punctuation)
- [Transcoding](#transcoding)
- [Joining arguments](#joining-arguments)
- [Ranges](#ranges)

### Basic examples

```c++
#include <strf/to_string.hpp>
#include <assert>

constexpr int x = 255;

void basic_samples()
{
    // Creating a std::string
    auto str = strf::to_string(x, " in hexadecimal is ", *strf::hex(x), '.');
    assert(str == "255 in hexadecimal is 0xff.");

    // Alternative syntax to enable i18n
    auto str_tr = strf::to_string.tr("{} in hexadecimal is {}.", x, *strf::hex(x));
    assert(str_tr == str);

    // Applying a facet
    auto to_string_mc = strf::to_string.with(strf::mixedcase);
    auto str_mc = to_string_mc(x, " in hexadecimal is ", *strf::hex(x), '.');
    assert(str_mc == "255 in hexadecimal is 0xFF.");

    // Achieving the same result, but in multiple steps:
    strf::string_maker str_maker;
    strf::to(str_maker) (x, " in hexadecimal is ");
    strf::to(str_maker).with(strf::mixedcase) (*strf::hex(x), '.');
    auto str_mc_2 = str_maker.finish();
    assert(str_mc_2 == str_mc);

    // Writing instead to char*
    char buff[200];
    strf::to(buff, sizeof(buff)) (x, " in hexadecimal is ", *strf::hex(x), '.');
    assert(str == buff);
}

```

### Alignment formatting

```c++
#include <strf/to_string.hpp>
#include <assert>

void alignment_formatting()
{
    using namespace strf::format_functions;

    assert(strf::to_string(dec(123) > 8)  == "     123");
    assert(strf::to_string(dec(123) < 8)  == "123     ");
    assert(strf::to_string(dec(123) ^ 8)  == "  123   ");

    assert(strf::to_string(right(123, 8))        == "     123");
    assert(strf::to_string(right(123, 8,  '~'))  == "~~~~~123");
    assert(strf::to_string(left(123, 8,  '~'))   == "123~~~~~");
    assert(strf::to_string(center(123, 8,  '~')) == "~~123~~~");

    assert(strf::to_u8string(right(123, 8, U'‥')) == u8"‥‥‥‥‥123");
}
```
**Note**: Strf calculates string width the same way as
[specified to `std::format` in C++20](https://timsong-cpp.github.io/cppwp/n4868/format#string.std-11), which takes into account grapheme clustering.

### Single character argument

```c++
#include <strf/to_string.hpp>
#include <assert>

void single_character_argument()
{
    using namespace strf::format_functions; // strf::multi, strf::right

    assert(strf::to_string('a') == "a");
    assert(strf::to_string(multi('a', 5)) == "aaaaa");
    assert(strf::to_string(multi('a', 5) > 8) == "   aaaaa");
    assert(strf::to_string(right('a', 8, '.').multi(5)) == "...aaaaa");

    // char32_t is implicitly transcoded
    assert(strf::to_string(U'a') == "a");
    assert(strf::to_u8string(U'\u2022') == u8"•");
    assert(strf::to_u8string(multi(U'\u2022', 5) > 8) == u8"   •••••");

    // strf::to_string(L'a') // doesn't compile: it must be char32_t or
                             // the same as the destination character type
}
```

### Numeric Formatting

```c++
#include <strf/to_string.hpp>
#include <numbers> // C++20, defines std::numbers::pi
#include <assert>

void numeric_formatting()
{
    using namespace strf::format_functions;

    //  In printf  |       In Strf
    //-------------+-----------------------
    //         Integers
    //-------------+-----------------------
    //      'd'    |   .dec() or dec(value)
    //      'o'    |   .oct() or oct(value)
    //      'x'    |   .hex() or hex(value)
    //             |   .bin() or bin(value)
    //-------------+-----------------------
    //      Floating Points
    //-------------+-----------------------
    //      'x'    |   hex(value).p(6) or   hex(value, 6)
    //      'f'    | fixed(value).p(6) or fixed(value, 6)
    //    'e' 'E'  |   sci(value).p(6) or   sci(value, 6)
    //      'g'    |   gen(value).p(6) or   gen(value, 6)
    //-------------+-----------------------
    //   Integers and Floating Points
    //-------------+-----------------------
    //      '+'    |    .operator+()
    //      '#'    |    .operator*()
    //      ' '    |    .operator~()
    //  .precision |    .p(precision), or as the second argument of
    //             |                 hex(), fixed(), sci() or gen()
    //      '0'    |    .pad0(width) or pad0(value, width)

    assert(strf::to_string(  fmt(1.)) == "1");
    assert(strf::to_string( *fmt(1.)) == "1.");
    assert(strf::to_string( +fmt(1.)) == "+1");
    assert(strf::to_string(+*fmt(1.)) == "+1.");
    assert(strf::to_string( ~fmt(1.5)) == " 1.5");
    assert(strf::to_string( ~left(1.5, 8, '*')) == "*1.5****");

    assert(strf::to_string(sci(1.0))     == "1e+00");
    assert(strf::to_string(fixed(1e+10)) == "10000000000");

    //----------------------------------------------------------------
    // Strf prints enough digits for exact recovery on parsing
    // regardless of notation
    using std::numbers::pi;
    assert(strf::to_string(fixed(pi)) == "3.141592653589793");

    // , unless a precision is specified
    assert(strf::to_string(fixed(pi).p(4)) == "3.1416");
    assert(strf::to_string(fixed(pi, 4))   == "3.1416");
    assert(strf::to_string(gen(pi, 4))     == "3.142");

    // ---------------------------------------------------------------
    // pad0()
    assert(strf::to_string(+*pad0(1., 5))     == "+001.");
    assert(strf::to_string(+*pad0(1., 5) > 8) == "   +001.");

    // For nan and inf, pad0 sets the minimum alignment width:
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    assert(strf::to_string(pad0(nan, 8))              == "     nan");
    assert(strf::to_string(left(nan, 4, '*').pad0(8)) == "nan*****");
    assert(strf::to_string(left(nan, 8, '*').pad0(4)) == "nan*****");

    //----------------------------------------------------------------
    // The strf::lettercase facet
    constexpr auto to_string_upper = strf::to_string.with(strf::uppercase);
    constexpr auto to_string_lower = strf::to_string.with(strf::lowercase); // default
    constexpr auto to_string_mixed = strf::to_string.with(strf::mixedcase);

    assert(to_string_upper(*hex(0xaa), " ", sci(1.)) == "0XAA 1E+00");
    assert(to_string_lower(*hex(0xaa), " ", sci(1.)) == "0xaa 1e+00");
    assert(to_string_mixed(*hex(0xaa), " ", sci(1.)) == "0xAA 1e+00");
}
```

### Numeric Punctuation

```c++
#include <strf/to_string.hpp>
#include <strf/locale.hpp> // strf::locale_numpunct

void numeric_punctuation()
{
    // German punctuation
    auto s = strf::to_string.with(strf::numpunct_de_DE) (!strf::fixed(1000000.5));
    assert(s == "1.000.000,5");

    // You have to use format function `strf::punct` or `.punct()`
    // or operator! to apply punctuation
    s = strf::to_string.with(strf::numpunct_de_DE)
        ( !strf::fixed(1000000.5), "  ", strf::fixed(1000000.5)
        , "  ", strf::punct(10000), "  ", 100000 );
    assert(s == "1.000.000,5  1000000.5  10.000  100000");

    // Use strf::locale_numpunct to get the
    // punctuation of current global locale
    try {
        const std::locale loc("en_US.UTF-8");
        const auto previous_loc = std::locale::global(loc);
        const auto loc_punct = strf::locale_numpunct(); // from <strf/locale.hpp>
        std::locale::global(previous_loc); // this does not affect loc_punt

        const auto result = strf::to_string.with(loc_punct) (*!strf::fixed(1000000.5));
        assert(result == "1,000,000.5");
    } catch (std::runtime_error&) {
        // locale not supported
    }

    // Manually specifing a punctuation
    constexpr auto my_fancy_punct = strf::numpunct<10>(3)
        .thousands_sep(U'•')
        .decimal_point(U'⎖'); // U+2396
    auto s2 = strf::to_u8string.with(my_fancy_punct) (!strf::fixed(1000000.5));
    assert(s2 == u8"1•000•000⎖5");

    // With variable grouping
    constexpr auto my_fancy_punct_2 = strf::numpunct<10>(3, 2, 1)
        .thousands_sep(U'•')
        .decimal_point(U'⎖');
    auto s3 = strf::to_u8string.with(my_fancy_punct_2)
        (strf::punct(10000000.125));
    assert(s3 == u8"1•0•0•00•000⎖125");

    // Non-decimal bases
    constexpr auto my_hex_punct = strf::numpunct<16>(4).thousands_sep('\'');
    auto s4 = strf::to_string.with(my_hex_punct)(!strf::hex(0xFFFFFFF));
    assert(s4 == "fff'ffff");
}
```

### Transcoding

```c++
#include <strf/to_string.hpp>

void transcoding()
{
    // Converting UTF-16 to UTF-8
    auto str_narrow = strf::to_string("He was born in ", strf::transcode(u"خنيفرة"), '.');
    assert(str_narrow == "He was born in \xd8\xae\xd9\x86\xd9\x8a\xd9\x81\xd8\xb1\xd8\xa9.");

    auto str_u8 = strf::to_u8string(u8"He was born in ", strf::transcode(u"خنيفرة"), u8'.');
    assert(str_u8 == u8"He was born in خنيفرة.");


    // Converting UTF-8 to UTF-16
    assert(strf::to_u16string(strf::transcode(str_narrow)) == u"He was born in خنيفرة.");
    assert(strf::to_u16string(strf::transcode(str_u8)) == u"He was born in خنيفرة.");


    // Converting UTF-16 to ISO-8859-6
    auto str_narrow_2 = strf::to_string.with(strf::iso_8859_6<char>)
        ( "He was born in ", strf::transcode(u"خنيفرة"), '.');
    assert(str_narrow_2 == "He was born in \xce\xe6\xea\xe1\xd1\xc9.");


    // Converting char8_t string to ISO-8859-6
    auto str_narrow_3 = strf::to_string.with(strf::iso_8859_6<char>)
        ( "He was born in ", strf::transcode(u8"خنيفرة"), '.');
    assert(str_narrow_3 == str_narrow_2);


    // Converting UTF-8 to ISO-8859-6
    // ( if the source character type is the same as the destination
    //   character type, the charset must be specified inside strf::transcode() )
    auto str_narrow_4 = strf::to_string.with(strf::iso_8859_6<char>)
        ( "He was born in "
        , strf::transcode( "\xd8\xae\xd9\x86\xd9\x8a\xd9\x81\xd8\xb1\xd8\xa9"
                         , strf::utf8<char> )
        , '.' );
    assert(str_narrow_4 == str_narrow_2);


    // Converting ISO-8859-6 to UTF-16
    auto str_u16 = strf::to_u16string
        ( u"He was born in "
        , strf::transcode("\xce\xe6\xea\xe1\xd1\xc9", strf::iso_8859_6<char>)
        , u'.' );
    assert(str_u16 == u"He was born in خنيفرة.");

    // or
    str_u16 = strf::to_u16string.with(strf::iso_8859_6<char>)
        ( u"He was born in "
        , strf::transcode("\xce\xe6\xea\xe1\xd1\xc9")
        , u'.' );
    assert(str_u16 == u"He was born in خنيفرة.");


    // Several transcodings in a single statemet
    auto str = strf::to_u8string( strf::transcode(u"aaa--")
                                , strf::transcode(U"bbb--")
                                , strf::transcode( "\x80\xA4"
                                                 , strf::windows_1252<char> ) );
    assert(str == u8"aaa--bbb--\u20AC\u00A4");
}
```

### Joining arguments

```c++
#include <strf/to_string.hpp>

void joins()
{
    // strf::join makes a sequence of arguments to be treated as one
    const char* dirname = "/tmp";
    const char* filename = "tmp1234";
    auto s = strf::to_string.tr( "Could not open file {}"
                               , strf::join(dirname, '/', filename) );
    assert(s == "Could not open file /tmp/tmp1234");

    // join_left, join_center and join_right align several arguments as one
    const int value = 255;
    s = strf::to_string
        ( strf::join_center(40, U'.')
            ( value, " in hexadecimal is ", strf::hex(value) ) );

    assert(s == "........255 in hexadecimal is ff........");


    // You can also use operators < > ^ to a align a join
    auto j = strf::join(value, " in hexadecimal is ", strf::hex(value));
    s = strf::to_string(j ^ 40);
    assert(s == "        255 in hexadecimal is ff        ");
}
```

### Ranges

```c++
#include <strf/to_string.hpp>

void ranges()
{
    int array[] = {10, 20, 30, 40};

    // Basic
    auto str = strf::to_string( "--[", strf::range(array), "]--");
    assert(str == "--[10203040]--");


    // With separator
    str = strf::to_string( "--[", strf::separated_range(array, " / "), "]--");
    assert(str == "--[10 / 20 / 30 / 40]--");


    // With separator and formatting
    str = strf::to_string( "--["
                         , *strf::fmt_separated_range(array, " / ").hex().p(4)
                         , "]--");
    assert(str == "--[0x000a / 0x0014 / 0x001e / 0x0028]--");


    // Transforming the elements
    auto func = [](int x){ return strf::join('<', strf::pad0(x, 4), '>'); };

    str = strf::to_string(strf::range(array, func));
    assert(str == "<0010><0020><0030><0040>");

    str = strf::to_string(strf::separated_range(array, " ", func));
    assert(str == "<0010> <0020> <0030> <0040>");
}
```