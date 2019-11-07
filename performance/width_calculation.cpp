//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>
#include <codecvt>

#include <stringify.hpp>
#include "loop_timer.hpp"

class width_accumulator: public strf::basic_outbuf<char32_t>
{
public:

    width_accumulator(strf::width_t limit)
        : strf::basic_outbuf<char32_t>(_buff, _buff + _buff_size)
        , _limit(limit)
    {
    }

    void recycle() override;

    strf::width_t get_result()
    {
        if (_limit > 0)
        {
            _sum += _wfunc(_limit, _buff, this->pos());
            this->set_pos(_buff);
        }
        return _sum;
    }

private:

    strf::width_t _wfunc(strf::width_t limit, const char32_t* it, const char32_t* end )
    {
        strf::width_t w = 0;
        for (; w < limit && it != end; ++it)
        {
            auto ch = *it;
            w += ( ch == U'\u2E3A' ? 4
                 : ch == U'\u2014' ? 2
                 : 1 );
        }
        return w;
    }

    constexpr static std::size_t _buff_size = 16;
    char32_t _buff[_buff_size];
    const strf::width_t _limit;
    strf::width_t _sum = 0;
};

void width_accumulator::recycle()
{
    auto p = this->pos();
    this->set_pos(_buff);
    if (this->good())
    {
        _sum += _wfunc(_limit - _sum, _buff, p);
        this->set_good(_sum < _limit);
    }
}

template <typename CharT>
class custom_wcalc: public strf::width_calculator<CharT>
{
public:

    strf::width_t width_of(CharT ch, strf::encoding<CharT> enc) const override
    {
        auto ch32 = enc.decode_single_char(ch);
        return ( ch32 == U'\u2E3A' ? 4
               : ch32 == U'\u2014' ? 2
               : 1 );
    }

    strf::width_t width( strf::width_t limit
                       , const CharT* str
                       , std::size_t str_len
                       , strf::encoding<CharT> enc
                       , strf::encoding_error enc_err
                       , strf::surrogate_policy allow_surr ) const override
    {
        width_accumulator acc(limit);
        enc.to_u32().transcode( acc, str, str + str_len
                                , enc_err, allow_surr );
        return acc.get_result();
    }
};

int main()
{
    char u8dest[100000];
    char16_t u16dest[100000];

    const std::string u8str5 {5, 'x'};
    const std::string u8str50 {50, 'x'};
    const std::u16string u16str5 {5, u'x'};
    const std::u16string u16str50 {50, u'x'};

    const auto print = strf::write(stdout);

    print("UTF-8:\n");

    PRINT_BENCHMARK("strf::write(u8dest) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::write(u8dest) (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as_u32len<char>{}) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as_u32len<char>())
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(custom_wcalc<char>{}) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::write(u8dest)
            .facets(custom_wcalc<char>{})
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u8dest) (strf::join_right(5)(u8str5))")
    {
        (void)strf::write(u8dest) (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as_u32len<char>{}) (strf::join_right(5)(u8str5))")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as_u32len<char>())
            (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(custom_wcalc<char>{}) (strf::join_right(5)(u8str5))")
    {
        (void)strf::write(u8dest)
            .facets(custom_wcalc<char>{})
            (strf::join_right(5)(u8str5));
    }

    print('\n');

    PRINT_BENCHMARK("strf::write(u8dest) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::write(u8dest) (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as_u32len<char>{}) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as_u32len<char>{})
            (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(custom_wcalc<char>{}) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::write(u8dest)
            .facets(custom_wcalc<char>{})
            (strf::fmt(u8str50) > 50);
    }

    PRINT_BENCHMARK("strf::write(u8dest) (strf::join_right(50)(u8str50))")
    {
        (void)strf::write(u8dest) (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(strf::width_as_u32len<char>{}) (strf::join_right(50)(u8str50))")
    {
        (void)strf::write(u8dest)
            .facets(strf::width_as_u32len<char>())
            (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::write(u8dest) .facets(custom_wcalc<char>{}) (strf::join_right(50)(u8str50))")
    {
        (void)strf::write(u8dest)
            .facets(custom_wcalc<char>{})
            (strf::join_right(50)(u8str50));
    }

    print("\nUTF-16:\n");

    PRINT_BENCHMARK("strf::write(u16dest) (strf::fmt_cv(u16str5) > 5)")
    {
        (void)strf::write(u16dest) (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char16_t>{}) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char16_t>{})
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char16_t>{}) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char16_t>{})
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) (strf::join_right(5)(u16str5))")
    {
        (void)strf::write(u16dest) (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char16_t>{}) (strf::join_right(5)(u16str5))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char16_t>())
            (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char16_t>{}) (strf::join_right(5)(u16str5))")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char16_t>{})
            (strf::join_right(5)(u16str5));
    }

    print('\n');
    PRINT_BENCHMARK("strf::write(u16dest) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::write(u16dest) (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char16_t>{}) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char16_t>{})
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char16_t>{}) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char16_t>{})
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u16dest) (strf::join_right(50)(u16str50))")
    {
        (void)strf::write(u16dest) (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char16_t>{}) (strf::join_right(50)(u16str50))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char16_t>())
            (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char16_t>{}) (strf::join_right(50)(u16str50))")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char16_t>{})
            (strf::join_right(50)(u16str50));
    }

    print("\nWhen converting UTF-8 to UTF-16:\n");

    PRINT_BENCHMARK("strf::write(u16dest) (strf::fmt_cv(u8str5) > 5)")
    {
        (void)strf::write(u16dest) (strf::fmt_cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char>{}) (strf::fmt_cv(u8str5) > 5)")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char>())
            (strf::fmt_cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char>{}) (strf::fmt_cv(u8str5) > 5)")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char>{})
            (strf::fmt_cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::write(u16dest) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::write(u16dest) (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char>{}) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char>())
            (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char>{}) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char>{})
            (strf::join_right(5)(strf::cv(u8str5)));
    }

        PRINT_BENCHMARK("strf::write(u16dest) (strf::fmt_cv(u8str50) > 50)")
    {
        (void)strf::write(u16dest) (strf::fmt_cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char>{}) (strf::fmt_cv(u8str50) > 50)")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char>())
            (strf::fmt_cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char>{}) (strf::fmt_cv(u8str50) > 50)")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char>{})
            (strf::fmt_cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::write(u16dest) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::write(u16dest) (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(strf::width_as_u32len<char>{}) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::write(u16dest)
            .facets(strf::width_as_u32len<char>())
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::write(u16dest) .facets(custom_wcalc<char>{}) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::write(u16dest)
            .facets(custom_wcalc<char>{})
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    return 0;
}
