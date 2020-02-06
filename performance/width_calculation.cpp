//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <locale>
#include <fstream>
#include <codecvt>

#include <strf.hpp>
#include "loop_timer.hpp"

class width_accumulator: public strf::underlying_outbuf<4>
{
public:

    width_accumulator(strf::width_t limit)
        : strf::underlying_outbuf<4>(buff_, buff_ + buff_size_)
        , limit_(limit)
    {
    }

    void recycle() override;

    strf::width_t get_result()
    {
        if (limit_ > 0)
        {
            sum_ += wfunc_(limit_, buff_, this->pos());
            this->set_pos(buff_);
        }
        return sum_;
    }

private:

    strf::width_t wfunc_(strf::width_t limit, const char32_t* it, const char32_t* end )
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

    constexpr static std::size_t buff_size_ = 16;
    char32_t buff_[buff_size_];
    const strf::width_t limit_;
    strf::width_t sum_ = 0;
};

void width_accumulator::recycle()
{
    auto p = this->pos();
    this->set_pos(buff_);
    if (this->good())
    {
        sum_ += wfunc_(limit_ - sum_, buff_, p);
        this->set_good(sum_ < limit_);
    }
}


class custom_wcalc
{
public:
    using category = strf::width_calculator_c;

    template <typename Encoding>
    strf::width_t width
        ( const Encoding& enc
        , strf::underlying_outbuf_char_type<Encoding::char_size> ch ) const noexcept
    {
        auto ch32 = enc.decode_single_char(ch);
        return ( ch32 == U'\u2E3A' ? 4
               : ch32 == U'\u2014' ? 2
               : 1 );
    }

    template <typename Encoding>
    constexpr STRF_HD strf::width_t width
        ( const Encoding& enc
        , strf::width_t limit
        , const strf::underlying_outbuf_char_type<Encoding::char_size>* str
        , std::size_t str_len
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) const noexcept
    {
        width_accumulator acc(limit);
        enc.to_u32().transcode(acc, str, str + str_len, enc_err, allow_surr);
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

    const auto print = strf::to(stdout);

    print("UTF-8:\n");

    PRINT_BENCHMARK("strf::to(u8dest) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest) (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len())
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc{}) (strf::fmt(u8str5) > 5)")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc{})
            (strf::fmt(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u8dest) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest) (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len())
            (strf::join_right(5)(u8str5));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc{}) (strf::join_right(5)(u8str5))")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc{})
            (strf::join_right(5)(u8str5));
    }

    print('\n');

    PRINT_BENCHMARK("strf::to(u8dest) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest) (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len{})
            (strf::fmt(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc{}) (strf::fmt(u8str50) > 50)")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc{})
            (strf::fmt(u8str50) > 50);
    }

    PRINT_BENCHMARK("strf::to(u8dest) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest) (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(strf::width_as_u32len{}) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest)
            .with(strf::width_as_u32len())
            (strf::join_right(50)(u8str50));
    }
    PRINT_BENCHMARK("strf::to(u8dest) .with(custom_wcalc{}) (strf::join_right(50)(u8str50))")
    {
        (void)strf::to(u8dest)
            .with(custom_wcalc{})
            (strf::join_right(50)(u8str50));
    }

    print("\nUTF-16:\n");

    PRINT_BENCHMARK("strf::to(u16dest) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest) (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len{})
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::fmt(u16str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::fmt(u16str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest) (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(5)(u16str5));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::join_right(5)(u16str5))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::join_right(5)(u16str5));
    }

    print('\n');
    PRINT_BENCHMARK("strf::to(u16dest) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest) (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len{})
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::fmt(u16str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::fmt(u16str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest) (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(50)(u16str50));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::join_right(50)(u16str50))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::join_right(50)(u16str50));
    }

    print("\nWhen converting UTF-8 to UTF-16:\n");

    PRINT_BENCHMARK("strf::to(u16dest) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest) (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::cv(u8str5) > 5)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::cv(u8str5) > 5);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest) (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(5)(strf::cv(u8str5)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::join_right(5)(strf::cv(u8str5)))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::join_right(5)(strf::cv(u8str5)));
    }

        PRINT_BENCHMARK("strf::to(u16dest) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest) (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::cv(u8str50) > 50)")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::cv(u8str50) > 50);
    }
    PRINT_BENCHMARK("strf::to(u16dest) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest) (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(strf::width_as_u32len{}) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest)
            .with(strf::width_as_u32len())
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    PRINT_BENCHMARK("strf::to(u16dest) .with(custom_wcalc{}) (strf::join_right(50)(strf::cv(u8str50)))")
    {
        (void)strf::to(u16dest)
            .with(custom_wcalc{})
            (strf::join_right(50)(strf::cv(u8str50)));
    }
    return 0;
}
