#ifndef STRF_DETAIL_TR_PRINTER_HPP
#define STRF_DETAIL_TR_PRINTER_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printer.hpp>
#include <strf/detail/preprinting.hpp>
#include <strf/detail/standard_lib_functions.hpp>

namespace strf {

struct tr_error_notifier_c;

struct default_tr_error_notifier
{
    using category = strf::tr_error_notifier_c;

    template <typename Charset>
    inline STRF_HD void handle
        ( const typename Charset::code_unit* str
        , std::ptrdiff_t str_len
        , Charset charset
        , std::ptrdiff_t err_pos ) noexcept
    {
        (void) str;
        (void) str_len;
        (void) err_pos;
        (void) charset;
    }
};

struct tr_error_notifier_c {
    static constexpr STRF_HD strf::default_tr_error_notifier get_default() noexcept
    {
        return strf::default_tr_error_notifier{};
    }
};

namespace detail {

template <typename CharT>
struct read_uint_result
{
    std::ptrdiff_t value;
    const CharT* it;
};

template <typename CharT>
STRF_HD read_uint_result<CharT> read_uint(const CharT* it, const CharT* end, std::ptrdiff_t limit) noexcept
{
    std::ptrdiff_t value = *it -  static_cast<CharT>('0');
    ++it;
    while (it < end) {
        CharT ch = *it;
        if (ch < static_cast<CharT>('0') || static_cast<CharT>('9') < ch) {
            break;
        }
        value *= 10;
        value += ch - static_cast<CharT>('0');
        if(value >= limit) {
            value = limit + 1;
            break;
        }
        ++it;
    }
    return {value, it};
}

template <typename CharT>
STRF_HD inline std::ptrdiff_t tr_string_size
    ( const strf::preprinting<strf::precalc_size::no, strf::precalc_width::no>*
    , std::ptrdiff_t
    , const CharT*
    , const CharT*
    , std::ptrdiff_t ) noexcept
{
    return 0;
}

template <typename CharT, strf::precalc_size PreSize, strf::precalc_width PreWidth>
struct tr_preprinting;

struct eval_to_false_t {
    constexpr STRF_HD operator bool() const { return false; };
};

template <typename CharT>
class tr_pre_size
{
    const std::ptrdiff_t* size_array_;
    std::ptrdiff_t array_size_;
    int replacement_char_size_;
    std::ptrdiff_t size_ = 0;

public:
    constexpr STRF_HD tr_pre_size
        ( const std::ptrdiff_t* size_array
        , std::ptrdiff_t array_size
        , int replacement_char_size )
        : size_array_(size_array)
        , array_size_(array_size)
        , replacement_char_size_(replacement_char_size)
    {
    }

    STRF_HD std::ptrdiff_t accumulated_ssize() const
    {
        return size_;
    }

    STRF_HD eval_to_false_t account_arg(std::ptrdiff_t index)
    {
        if (index < array_size_) {
            size_ += size_array_[index];
        } else {
            size_ += replacement_char_size_;
        }
        return {};
    }

    STRF_HD eval_to_false_t account_string(const CharT* begin, const CharT* end)
    {
        size_ += (end - begin);
        return {};
    }
    constexpr STRF_HD std::ptrdiff_t num_args() const
    {
        return array_size_;
    }
};

template <typename CharT, typename Charset, typename WidthCalculator>
class tr_pre_width
{
    const strf::width_t* width_array_;
    std::ptrdiff_t array_size_;
    strf::width_t remaining_width_;
    WidthCalculator wcalc_;
    Charset charset_;

    STRF_HD STRF_CONSTEXPR_IN_CXX14 bool subtract_(strf::width_t w)
    {
        if (w < remaining_width_) {
            remaining_width_ -= w;
            return false;
        }
        remaining_width_ = 0;
        return true;
    }
public:

    constexpr STRF_HD tr_pre_width
        ( const strf::width_t* width_array
        , std::ptrdiff_t width_array_size
        , strf::width_t width
        , WidthCalculator wcalc
        , Charset charset )
        : width_array_(width_array)
        , array_size_(width_array_size)
        , remaining_width_(width)
        , wcalc_(wcalc)
        , charset_(charset)
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool account_arg(std::ptrdiff_t index)
    {
        return subtract_(index < array_size_ ? width_array_[index] : strf::width_t(1));
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool account_string(const CharT* begin, const CharT* end)
    {
        const auto str_width = wcalc_.str_width
            (charset_, remaining_width_, begin, end);
        return subtract_(str_width);
    }
    constexpr STRF_HD strf::width_t remaining_width() const
    {
        return remaining_width_ > 0 ? remaining_width_ : strf::width_t(0);
    }
    constexpr STRF_HD std::ptrdiff_t num_args() const
    {
        return array_size_;
    }
};



template <typename CharT, typename Charset, typename WidthCalculator>
class tr_pre_size_and_width
{
    const std::ptrdiff_t* size_array_;
    const strf::width_t* width_array_;
    std::ptrdiff_t array_size_;
    std::ptrdiff_t size_ = 0;
    strf::width_t remaining_width_;
    WidthCalculator wcalc_;
    Charset charset_;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void subtract_width_(strf::width_t w)
    {
        if (w <= remaining_width_) {
            remaining_width_ -= w;
        } else {
            remaining_width_ = 0;
        }
    }

public:
    constexpr STRF_HD tr_pre_size_and_width
        ( const std::ptrdiff_t* size_array
        , const strf::width_t* width_array
        , std::ptrdiff_t array_size
        , strf::width_t width
        , WidthCalculator wcalc
        , Charset charset )
        : size_array_(size_array)
        , width_array_(width_array)
        , array_size_(array_size)
        , remaining_width_(width)
        , wcalc_(wcalc)
        , charset_(charset)
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD eval_to_false_t account_arg(std::ptrdiff_t index)
    {
        if (index < array_size_) {
            size_ += size_array_[index];
            subtract_width_(width_array_[index]);
        } else {
            size_ += charset_.replacement_char_size();
            subtract_width_(1);
        }
        return {};
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD eval_to_false_t account_string
        ( const CharT* begin, const CharT* end )
    {
        size_ += (end - begin);
        auto w = wcalc_.str_width(charset_, remaining_width_, begin, end);
        subtract_width_(w);
        return {};
    }
    STRF_HD std::ptrdiff_t accumulated_ssize() const
    {
        return size_;
    }
    constexpr STRF_HD strf::width_t remaining_width() const
    {
        return remaining_width_;
    }
    constexpr STRF_HD std::ptrdiff_t num_args() const
    {
        return array_size_;
    }
};


template <typename CharT, typename TrPre>
STRF_HD void tr_do_preprinting
    ( TrPre & pre
    , const CharT* it
    , const CharT* end )
{
    std::ptrdiff_t arg_idx = 0;
    while (it < end) {
        const CharT* prev = it;
        it = strf::detail::str_find<CharT>(it, static_cast<std::size_t>(end - it), '{');
        if (it == nullptr) {
            pre.account_string(prev, end);
            return;
        }
        if (pre.account_string(prev, it)) {
            return;
        }
        ++it;

        after_the_opening_brace:
        if (it == end) {
            pre.account_arg(arg_idx);
            return;
        }

        auto ch = *it;
        if (ch == '}') {
            if (static_cast<bool>(pre.account_arg(arg_idx))) {
                return;
            }
            ++arg_idx;
            ++it;
        } else if (CharT('0') <= ch && ch <= CharT('9')) {
            auto result = strf::detail::read_uint(it, end, pre.num_args());
            if (static_cast<bool>(pre.account_arg(result.value))) {
                return;
            }
            it = strf::detail::str_find<CharT>(result.it, static_cast<std::size_t>(end - result.it), '}');
            if (it == nullptr) {
                return;
            }
            ++it;
        } else if(ch == '{') {
            const auto *it2 = it + 1;
            it2 = strf::detail::str_find<CharT>(it2, static_cast<std::size_t>(end - it2), '{');
            if (it2 == nullptr) {
                pre.account_string(it, end);
                return;
            }
            if (pre.account_string(it, it2)) {
                return;
            }
            it = it2 + 1;
            goto after_the_opening_brace;
        } else {
            if (ch != '-') {
                if (static_cast<bool>(pre.account_arg(arg_idx))) {
                    return;
                }
                ++arg_idx;
            }
            const auto *it2 = it + 1;
            it = strf::detail::str_find<CharT>(it2, static_cast<std::size_t>(end - it2), '}');
            if (it == nullptr) {
                return;
            }
            ++it;
        }
    }
}

template <typename CharT>
STRF_HD std::ptrdiff_t tr_string_size
    ( const strf::preprinting<strf::precalc_size::yes, strf::precalc_width::no>* args_pre
    , std::ptrdiff_t num_args
    , const CharT* it
    , const CharT* end
    , std::ptrdiff_t inv_arg_size ) noexcept
{
    std::ptrdiff_t count = 0;
    std::ptrdiff_t arg_idx = 0;

    while (it < end) {
        const CharT* prev = it;
        it = strf::detail::str_find<CharT>(it, static_cast<std::size_t>(end - it), '{');
        if (it == nullptr) {
            count += (end - prev);
            break;
        }
        count += (it - prev);
        ++it;

        after_the_opening_brace:
        if (it == end) {
            if (arg_idx < num_args) {
                count += args_pre[arg_idx].accumulated_ssize();
            } else {
                count += inv_arg_size;
            }
            break;
        }

        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                count += args_pre[arg_idx].accumulated_ssize();
                ++arg_idx;
            } else {
                count += inv_arg_size;
            }
            ++it;
        } else if (CharT('0') <= ch && ch <= CharT('9')) {
            auto result = strf::detail::read_uint(it, end, num_args);

            if (result.value < num_args) {
                count += args_pre[result.value].accumulated_ssize();
            } else {
                count += inv_arg_size;
            }
            it = strf::detail::str_find<CharT>(result.it, static_cast<std::size_t>(end - result.it), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            const auto *it2 = it + 1;
            it2 = strf::detail::str_find<CharT>(it2, static_cast<std::size_t>(end - it2), '{');
            if (it2 == nullptr) {
                return count += end - it;
            }
            count += (it2 - it);
            it = it2 + 1;
            goto after_the_opening_brace;
        } else {
            if (ch != '-') {
                if (arg_idx < num_args) {
                    count += args_pre[arg_idx].accumulated_ssize();
                    ++arg_idx;
                } else {
                    count += inv_arg_size;
                }
            }
            const auto *it2 = it + 1;
            it = strf::detail::str_find<CharT>(it2, static_cast<std::size_t>(end - it2), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        }
    }
    return count;
}

template <typename Charset, typename ErrHandler>
STRF_HD void tr_string_write
    ( const typename Charset::code_unit* str
    , const typename Charset::code_unit* str_end
    , const strf::printer<typename Charset::code_unit>* const * args
    , std::ptrdiff_t num_args
    , strf::destination<typename Charset::code_unit>& dst
    , Charset charset
    , ErrHandler err_handler )
{
    std::ptrdiff_t arg_idx = 0;
    using char_type = typename Charset::code_unit;

    auto it = str;
    const std::ptrdiff_t str_len = str_end - str;
    while (it < str_end) {
        const char_type* prev = it;
        it = strf::detail::str_find<char_type>(it, static_cast<std::size_t>(str_end - it), '{');
        if (it == nullptr) {
            dst.write(prev, str_end - prev);
            return;
        }
        dst.write(prev, it - prev);
        ++it;
        after_the_opening_brace:
        if (it == str_end) {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(dst);
            } else {
                charset.write_replacement_char(dst);
                err_handler.handle(str, str_len, charset, (it - str) - 1);
            }
            break;
        }
        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(dst);
                ++arg_idx;
            } else {
                charset.write_replacement_char(dst);
                err_handler.handle(str, str_len, charset, (it - str) - 1);
            }
            ++it;
        } else if (char_type('0') <= ch && ch <= char_type('9')) {
            auto result = strf::detail::read_uint(it, str_end, num_args);
            if (result.value < num_args) {
                args[result.value]->print_to(dst);
            } else {
                charset.write_replacement_char(dst);
                err_handler.handle(str, str_len, charset, (it - str) - 1);
            }
            it = strf::detail::str_find<char_type>(result.it, static_cast<std::size_t>(str_end - result.it), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            auto it2 = it + 1;
            it2 = strf::detail::str_find<char_type>(it2, static_cast<std::size_t>(str_end - it2), '{');
            if (it2 == nullptr) {
                dst.write(it, str_end - it);
                return;
            }
            dst.write(it, (it2 - it));
            it = it2 + 1;
            goto after_the_opening_brace;
        } else {
            if (ch != '-') {
                if (arg_idx < num_args) {
                    args[arg_idx]->print_to(dst);
                    ++arg_idx;
                } else {
                    charset.write_replacement_char(dst);
                    err_handler.handle(str, str_len, charset, (it - str) - 1);
                }
            }
            auto it2 = it + 1;
            it = strf::detail::str_find<char_type>(it2, static_cast<std::size_t>(str_end - it2), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        }
    }
}

template <typename Charset, typename ErrHandler>
class tr_string_printer
{
    using char_type = typename Charset::code_unit;
public:

    template <strf::precalc_size SizeRequested>
    STRF_HD tr_string_printer
        ( strf::preprinting<SizeRequested, strf::precalc_width::no>* pre
        , const strf::preprinting<SizeRequested, strf::precalc_width::no>* args_pre
        , std::initializer_list<const strf::printer<char_type>*> printers
        , const char_type* tr_string
        , const char_type* tr_string_end
        , Charset charset
        , ErrHandler err_handler ) noexcept
        : tr_string_(tr_string)
        , tr_string_end_(tr_string_end)
        , printers_array_(printers.begin())
        , num_printers_(static_cast<std::ptrdiff_t>(printers.size()))
        , charset_(charset)
        , err_handler_(err_handler)
    {
        STRF_MAYBE_UNUSED(pre);
        STRF_IF_CONSTEXPR (static_cast<bool>(SizeRequested)) {
            auto invalid_arg_size = charset.replacement_char_size();
            const std::ptrdiff_t s = strf::detail::tr_string_size
                ( args_pre, static_cast<std::ptrdiff_t>(printers.size())
                , tr_string, tr_string_end, invalid_arg_size );
            pre->add_size(s);
        } else {
            (void) args_pre;
        }
    }

    STRF_HD void print_to(strf::destination<char_type>& dst) const
    {
        strf::detail::tr_string_write
            ( tr_string_, tr_string_end_, printers_array_, num_printers_
            , dst, charset_, err_handler_ );
    }

private:

    const char_type* tr_string_;
    const char_type* tr_string_end_;
    const strf::printer<char_type>* const * printers_array_;
    std::ptrdiff_t num_printers_;
    Charset charset_;
    ErrHandler err_handler_;
};

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_TR_PRINTER_HPP

