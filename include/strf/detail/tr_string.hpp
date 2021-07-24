#ifndef STRF_DETAIL_TR_STRING_HPP
#define STRF_DETAIL_TR_STRING_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuff_functions.hpp>
#include <strf/printer.hpp>
#include <strf/detail/facets/charset.hpp>

namespace strf {

struct tr_error_notifier_c;

struct default_tr_error_notifier
{
    using category = strf::tr_error_notifier_c;

    template <typename Charset>
    inline STRF_HD void handle
        ( const typename Charset::code_unit* str
        , std::size_t str_len
        , std::size_t err_pos
        , Charset charset ) noexcept
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
    std::size_t value;
    const CharT* it;
};

template <typename CharT>
STRF_HD read_uint_result<CharT> read_uint(const CharT* it, const CharT* end, std::size_t limit) noexcept
{
    std::size_t value = *it -  static_cast<CharT>('0');
    ++it;
    while (it != end) {
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
STRF_HD inline std::size_t tr_string_size
    ( const strf::print_preview<strf::preview_size::no, strf::preview_width::no>*
    , std::size_t
    , const CharT*
    , const CharT*
    , std::size_t ) noexcept
{
    return 0;
}

template <typename CharT>
STRF_HD std::size_t tr_string_size
    ( const strf::print_preview<strf::preview_size::yes, strf::preview_width::no>* args_preview
    , std::size_t num_args
    , const CharT* it
    , const CharT* end
    , std::size_t inv_arg_size ) noexcept
{
    std::size_t count = 0;
    std::size_t arg_idx = 0;

    while (it != end) {
        const CharT* prev = it;
        it = strf::detail::str_find<CharT>(it, (end - it), '{');
        if (it == nullptr) {
            count += (end - prev);
            break;
        }
        count += (it - prev);
        ++it;

        after_the_brace:
        if (it == end) {
            if (arg_idx < num_args) {
                count += args_preview[arg_idx].accumulated_size();
            } else {
                count += inv_arg_size;
            }
            break;
        }

        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                count += args_preview[arg_idx].accumulated_size();
                ++arg_idx;
            } else {
                count += inv_arg_size;
            }
            ++it;
        } else if (CharT('0') <= ch && ch <= CharT('9')) {
            auto result = strf::detail::read_uint(it, end, num_args);

            if (result.value < num_args) {
                count += args_preview[result.value].accumulated_size();
            } else {
                count += inv_arg_size;
            }
            it = strf::detail::str_find<CharT>(result.it, end - result.it, '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            auto it2 = it + 1;
            it2 = strf::detail::str_find<CharT>(it2, end - it2, '{');
            if (it2 == nullptr) {
                return count += end - it;
            }
            count += (it2 - it);
            it = it2 + 1;
            goto after_the_brace;
        } else {
            if (ch != '-') {
                if (arg_idx < num_args) {
                    count += args_preview[arg_idx].accumulated_size();
                    ++arg_idx;
                } else {
                    count += inv_arg_size;
                }
            }
            auto it2 = it + 1;
            it = strf::detail::str_find<CharT>(it2, (end - it2), '}');
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
    , std::size_t num_args
    , strf::basic_outbuff<typename Charset::code_unit>& ob
    , Charset charset
    , ErrHandler err_handler )
{
    std::size_t arg_idx = 0;
    using char_type = typename Charset::code_unit;

    auto it = str;
    std::size_t str_len = str_end - str;
    while (it != str_end) {
        const char_type* prev = it;
        it = strf::detail::str_find<char_type>(it, (str_end - it), '{');
        if (it == nullptr) {
            ob.write(prev, str_end - prev);
            return;
        }
        ob.write(prev, it - prev);
        ++it;
        after_the_brace:
        if (it == str_end) {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(ob);
            } else {
                charset.write_replacement_char(ob);
                err_handler.handle(str, str_len, (it - str) - 1, charset);
            }
            break;
        }
        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(ob);
                ++arg_idx;
            } else {
                charset.write_replacement_char(ob);
                err_handler.handle(str, str_len, (it - str) - 1, charset);
            }
            ++it;
        } else if (char_type('0') <= ch && ch <= char_type('9')) {
            auto result = strf::detail::read_uint(it, str_end, num_args);
            if (result.value < num_args) {
                args[result.value]->print_to(ob);
            } else {
                charset.write_replacement_char(ob);
                err_handler.handle(str, str_len, (it - str) - 1, charset);
            }
            it = strf::detail::str_find<char_type>(result.it, str_end - result.it, '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            auto it2 = it + 1;
            it2 = strf::detail::str_find<char_type>(it2, str_end - it2, '{');
            if (it2 == nullptr) {
                ob.write(it, str_end - it);
                return;
            }
            ob.write(it, (it2 - it));
            it = it2 + 1;
            goto after_the_brace;
        } else {
            if (ch != '-') {
                if (arg_idx < num_args) {
                    args[arg_idx]->print_to(ob);
                    ++arg_idx;
                } else {
                    charset.write_replacement_char(ob);
                    err_handler.handle(str, str_len, (it - str) - 1, charset);
                }
            }
            auto it2 = it + 1;
            it = strf::detail::str_find<char_type>(it2, (str_end - it2), '}');
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

    template <strf::preview_size SizeRequested>
    STRF_HD tr_string_printer
        ( strf::print_preview<SizeRequested, strf::preview_width::no>& preview
        , const strf::print_preview<SizeRequested, strf::preview_width::no>* args_preview
        , std::initializer_list<const strf::printer<char_type>*> printers
        , const char_type* tr_string
        , const char_type* tr_string_end
        , Charset charset
        , ErrHandler err_handler ) noexcept
        : tr_string_(tr_string)
        , tr_string_end_(tr_string_end)
        , printers_array_(printers.begin())
        , num_printers_(printers.size())
        , charset_(charset)
        , err_handler_(err_handler)
    {
        STRF_IF_CONSTEXPR (static_cast<bool>(SizeRequested)) {
            auto invalid_arg_size = charset.replacement_char_size();
            std::size_t s = strf::detail::tr_string_size
                ( args_preview, printers.size(), tr_string, tr_string_end
                , invalid_arg_size );
            preview.add_size(s);
        } else {
            (void) args_preview;
        }
    }

    STRF_HD void print_to(strf::basic_outbuff<char_type>& ob) const
    {
        strf::detail::tr_string_write
            ( tr_string_, tr_string_end_, printers_array_, num_printers_
            , ob, charset_, err_handler_ );
    }

private:

    const char_type* tr_string_;
    const char_type* tr_string_end_;
    const strf::printer<char_type>* const * printers_array_;
    std::size_t num_printers_;
    Charset charset_;
    ErrHandler err_handler_;
};


} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_TR_STRING_HPP

