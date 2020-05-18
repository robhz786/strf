#ifndef STRF_DETAIL_TR_STRING_HPP
#define STRF_DETAIL_TR_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <limits>
#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/standard_lib_functions.hpp>

namespace strf {

enum class tr_invalid_arg
{
    replace, ignore
};

struct tr_invalid_arg_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD tr_invalid_arg get_default() noexcept
    {
        return tr_invalid_arg::replace;
    }
};

template <typename Facet>
struct facet_traits;

template <>
struct facet_traits<strf::tr_invalid_arg>
{
    using category = strf::tr_invalid_arg_c;
    static constexpr bool store_by_value = true;
};


namespace detail {

template <typename CharT>
struct read_uint_result
{
    std::size_t value;
    const CharT* it;
};


template <typename CharT>
read_uint_result<CharT> read_uint(const CharT* it, const CharT* end) noexcept
{
    std::size_t value = *it -  static_cast<CharT>('0');
    constexpr long limit = std::numeric_limits<long>::max() / 10 - 9;
    ++it;
    while (it != end) {
        CharT ch = *it;
        if (ch < static_cast<CharT>('0') || static_cast<CharT>('9') < ch) {
            break;
        }
        if(value > limit) {
            value = std::numeric_limits<std::size_t>::max();
            break;
        }
        value *= 10;
        value += ch - static_cast<CharT>('0');
        ++it;
    }
    return {value, it};
}

constexpr std::size_t trstr_invalid_arg_size_when_stop = (std::size_t)-1;

template <typename CharT>
inline std::size_t tr_string_size
    ( const strf::print_preview<strf::preview_size::no, strf::preview_width::no>*
    , std::size_t
    , const CharT*
    , const CharT*
    , std::size_t ) noexcept
{
    return 0;
}

template <typename CharT>
std::size_t tr_string_size
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
                count += args_preview[arg_idx].get_size();
            } else if (inv_arg_size != trstr_invalid_arg_size_when_stop) {
                count += inv_arg_size;
            }
            break;
        }

        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                count += args_preview[arg_idx].get_size();
                ++arg_idx;
            } else if(inv_arg_size == trstr_invalid_arg_size_when_stop) {
                break;
            } else {
                count += inv_arg_size;
            }
            ++it;
        } else if (CharT('0') <= ch && ch <= CharT('9')) {
            auto result = strf::detail::read_uint(it, end);

            if (result.value < num_args) {
                count += args_preview[result.value].get_size();
            } else if(inv_arg_size == trstr_invalid_arg_size_when_stop) {
                break;
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
                    count += args_preview[arg_idx].get_size();
                    ++arg_idx;
                } else if(inv_arg_size == trstr_invalid_arg_size_when_stop) {
                    break;
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

template <std::size_t CharSize>
void tr_string_write
    ( const strf::underlying_char_type<CharSize>* it
    , const strf::underlying_char_type<CharSize>* end
    , const strf::printer<CharSize>* const * args
    , std::size_t num_args
    , strf::underlying_outbuf<CharSize>& ob
    , strf::write_replacement_char_f<CharSize> write_replacement_char
    , strf::tr_invalid_arg policy )
{
    using char_type = strf::underlying_char_type<CharSize>;
    std::size_t arg_idx = 0;

    while (it != end) {
        const char_type* prev = it;
        it = strf::detail::str_find<char_type>(it, (end - it), '{');
        if (it == nullptr) {
            strf::write(ob, prev, end - prev);
            return;
        }
        strf::write(ob, prev, it - prev);
        ++it;
        after_the_brace:
        if (it == end) {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(ob);
            } else if (policy == strf::tr_invalid_arg::replace) {
                write_replacement_char(ob);
            }
            break;
        }
        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(ob);
                ++arg_idx;
            } else if (policy == strf::tr_invalid_arg::replace) {
                write_replacement_char(ob);
            }
            ++it;
        } else if (char_type('0') <= ch && ch <= char_type('9')) {
            auto result = strf::detail::read_uint(it, end);
            if (result.value < num_args) {
                args[result.value]->print_to(ob);
            } else if (policy == strf::tr_invalid_arg::replace) {
                write_replacement_char(ob);
            }
            it = strf::detail::str_find<char_type>(result.it, end - result.it, '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            auto it2 = it + 1;
            it2 = strf::detail::str_find<char_type>(it2, end - it2, '{');
            if (it2 == nullptr) {
                strf::write(ob, it, end - it);
                return;
            }
            strf::write(ob, it, (it2 - it));
            it = it2 + 1;
            goto after_the_brace;
        } else {
            if (ch != '-') {
                if (arg_idx < num_args) {
                    args[arg_idx]->print_to(ob);
                    ++arg_idx;
                } else if (policy == strf::tr_invalid_arg::replace) {
                    write_replacement_char(ob);
                }
            }
            auto it2 = it + 1;
            it = strf::detail::str_find<char_type>(it2, (end - it2), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        }
    }
}

template <std::size_t CharSize>
class tr_string_printer
{
public:

    using char_type = strf::underlying_char_type<CharSize>;

    template <strf::preview_size SizeRequested, typename CharEncoding>
    tr_string_printer
        ( strf::print_preview<SizeRequested, strf::preview_width::no>& preview
        , const strf::print_preview<SizeRequested, strf::preview_width::no>* args_preview
        , std::initializer_list<const strf::printer<CharSize>*> printers
        , const char_type* tr_string
        , const char_type* tr_string_end
        , CharEncoding enc
        , strf::tr_invalid_arg policy ) noexcept
        : tr_string_(reinterpret_cast<const char_type*>(tr_string))
        , tr_string_end_(reinterpret_cast<const char_type*>(tr_string_end))
        , write_replacement_char_func_(enc.write_replacement_char_func())
        , printers_array_(printers.begin())
        , num_printers_(printers.size())
        , policy_(policy)
    {
        STRF_IF_CONSTEXPR (static_cast<bool>(SizeRequested)) {
            std::size_t invalid_arg_size;
            switch (policy) {
                case strf::tr_invalid_arg::replace:
                    invalid_arg_size = enc.replacement_char_size();
                    break;
                default:
                    STRF_ASSERT(policy == strf::tr_invalid_arg::ignore);
                    invalid_arg_size = 0;
                    break;
            }
            std::size_t s = strf::detail::tr_string_size
                ( args_preview, printers.size(), tr_string, tr_string_end
                , invalid_arg_size );
            preview.add_size(s);
        } else {
            (void) args_preview;
        }
    }

    void print_to(strf::underlying_outbuf<CharSize>& ob) const
    {
        strf::detail::tr_string_write
            ( tr_string_, tr_string_end_, printers_array_, num_printers_
            , ob, write_replacement_char_func_, policy_ );
    }

private:

    const char_type* tr_string_;
    const char_type* tr_string_end_;
    strf::write_replacement_char_f<CharSize> write_replacement_char_func_;
    const strf::printer<CharSize>* const * printers_array_;
    std::size_t num_printers_;
    strf::tr_invalid_arg policy_;
};


} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_TR_STRING_HPP

