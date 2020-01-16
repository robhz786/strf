#ifndef STRF_DETAIL_TR_STRING_HPP
#define STRF_DETAIL_TR_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <limits>
#include <strf/detail/facets/encoding.hpp>

namespace strf {

class tr_string_syntax_error: public strf::stringify_error
{
    using strf::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: Tr-string syntax error";
    }
};

namespace detail {

#if defined(__cpp_exceptions)

inline void throw_string_syntax_error()
{
    throw strf::tr_string_syntax_error();
}

#else // defined(__cpp_exceptions)

inline void throw_string_syntax_error()
{
    std::abort();
}

#endif // defined(__cpp_exceptions)

} // namespace detail

enum class tr_invalid_arg
{
    replace, stop, ignore
};

struct tr_invalid_arg_c
{
    static constexpr bool constrainable = false;

    static constexpr tr_invalid_arg get_default() noexcept
    {
        return tr_invalid_arg::replace;
    }
};

template <typename Facet>
class facet_trait;

template <>
class facet_trait<strf::tr_invalid_arg>
{
public:
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
std::size_t invalid_arg_size
    ( strf::encoding<CharT> enc
    , tr_invalid_arg policy ) noexcept
{
    switch(policy) {
        case tr_invalid_arg::replace:
            return enc.replacement_char_size();
        case tr_invalid_arg::stop:
            return strf::detail::trstr_invalid_arg_size_when_stop;
        default:
            return 0;
    }
}

template <typename CharT>
inline std::size_t tr_string_size
    ( const strf::print_preview<false, false>*
    , std::size_t
    , const CharT*
    , const CharT*
    , std::size_t ) noexcept
{
    return 0;
}

template <typename CharT>
std::size_t tr_string_size
    ( const strf::print_preview<true, false>* args_preview
    , std::size_t num_args
    , const CharT* it
    , const CharT* end
    , std::size_t inv_arg_size ) noexcept
{
    using traits = std::char_traits<CharT>;

    std::size_t count = 0;
    std::size_t arg_idx = 0;

    while (it != end) {
        const CharT* prev = it;
        it = traits::find(it, (end - it), '{');
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
            it = traits::find(result.it, end - result.it, '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            auto it2 = it + 1;
            it2 = traits::find(it2, end - it2, '{');
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
            it = traits::find(it2, (end - it2), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        }
    }
    return count;
}

template <typename CharT>
void tr_string_write
    ( const CharT* it
    , const CharT* end
    , const strf::printer<CharT>* const * args
    , std::size_t num_args
    , strf::basic_outbuf<CharT>& ob
    , strf::encoding<CharT> enc
    , strf::tr_invalid_arg policy )
{
    using traits = std::char_traits<CharT>;
    std::size_t arg_idx = 0;

    while (it != end) {
        const CharT* prev = it;
        it = traits::find(it, (end - it), '{');
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
                enc.write_replacement_char(ob);
            } else if (policy == strf::tr_invalid_arg::stop) {
                strf::detail::throw_string_syntax_error();
            }
            break;
        }
        auto ch = *it;
        if (ch == '}') {
            if (arg_idx < num_args) {
                args[arg_idx]->print_to(ob);
                ++arg_idx;
            } else if (policy == strf::tr_invalid_arg::replace) {
                enc.write_replacement_char(ob);
            } else if (policy == strf::tr_invalid_arg::stop) {
                strf::detail::throw_string_syntax_error();
            }
            ++it;
        } else if (CharT('0') <= ch && ch <= CharT('9')) {
            auto result = strf::detail::read_uint(it, end);
            if (result.value < num_args) {
                args[result.value]->print_to(ob);
            } else if (policy == strf::tr_invalid_arg::replace) {
                enc.write_replacement_char(ob);
            } else if (policy == strf::tr_invalid_arg::stop) {
                strf::detail::throw_string_syntax_error();
            }
            it = traits::find(result.it, end - result.it, '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        } else if(ch == '{') {
            auto it2 = it + 1;
            it2 = traits::find(it2, end - it2, '{');
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
                    enc.write_replacement_char(ob);
                } else if (policy == strf::tr_invalid_arg::stop) {
                    strf::detail::throw_string_syntax_error();
                }
            }
            auto it2 = it + 1;
            it = traits::find(it2, (end - it2), '}');
            if (it == nullptr) {
                break;
            }
            ++it;
        }
    }
}

template <typename CharT>
class tr_string_printer
{
public:
    using char_type = CharT;

    template <bool SizeRequested>
    tr_string_printer
        ( strf::print_preview<SizeRequested, false>& preview
        , const strf::print_preview<SizeRequested, false>* args_preview
        , std::initializer_list<const strf::printer<CharT>*> printers
        , const CharT* tr_string
        , const CharT* tr_string_end
        , strf::encoding<CharT> enc
        , strf::tr_invalid_arg policy ) noexcept
        : _tr_string(tr_string)
        , _tr_string_end(tr_string_end)
        , _enc(enc)
        , _policy(policy)
        , _printers_array(printers.begin())
        , _num_printers(printers.size())
    {
        preview.add_size
            ( strf::detail::tr_string_size
                ( args_preview, _num_printers, _tr_string, _tr_string_end
                , strf::detail::invalid_arg_size(_enc, _policy) ) );
    }

    void print_to(strf::basic_outbuf<CharT>& ob) const
    {
        strf::detail::tr_string_write
            ( _tr_string, _tr_string_end, _printers_array, _num_printers
            , ob, _enc, _policy );
    }

    const CharT* _tr_string;
    const CharT* _tr_string_end;
    strf::encoding<CharT> _enc;
    strf::tr_invalid_arg _policy;
    const strf::printer<CharT>* const * _printers_array;
    std::size_t _num_printers;
};



} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_TR_STRING_HPP

