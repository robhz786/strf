#ifndef BOOST_STRINGIFY_V0_DETAIL_TR_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_TR_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/detail/facets/encoding.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class tr_invalid_arg
{
    replace, stop, ignore
};

struct tr_invalid_arg_c
{
    static constexpr bool constrainable = false;

    static tr_invalid_arg get_default()
    {
        return tr_invalid_arg::replace;
    }
};

template <typename Facet>
class facet_trait;

template <>
class facet_trait<stringify::v0::tr_invalid_arg>
{
public:
    using category = stringify::v0::tr_invalid_arg_c;
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
read_uint_result<CharT> read_uint(const CharT* it, const CharT* end)
{
    std::size_t value = *it -  static_cast<CharT>('0');
    constexpr long limit = std::numeric_limits<long>::max() / 10 - 9;
    ++it;
    while (it != end)
    {
        CharT ch = *it;
        if (ch < static_cast<CharT>('0') || static_cast<CharT>('9') < ch)
        {
            break;
        }
        if(value > limit)
        {
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
    ( stringify::v0::encoding<CharT> enc
    , tr_invalid_arg policy )
{
    switch(policy)
    {
        case tr_invalid_arg::replace:
            return enc.replacement_char_size();
        case tr_invalid_arg::stop:
            return stringify::v0::detail::trstr_invalid_arg_size_when_stop;
        default:
            return 0;
    }
}

template <typename CharT>
std::size_t tr_string_size
    ( const CharT* it
    , const CharT* end
    , const stringify::v0::printer<CharT>* const * args
    , std::size_t num_args
    , std::size_t inv_arg_size )
{
    using traits = std::char_traits<CharT>;

    std::size_t count = 0;
    std::size_t arg_idx = 0;

    while (it != end)
    {
        const CharT* prev = it;
        it = traits::find(it, (end - it), '{');
        if (it == nullptr)
        {
            count += (end - prev);
            break;
        }
        count += (it - prev);
        ++it;

        after_the_brace:
        if (it == end)
        {
            if (arg_idx < num_args)
            {
                count += args[arg_idx]->necessary_size();
            }
            else if (inv_arg_size != trstr_invalid_arg_size_when_stop)
            {
                count += inv_arg_size;
            }
            break;
        }

        auto ch = *it;
        if (ch == '}')
        {
            if (arg_idx < num_args)
            {
                count += args[arg_idx]->necessary_size();
                ++arg_idx;
            }
            else if(inv_arg_size == trstr_invalid_arg_size_when_stop)
            {
                break;
            }
            else
            {
                count += inv_arg_size;
            }
            ++it;
        }
        else if (CharT('0') <= ch && ch <= CharT('9'))
        {
            auto result = stringify::v0::detail::read_uint(it, end);

            if (result.value < num_args)
            {
                count += args[result.value]->necessary_size();
            }
            else if(inv_arg_size == trstr_invalid_arg_size_when_stop)
            {
                break;
            }
            else
            {
                count += inv_arg_size;
            }
            it = traits::find(result.it, end - result.it, '}');
            if (it == nullptr)
            {
                break;
            }
            ++it;
        }
        else if(ch == '{')
        {
            auto it2 = it + 1;
            it2 = traits::find(it2, end - it2, '{');
            if (it2 == nullptr)
            {
                return count += end - it;
            }
            count += (it2 - it);
            it = it2 + 1;
            goto after_the_brace;
        }
        else
        {
            if (ch != '-')
            {
                if (arg_idx < num_args)
                {
                    count += args[arg_idx]->necessary_size();
                    ++arg_idx;
                }
                else if(inv_arg_size == trstr_invalid_arg_size_when_stop)
                {
                    break;
                }
                else
                {
                    count += inv_arg_size;
                }
            }
            auto it2 = it + 1;
            it = traits::find(it2, (end - it2), '}');
            if (it == nullptr)
            {
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
    , const stringify::v0::printer<CharT>* const * args
    , std::size_t num_args
    , stringify::v0::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , stringify::v0::tr_invalid_arg policy )
{
    using traits = std::char_traits<CharT>;
    std::size_t arg_idx = 0;

    while (it != end)
    {
        const CharT* prev = it;
        it = traits::find(it, (end - it), '{');
        if (it == nullptr)
        {
            stringify::v0::write(ob, prev, end - prev);
            return;
        }

        stringify::v0::write(ob, prev, it - prev);
        ++it;

        after_the_brace:
        if (it == end)
        {
            if (arg_idx < num_args)
            {
                args[arg_idx]->print_to(ob);
            }
            else if (policy == stringify::v0::tr_invalid_arg::replace)
            {
                enc.write_replacement_char(ob);
            }
            else if (policy == stringify::v0::tr_invalid_arg::stop)
            {
                throw stringify::v0::tr_string_syntax_error();
            }
            break;
        }

        auto ch = *it;
        if (ch == '}')
        {
            if (arg_idx < num_args)
            {
                args[arg_idx]->print_to(ob);
                ++arg_idx;
            }
            else if (policy == stringify::v0::tr_invalid_arg::replace)
            {
                enc.write_replacement_char(ob);
            }
            else if (policy == stringify::v0::tr_invalid_arg::stop)
            {
                throw stringify::v0::tr_string_syntax_error();
            }
            ++it;
        }
        else if (CharT('0') <= ch && ch <= CharT('9'))
        {
            auto result = stringify::v0::detail::read_uint(it, end);

            if (result.value < num_args)
            {
                args[result.value]->print_to(ob);            }
            else if (policy == stringify::v0::tr_invalid_arg::replace)
            {
                enc.write_replacement_char(ob);
            }
            else if (policy == stringify::v0::tr_invalid_arg::stop)
            {
                throw stringify::v0::tr_string_syntax_error();
            }

            it = traits::find(result.it, end - result.it, '}');
            if (it == nullptr)
            {
                break;
            }
            ++it;
        }
        else if(ch == '{')
        {
            auto it2 = it + 1;
            it2 = traits::find(it2, end - it2, '{');
            if (it2 == nullptr)
            {
                stringify::v0::write(ob, it, end - it);
                return;
            }
            stringify::v0::write(ob, it, (it2 - it));
            it = it2 + 1;
            goto after_the_brace;
        }
        else
        {
            if (ch != '-')
            {
                if (arg_idx < num_args)
                {
                    args[arg_idx]->print_to(ob);
                    ++arg_idx;
                }
                else if (policy == stringify::v0::tr_invalid_arg::replace)
                {
                    enc.write_replacement_char(ob);
                }
                else if (policy == stringify::v0::tr_invalid_arg::stop)
                {
                    throw stringify::v0::tr_string_syntax_error();
                }
            }
            auto it2 = it + 1;
            it = traits::find(it2, (end - it2), '}');
            if (it == nullptr)
            {
                break;
            }
            ++it;
        }
    }
}

// template <typename PrintersTuple>
// class tr_string_printer
// {
// public:
//     using char_type = typename PrintersTuple::char_type;

//     tr_string_printer( const PrintersTuple& pt
//                      , const char_type* tr_string
//                      , const char_type* tr_string_end
//                      , stringify::v0::encoding<char_type> enc
//                      , stringify::v0::tr_invalid_arg policy )
//         : _tr_string(tr_string)
//         , _tr_string_end(tr_string_end)
//         , _enc(enc)
//         , _policy(policy)
//     {
//         pt.fill_array(_printers_array);
//     }

//     std::size_t necessary_size() const
//     {
//         return stringify::v0::detail::tr_string_size
//             ( _tr_string, _tr_string_end
//             , _printers_array, PrintersTuple::size
//             , stringify::v0::detail::invalid_arg_size(_enc, _policy) );
//     }

//     void print_to(stringify::v0::basic_outbuf<char_type>& ob) const
//     {
//         stringify::v0::detail::tr_string_write
//             ( _tr_string, _tr_string_end
//             , _printers_array, PrintersTuple::size
//             , ob, _enc, _policy );
//     }

//     const char_type* _tr_string;
//     const char_type* _tr_string_end;
//     stringify::v0::encoding<char_type> _enc;
//     stringify::v0::tr_invalid_arg _policy;
//     const stringify::v0::printer<char_type>* _printers_array[PrintersTuple::size];
// };

template <typename CharT>
class tr_string_printer
{
public:
    using char_type = CharT;

    tr_string_printer
        ( std::initializer_list<const stringify::v0::printer<CharT>*> printers
        , const CharT* tr_string
        , const CharT* tr_string_end
        , stringify::v0::encoding<CharT> enc
        , stringify::v0::tr_invalid_arg policy )
        : _tr_string(tr_string)
        , _tr_string_end(tr_string_end)
        , _enc(enc)
        , _policy(policy)
        , _printers_array(printers.begin())
        , _num_printers(printers.size())
    {
    }

    std::size_t necessary_size() const
    {
        return stringify::v0::detail::tr_string_size
            ( _tr_string, _tr_string_end
            , _printers_array, _num_printers
            , stringify::v0::detail::invalid_arg_size(_enc, _policy) );
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const
    {
        stringify::v0::detail::tr_string_write
            ( _tr_string, _tr_string_end
            , _printers_array, _num_printers
            , ob, _enc, _policy );
    }

    const CharT* _tr_string;
    const CharT* _tr_string_end;
    stringify::v0::encoding<CharT> _enc;
    stringify::v0::tr_invalid_arg _policy;
    const stringify::v0::printer<CharT>* const * _printers_array;
    std::size_t _num_printers;
};



} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_TR_STRING_HPP

