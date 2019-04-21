#ifndef BOOST_STRINGIFY_V0_DETAIL_ASM_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_ASM_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class asm_invalid_arg
{
    replace, stop, ignore
};

struct asm_invalid_arg_category
{
    static constexpr bool constrainable = false;

    static asm_invalid_arg get_default()
    {
        return asm_invalid_arg::replace;
    }
};

template <typename Facet>
class facet_trait;

template <>
class facet_trait<stringify::v0::asm_invalid_arg>
{
public:
    using category = stringify::v0::asm_invalid_arg_category;
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

constexpr std::size_t asmstr_invalid_arg_size_when_stop = (std::size_t)-1;

template <typename CharT>
std::size_t invalid_arg_size
    ( stringify::v0::encoding<CharT> enc
    , asm_invalid_arg policy )
{
    switch(policy)
    {
        case asm_invalid_arg::replace:
            return enc.replacement_char_size();
        case asm_invalid_arg::stop:
            return stringify::v0::detail::asmstr_invalid_arg_size_when_stop;
        default:
            return 0;
    }
}

template <typename CharT>
std::size_t asm_string_size
    ( const CharT* it
    , const CharT* end
    , std::initializer_list<const stringify::v0::printer<CharT>*> args
    , std::size_t inv_arg_size )
{
    using traits = std::char_traits<CharT>;

    std::size_t count = 0;
    std::size_t arg_idx = 0;
    const std::size_t num_args = args.size();

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
                count += args.begin()[arg_idx]->necessary_size();
            }
            else if (inv_arg_size != asmstr_invalid_arg_size_when_stop)
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
                count += args.begin()[arg_idx]->necessary_size();
                ++arg_idx;
            }
            else if(inv_arg_size == asmstr_invalid_arg_size_when_stop)
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
                count += args.begin()[result.value]->necessary_size();
            }
            else if(inv_arg_size == asmstr_invalid_arg_size_when_stop)
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
                    count += args.begin()[arg_idx]->necessary_size();
                    ++arg_idx;
                }
                else if(inv_arg_size == asmstr_invalid_arg_size_when_stop)
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
bool asm_string_write
    ( const CharT* it
    , const CharT* end
    , std::initializer_list<const stringify::v0::printer<CharT>*> args
    , stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , stringify::v0::asm_invalid_arg policy )
{
    using traits = std::char_traits<CharT>;
    std::size_t arg_idx = 0;
    const std::size_t num_args = args.size();

    while (it != end)
    {
        const CharT* prev = it;
        it = traits::find(it, (end - it), '{');
        if (it == nullptr)
        {
            return stringify::v0::detail::write_str(ob, prev, end - prev);
        }

        if ( ! stringify::v0::detail::write_str(ob, prev, it - prev))
        {
            return false;
        }
        ++it;

        after_the_brace:
        if (it == end)
        {
            if (arg_idx < num_args)
            {
                return args.begin()[arg_idx]->write(ob);
            }
            else if (policy == stringify::v0::asm_invalid_arg::replace)
            {
                if (! enc.write_replacement_char(ob))
                {
                    return false;
                }
            }
            else if (policy == stringify::v0::asm_invalid_arg::stop)
            {
                ob.set_error(std::errc::invalid_argument);
                return false;
            }

            break;
        }

        auto ch = *it;
        if (ch == '}')
        {
            if (arg_idx < num_args)
            {
                if (! args.begin()[arg_idx]->write(ob))
                {
                    return false;
                }
                ++arg_idx;
            }
            else if (policy == stringify::v0::asm_invalid_arg::replace)
            {
                if (! enc.write_replacement_char(ob))
                {
                    return false;
                }
            }
            else if (policy == stringify::v0::asm_invalid_arg::stop)
            {
                ob.set_error(std::errc::invalid_argument);
                return false;
            }
            ++it;
        }
        else if (CharT('0') <= ch && ch <= CharT('9'))
        {
            auto result = stringify::v0::detail::read_uint(it, end);

            if (result.value < num_args)
            {
                if (! args.begin()[result.value]->write(ob))
                {
                    return false;
                }
            }
            else if (policy == stringify::v0::asm_invalid_arg::replace)
            {
                if (! enc.write_replacement_char(ob))
                {
                    return false;
                }
            }
            else if (policy == stringify::v0::asm_invalid_arg::stop)
            {
                ob.set_error(std::errc::invalid_argument);
                return false;
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
                return stringify::v0::detail::write_str(ob, it, end - it);
            }
            if (!stringify::v0::detail::write_str(ob, it, (it2 - it)))
            {
                return false;
            }
            it = it2 + 1;
            goto after_the_brace;
        }
        else
        {
            if (ch != '-')
            {
                if (arg_idx < num_args)
                {
                    if (! args.begin()[arg_idx]->write(ob))
                    {
                        return false;
                    }
                    ++arg_idx;
                }
                else if (policy == stringify::v0::asm_invalid_arg::replace)
                {
                    if (! enc.write_replacement_char(ob))
                    {
                        return false;
                    }
                }
                else if (policy == stringify::v0::asm_invalid_arg::stop)
                {
                    ob.set_error(std::errc::invalid_argument);
                    return false;
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
    return true;
}


} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_ASM_STRING_HPP

