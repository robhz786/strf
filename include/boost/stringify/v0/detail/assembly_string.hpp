#ifndef BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP
#define BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_arg.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharT>
class asm_string_processor
{
public:

    virtual ~asm_string_processor()
    {
    }

    virtual bool good() = 0;

    virtual void put(const CharT* begin, const CharT* end) = 0;

    virtual void put_arg(std::size_t index) = 0;
};


template <typename CharT, typename FTuple>
class asm_string_writer: public asm_string_processor<CharT>
{
    using arg_type =  stringify::v0::input_arg<CharT, FTuple>;
    using arglist_type = std::initializer_list<arg_type>;

public:

    asm_string_writer
        ( const arglist_type& arglist
        , stringify::v0::output_writer<CharT>& writer
        , const FTuple& ft
        )
        : m_arglist(arglist)
        , m_writer(writer)
        , m_ftuple(ft)
    {
    }

    virtual bool good()
    {
        return m_writer.good();
    }

    virtual void put(const CharT* begin, const CharT* end)
    {
        BOOST_ASSERT(end >= begin);
        m_writer.put(begin, end - begin);
    }

    virtual void put_arg(std::size_t index)
    {
        if (index >= m_arglist.size())
        {
            m_writer.set_error(std::make_error_code(std::errc::value_too_large));
        }
        else
        {
            m_arglist.begin()[index].write(m_writer, m_ftuple);
        }
    }

private:

    arglist_type m_arglist;
    stringify::v0::output_writer<CharT>& m_writer;
    const FTuple& m_ftuple;
};

template <typename CharT, typename FTuple>
class asm_string_measurer: public asm_string_processor<CharT>
{
    using arg_type =  stringify::v0::input_arg<CharT, FTuple>;
    using arglist_type = std::initializer_list<arg_type>;

public:

    explicit asm_string_measurer
        ( const arglist_type& arglist
        , const FTuple& ft
        )
        : m_arglist(arglist)
        , m_ftuple(ft)
    {
    }

    virtual bool good()
    {
        return true;
    }

    virtual void put(const CharT* begin, const CharT* end)
    {
        BOOST_ASSERT(end >= begin);
        m_length += (end - begin);
    }

    virtual void put_arg(std::size_t index)
    {
        if (index < m_arglist.size())
        {
            m_length += m_arglist.begin()[index].length(m_ftuple);
        }
    }

    std::size_t result() const
    {
        return m_length;
    }

private:

    std::size_t m_length = 0;
    arglist_type m_arglist;
    const FTuple& m_ftuple;
};


template <typename CharT>
struct read_uint_result
{
    std::size_t value;
    const CharT* ptr;
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

template <typename CharT>
const CharT* after_closing_bracket(const CharT* it, const CharT* end)
{
    it = std::find(it, end, static_cast<CharT>('}'));
    return it == end ? end : it + 1;
}


template <typename CharT>
void parse_asm_string
    ( const CharT* it
    , const CharT* end
    , asm_string_processor<CharT>& proc
    )
{
    std::size_t arg_idx = 0;

    while(it != end && proc.good())
    {
        {
            auto prev = it;
            it = std::find(it, end, static_cast<CharT>('{'));
            proc.put(prev, it);
        }
        if (it == end)
        {
            return;
        }
        if (++it == end)
        {
            proc.put_arg(arg_idx);
            return;
        }
        CharT ch = *it;
        if (ch == static_cast<CharT>('}'))
        {
            ++it;
            proc.put_arg(arg_idx);
            ++arg_idx;
        }
        else if (static_cast<CharT>('0') <= ch && ch <= static_cast<CharT>('9'))
        {
            auto res = read_uint(it, end);
            it = after_closing_bracket(res.ptr, end);
            proc.put_arg(res.value);
        }
        else if (ch == static_cast<CharT>('/'))
        {
            ++it;
        }
        else if (ch == static_cast<CharT>('-'))
        {
            it = after_closing_bracket(it + 1 , end);
        }
        else
        {
            it = after_closing_bracket(it + 1, end);
            proc.put_arg(arg_idx);
            ++arg_idx;
        }
    }
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
void parse_asm_string<char>
    ( const char* it, const char* end, asm_string_processor<char>& proc);

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
void parse_asm_string<char16_t>
    ( const char16_t* it, const char16_t* end, asm_string_processor<char16_t>& proc);

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
void parse_asm_string<char32_t>
    ( const char32_t* it, const char32_t* end, asm_string_processor<char32_t>& proc);

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
void parse_asm_string<wchar_t>
    ( const wchar_t* it, const wchar_t* end, asm_string_processor<wchar_t>& proc);

#endif

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP

