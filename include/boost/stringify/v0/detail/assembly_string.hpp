#ifndef BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP
#define BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <boost/stringify/v0/basic_types.hpp>

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

template <typename CharIn, typename CharOut>
class asm_string_writer: public asm_string_processor<CharIn>
{
public:

    using formatter_ptr = const stringify::v0::formatter<CharOut>*;
    using arglist_type = std::initializer_list<formatter_ptr>;

    template <typename FTuple>
    asm_string_writer
        ( const FTuple&
        , stringify::v0::output_writer<CharOut>& dest
        , arglist_type args
        )
        : m_dest(dest)
        , m_args(args)
    {
    }

    bool good() override
    {
        return m_dest.good();
    }

    void put(const CharIn* begin, const CharIn* end) override
    {
        if(begin < end)
        {
            m_dest.put(begin, (end - begin));
        }
    }

    void put_arg(std::size_t index) override
    {
        if (index < m_args.size())
        {
            m_args.begin()[index]->write(m_dest);
        }
        else
        {
            m_dest.set_error(std::make_error_code(std::errc::value_too_large));
        }
    }

private:

    stringify::v0::output_writer<CharOut>& m_dest;
    arglist_type m_args;
};


template <typename CharIn, typename CharOut>
class asm_string_measurer: public asm_string_processor<CharIn>
{

    using formatter_ptr = const stringify::v0::formatter<CharOut>*;
    using arglist_type = std::initializer_list<formatter_ptr>;

public:

    explicit asm_string_measurer(const arglist_type& arglist)
        : m_arglist(arglist)
    {
    }

    virtual bool good()
    {
        return true;
    }

    virtual void put(const CharIn* begin, const CharIn* end)
    {
        BOOST_ASSERT(end >= begin);
        m_length += (end - begin);
    }

    virtual void put_arg(std::size_t index)
    {
        if (index < m_arglist.size())
        {
            m_length += m_arglist.begin()[index]->length();
        }
    }

    std::size_t result() const
    {
        return m_length;
    }

private:

    std::size_t m_length = 0;
    arglist_type m_arglist;
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
        auto prev = it;
        it = std::find(it, end, static_cast<CharT>('{'));

        if (it == end)
        {
            proc.put(prev, it);
            return;
        }
        if (it + 1 == end)
        {
            proc.put(prev, it);
            proc.put_arg(arg_idx);
            return;
        }

        CharT ch = *++it;

        if (ch == static_cast<CharT>('}'))
        {
            proc.put(prev, it - 1);
            proc.put_arg(arg_idx);
            ++it;
            ++arg_idx;
        }
        else if (static_cast<CharT>('0') <= ch && ch <= static_cast<CharT>('9'))
        {
            proc.put(prev, it - 1);
            auto res = read_uint(it, end);
            it = after_closing_bracket(res.ptr, end);
            proc.put_arg(res.value);
        }
        else if (ch == static_cast<CharT>('/'))
        {
            proc.put(prev, it);
            ++it;
        }
        else if (ch == static_cast<CharT>('-'))
        {
            proc.put(prev, it - 1);
            it = after_closing_bracket(it + 1 , end);
        }
        else
        {
            proc.put(prev, it - 1);
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

