#ifndef BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP
#define BOOST_STRINGIFY_V0_DETAIL_MSG_ASSEMBLY_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/facets/encoding.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT>
class asm_string_processor
{
public:

    virtual ~asm_string_processor()
    {
    }

    //virtual bool good() = 0;

    void put(const CharT* begin, const CharT* end)
    {
        if(begin < end)
        {
            do_put(begin, end);
        }
    }

    void put_arg(std::size_t index)
    {
        do_put_arg(index);
    }

    virtual void do_put(const CharT* begin, const CharT* end) = 0;

    virtual void do_put_arg(std::size_t index) = 0;
};

template <typename CharIn, typename CharOut>
class asm_string_writer: public asm_string_processor<CharIn>
{
public:

    using printer_ptr = const stringify::v0::printer<CharOut>*;
    using arglist_type = std::initializer_list<printer_ptr>;

    template <typename FPack>
    asm_string_writer
        ( stringify::v0::output_writer<CharOut>& dest
        , const FPack& fp
        , const arglist_type& args
        , bool sanitise )
        : m_dest(dest)
        , m_args(args)
        , m_sw
            ( dest
            , get_facet<stringify::v0::encoding_category<CharIn>>(fp)
            , sanitise )
    {
    }

    // bool good() override
    // {
    //     return m_dest.good();
    // }

    void do_put(const CharIn* begin, const CharIn* end) override
    {
        m_sw.write(begin, end);
    }

    void do_put_arg(std::size_t index) override
    {
        if (index < m_args.size())
        {
            m_args.begin()[index]->write();
        }
        else
        {
            m_dest.set_error(std::make_error_code(std::errc::value_too_large));
        }
    }

private:
    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& fp) const
    {
        using input_tag = stringify::v0::asm_string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }

    stringify::v0::output_writer<CharOut>& m_dest;
    arglist_type m_args;
    stringify::v0::string_writer<CharIn, CharOut> m_sw;
};


template <typename CharIn, typename CharOut>
class asm_string_measurer: public asm_string_processor<CharIn>
{

    using printer_ptr = const stringify::v0::printer<CharOut>*;
    using arglist_type = std::initializer_list<printer_ptr>;

public:

    template <typename FPack>
    asm_string_measurer
        ( stringify::v0::output_writer<CharOut>& dest
        , const FPack& fp
        , const arglist_type& args
        , bool sanitise )
        : m_args(args)
        , m_sw
            ( dest
            , get_facet<stringify::v0::encoding_category<CharIn>>(fp)
            , sanitise )
    {
    }

    // virtual bool good()
    // {
    //     return true;
    // }

    virtual void do_put(const CharIn* begin, const CharIn* end)
    {
        m_length += m_sw.necessary_size(begin, end);
    }

    virtual void do_put_arg(std::size_t index)
    {
        if (index < m_args.size())
        {
            m_length += m_args.begin()[index]->necessary_size();
        }
    }

    std::size_t result() const
    {
        return m_length;
    }

private:

    std::size_t m_length = 0;
    arglist_type m_args;
    stringify::v0::string_writer<CharIn, CharOut> m_sw;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& fp) const
    {
        using input_tag = stringify::v0::asm_string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }
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
    const CharT* prev = it;
    while(it != end)
    {
        if (*it != '{')
        {
            ++it;
            continue;
        }
        if (++it == end)
        {
            proc.put(prev, it - 1);
            proc.put_arg(arg_idx);
            prev = end;
            break;
        }
        CharT ch = *it;
        if(ch == static_cast<CharT>('}'))
        {
            proc.put(prev, it - 1);
            prev = ++it;
            proc.put_arg(arg_idx);
            ++arg_idx;
            continue;
        }
        if(static_cast<CharT>('0') <= ch && ch <= static_cast<CharT>('9'))
        {
            proc.put(prev, it - 1);
            auto res = read_uint(it, end);
            it = after_closing_bracket(res.ptr, end);
            prev = it;
            proc.put_arg(res.value);
            continue;
        }
        if(ch == static_cast<CharT>('{'))
        {
            proc.put(prev, it);
            ++it;
            prev = it;
            continue;
        }
        proc.put(prev, it - 1);
        it = after_closing_bracket(it, end);
        prev = it;
        if(ch != static_cast<CharT>('-'))
        {
            proc.put_arg(arg_idx);
            ++arg_idx;
        }
    }
    proc.put(prev, end);
}


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<wchar_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_writer<char32_t, char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<wchar_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class asm_string_measurer<char32_t, char32_t>;

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

