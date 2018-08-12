#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_TO_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_TO_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/syntax.hpp>
#include <boost/stringify/v0/expected.hpp>
#include <boost/stringify/v0/output_types/FILE.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {


// template <typename StringType>
// class string_maker: public buffered_writer<typename StringType::value_type>
// {

// public:
//     using char_type = typename StringType::value_type;
//     using parent = stringify::v0::buffered_writer<char_type>;


//     string_maker(stringify::v0::output_writer_init<char_type> init)
//         : stringify::v0::buffered_writer<char_type>{init}
//     {
//     }

//     ~string_maker()
//     {
//         this->discard();
//     }

//     stringify::v0::expected<StringType, std::error_code> finish()
//     {
//         auto ec = parent::finish();
//         if (ec == std::error_code{})
//         {
//             return {boost::stringify::v0::in_place_t{}, std::move(m_out)};
//         }
//         return {boost::stringify::v0::unexpect_t{}, ec};
//     }

//     StringType finish_exception()
//     {
//         parent::finish_exception();
//         return std::move(m_out);
//     }

//     void reserve(std::size_t size)
//     {
//         m_out.reserve(m_out.size() + size);
//     }

// protected:

//     bool do_put(const char_type* str, std::size_t count) override
//     {
//         m_out.append(str, count);
//         return true;
//     }

// private:

//     StringType m_out;

// };


template <typename StringType>
class string_maker: public output_writer<typename StringType::value_type>
{
public:

    using char_type = typename StringType::value_type;
    using Traits = typename StringType::traits_type;

    string_maker(stringify::v0::output_writer_init<char_type> init)
        : stringify::v0::output_writer<char_type>{init}
    {
        m_out.resize(m_out.capacity());
        m_it = &m_out[0];
        m_end = &m_out[0] + m_out.size();
    }

    void reserve(std::size_t s)
    {
        std::size_t it_pos = m_it - &m_out[0];
        std::size_t old_s = m_out.size();
        std::size_t new_s = old_s + s;
        if (new_s > m_out.capacity())
        {
            m_out.resize(new_s);
            m_it = &m_out[0] + it_pos;
            m_end = &m_out[0] + m_out.size();
        }
    }

    void set_error(std::error_code err) override
    {
        if(m_good)
        {
            m_err = err;
            m_good = false;
        }
    }

    bool good() const override
    {
        return m_good;
    }

    bool put(stringify::v0::piecemeal_writer<char_type>& src) override
    {
        do
        {
            m_it = src.write(m_it, m_end);
        }
        while(src.more() && (resize(), true));
        if(src.success())
        {
            return true;
        }
        set_error(src.get_error());
        return false;
    }
    
    // bool put32(char32_t ch) override
    // {
    //     if(m_good)
    //     {
    //         auto it = this->encode(m_it, m_end, ch);
    //         if(it != nullptr && it != m_end + 1)
    //         {
    //             m_it = it;
    //             return true;
    //         }
    //         if (it == nullptr)
    //         {
    //             return this->signal_encoding_error();
    //         }
    //         resize();
    //         return put32(ch);
    //     }
    //     return false;
    // }

    // bool put32(std::size_t count, char32_t ch) override
    // {
    //     if(m_good)
    //     {
    //         auto res = this->encode(m_it, m_end, count, ch);
    //         if (res.dest_it == nullptr)
    //         {
    //             return this->signal_encoding_error();
    //         }
    //         if(res.count == count)
    //         {
    //             m_it = res.dest_it;
    //             return true;
    //         }
    //         resize();
    //         return put32(count, ch);
    //     }
    //     return false;
    // }

    bool put(const char_type* str, std::size_t count) override
    {
        if(m_good)
        {
            if (m_it + count >= m_end)
            {
                resize();
            }
            Traits::copy(m_it, str, count);
            m_it += count;
            return true;
        }
        return false;
    }

    bool put(char_type ch) override
    {
        if( ! m_good)
        {
            return false;
        }
        if (m_it + 1 >= m_end)
        {
            resize();
        }
        Traits::assign(*m_it, ch);
        ++m_it;
        return true;
     }


    bool put(std::size_t count, char_type ch) override
    {
        if( ! m_good)
        {
            return false;
        }
        if (m_it + count >= m_end)
        {
            resize();
        }
        Traits::assign(m_it, count, ch);
        m_it += count;
        return true;
    }

    stringify::v0::expected<StringType, std::error_code> finish()
    {
        m_out.resize(m_it - &m_out[0]);
        if (m_err == std::error_code{})
        {
            return {boost::stringify::v0::in_place_t{}, std::move(m_out)};
        }
        return {boost::stringify::v0::unexpect_t{}, m_err};
    }

    StringType finish_exception()
    {
        if ( ! m_good)
        {
            throw std::system_error(m_err);
        }
        m_out.resize(m_it - &m_out[0]);
        return static_cast<StringType&&>(m_out);
    }

private:

    void resize()
    {
        std::size_t pos = m_it - &m_out[0];
        m_out.resize(m_out.capacity() * 2);
        m_it = &m_out[0] + pos;
        m_end = &m_out[0] + m_out.size();
    }

    StringType m_out;
    char_type* m_it;
    char_type* m_end;
    std::error_code m_err;
    bool m_good = true;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::wstring>;

#endif

} // namespace detail

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::string, std::error_code>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::u16string, std::error_code>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::u32string, std::error_code>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::wstring, std::error_code>;

#endif



template
    < typename CharT
    , typename Traits = std::char_traits<CharT>
    , typename Allocator = std::allocator<CharT>
    >
constexpr auto to_basic_string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker
         <std::basic_string<CharT, Traits, Allocator>>>();


constexpr auto to_string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::string>>();

constexpr auto to_u16string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::u16string>>();

constexpr auto to_u32string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::u32string>>();

constexpr auto to_wstring
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::wstring>>();


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_TO_STRING_HPP

