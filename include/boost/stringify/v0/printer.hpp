#ifndef BOOST_STRINGIFY_V0_PRINTER_HPP
#define BOOST_STRINGIFY_V0_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <algorithm>
#include <boost/outbuf.hpp>
#include <boost/stringify/v0/config.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

class stringify_error: public std::exception
{
    using std::exception::exception;
};

class encoding_failure: public boost::stringify::stringify_error
{
    using boost::stringify::v0::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: encoding conversion error";
    }
};

inline void throw_encoding_failure()
{
    throw boost::stringify::encoding_failure();
}

class tr_string_syntax_error: public boost::stringify::stringify_error
{
    using boost::stringify::v0::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: Tr-string syntax error";
    }
};

struct tag
{
    explicit tag() = default;
};

template <typename CharOut>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual void write(boost::basic_outbuf<CharOut>& ob) const = 0;

    virtual std::size_t necessary_size() const = 0;

    virtual int width(int limit) const = 0;
};

namespace detail {

template<std::size_t CharSize>
void write_fill_continuation
    ( boost::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename boost::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename boost::underlying_outbuf<CharSize>::char_type;

    std::size_t space = ob.size();
    BOOST_ASSERT(space < count);
    std::char_traits<char_type>::assign(ob.pos(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    ob.recycle();
    while (ob.good())
    {
        space = ob.size();
        if (count <= space)
        {
            std::char_traits<char_type>::assign(ob.pos(), count, ch);
            ob.advance(count);
            break;
        }
        std::char_traits<char_type>::assign(ob.pos(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
        ob.recycle();
    }
}

template <std::size_t CharSize>
inline void write_fill
    ( boost::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename boost::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename boost::underlying_outbuf<CharSize>::char_type;
    if (count <= ob.size()) // the common case
    {
        std::char_traits<char_type>::assign(ob.pos(), count, ch);
        ob.advance(count);
    }
    else
    {
        write_fill_continuation(ob, count, ch);
    }
}

template<typename CharT>
inline void write_fill
    ( boost::basic_outbuf<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = typename boost::underlying_outbuf<sizeof(CharT)>::char_type;
    write_fill(ob.as_underlying(), count, static_cast<u_char_type>(ch));
}

} // namespace detail

template <typename T>
struct identity
{
    using type = T;
};

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;

template <typename CharIn>
struct tr_string_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct is_tr_string_of
{
    template <typename T>
    using fn = std::is_same<stringify::v0::tr_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_tr_string: std::false_type
{
};

template <typename CharIn>
struct is_tr_string<stringify::v0::is_tr_string_of<CharIn>> : std::true_type
{
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_PRINTER_HPP

