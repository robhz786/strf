#ifndef BOOST_STRINGIFY_V0_PRINTER_HPP
#define BOOST_STRINGIFY_V0_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <algorithm>
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

class tr_string_syntax_error: public boost::stringify::stringify_error
{
    using boost::stringify::v0::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: Tr-string syntax error";
    }
};

constexpr std::size_t min_buff_size = 60;

struct tag
{
    explicit tag() = default;
};

template <typename CharOut>
class output_buffer_base
{
public:


    output_buffer_base(const output_buffer_base&) = delete;
    output_buffer_base(output_buffer_base&&) = delete;

    output_buffer_base& operator=(const output_buffer_base&) = delete;
    output_buffer_base& operator=(output_buffer_base&&) = delete;

    virtual ~output_buffer_base() = default;

    virtual void recycle() = 0;

    constexpr std::size_t buffer_size() const // todo
    // [[ ensures s: s > 0 ]]
    {
        return min_buff_size;
    }

    void ensure(std::size_t size_)
    {
        BOOST_ASSERT(size_ <= buffer_size());
        if (size_ > size())
        {
            this->recycle();
        }
        BOOST_ASSERT(size_ <= size());
    }

    void set_encoding_error()
    {
        throw stringify::v0::encoding_failure();
    }

    CharOut* pos() const noexcept
    {
        return _pos;
    }

    CharOut* end() const noexcept
    {
        return _end;
    }

    std::size_t size() const noexcept
    {
        return _end - _pos;
    }

    void advance(std::size_t n) noexcept
    {
        BOOST_ASSERT(n <= size());
        _pos += n;
    }
    void advance() noexcept
    {
        BOOST_ASSERT(_pos != _end);
        ++ _pos;
    }
    void advance_to(CharOut* p) noexcept
    {
        BOOST_ASSERT(_pos <= p && p <= _end);
        _pos = p;
    }

protected:

    output_buffer_base()
        : _pos(nullptr)
        , _end(nullptr)
    {
    }

    output_buffer_base(CharOut* buff_begin, CharOut* buff_end)
        : _pos(buff_begin)
        , _end(buff_end)
    {
        BOOST_ASSERT(buff_begin <= buff_end);
    }

    output_buffer_base(CharOut* buff_begin, std::size_t buff_size)
        : _pos(buff_begin)
        , _end(buff_begin + buff_size)
    {
    }
    void set_pos(CharOut* p)
    {
        _pos = p;
    }

    void set_end(CharOut* e)
    {
        _end = e;
    }

    virtual void on_error()
    {
    }

private:

    CharOut* _pos;
    CharOut* _end;
};


namespace detail {

template <std::size_t CharSize>
struct underlying_char_type_impl;

template <> struct underlying_char_type_impl<1>{using type = std::uint8_t;};
template <> struct underlying_char_type_impl<2>{using type = char16_t;};
template <> struct underlying_char_type_impl<4>{using type = char32_t;};

} // namespace detail

template <typename CharT>
using underlying_char_type
= typename detail::underlying_char_type_impl<sizeof(CharT)>::type;


template <typename CharOut>
class output_buffer
    : public stringify::v0::output_buffer_base
        < stringify::v0::underlying_char_type<CharOut> >
{
    using underlying_char_type = stringify::v0::underlying_char_type<CharOut>;
    using base_type = stringify::v0::output_buffer_base<underlying_char_type>;

public:

    using char_type = CharOut;

    CharOut* pos() const noexcept
    {
        return reinterpret_cast<CharOut*>(base_type::pos());
    }

    CharOut* end() const noexcept
    {
        return reinterpret_cast<CharOut*>(base_type::end());
    }

    void advance_to(CharOut* p) noexcept
    {
        base_type::advance_to(reinterpret_cast<underlying_char_type*>(p));
    }

    base_type& base()
    {
        return *this;
    }

protected:

    output_buffer() = default;

    output_buffer(CharOut* buff_begin, CharOut* buff_end)
        : base_type( reinterpret_cast<underlying_char_type*>(buff_begin)
                   , reinterpret_cast<underlying_char_type*>(buff_end) )
    {
    }

    output_buffer(CharOut* buff_begin, std::size_t count)
        : base_type(reinterpret_cast<underlying_char_type*>(buff_begin), count)
    {
    }

    void set_pos(CharOut* p)
    {
        base_type::set_pos(reinterpret_cast<underlying_char_type*>(p));
    }

    void set_end(CharOut* e)
    {
        base_type::set_end(reinterpret_cast<underlying_char_type*>(e));
    }
};

template <typename CharOut>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual void write(stringify::v0::output_buffer<CharOut>& ob) const = 0;

    virtual std::size_t necessary_size() const = 0;

    virtual int width(int limit) const = 0;
};

namespace detail {

template<typename CharT>
void write_str_continuation
    ( stringify::v0::output_buffer<CharT>& ob
    , const CharT* str
    , std::size_t len)
{
    using traits = std::char_traits<CharT>;
    std::size_t space = ob.size();
    BOOST_ASSERT(space < len);
    traits::copy(ob.pos(), str, space);
    str += space;
    len -= space;
    ob.advance_to(ob.end());
    while (true)
    {
        ob.recycle();
        space = ob.size();
        if (len <= space)
        {
            traits::copy(ob.pos(), str, len);
            ob.advance(len);
            break;
        }
        traits::copy(ob.pos(), str, space);
        len -= space;
        str += space;
        ob.advance_to(ob.end());
    }
}

template<typename CharT>
inline void write_str( stringify::v0::output_buffer<CharT>& ob
                     , const CharT* str
                     , std::size_t len )
{
    using traits = std::char_traits<CharT>;

    if (len <= ob.size()) // the common case
    {
        traits::copy(ob.pos(), str, len);
        ob.advance(len);
    }
    else
    {
        write_str_continuation(ob, str, len);
    }
}

template<typename CharT>
void write_fill_continuation( stringify::v0::output_buffer_base<CharT>& ob
                            , std::size_t count
                            , CharT ch )
{
    std::size_t space = ob.size();
    BOOST_ASSERT(space < count);
    std::char_traits<CharT>::assign(ob.pos(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    while (true)
    {
        ob.recycle();
        space = ob.size();
        if (count <= space)
        {
            std::char_traits<CharT>::assign(ob.pos(), count, ch);
            ob.advance(count);
            break;
        }
        std::char_traits<CharT>::assign(ob.pos(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
    }
}

template<typename CharT>
inline void write_fill( stringify::v0::output_buffer_base<CharT>& ob
                      , std::size_t count
                      , CharT ch )
{
    if (count <= ob.size()) // the common case
    {
        std::char_traits<CharT>::assign(ob.pos(), count, ch);
        ob.advance(count);
    }
    else
    {
        write_fill_continuation(ob, count, ch);
    }
}

template<typename CharT>
inline void write_fill
    ( stringify::v0::output_buffer<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = stringify::v0::underlying_char_type<CharT>;
    write_fill(ob.base(), count, static_cast<u_char_type>(ch));
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

