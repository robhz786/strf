#ifndef BOOST_STRINGIFY_V0_PRINTER_HPP
#define BOOST_STRINGIFY_V0_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <algorithm>
#include <boost/stringify/v0/outbuf.hpp>
#include <boost/stringify/v0/width_t.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

class stringify_error: public std::exception
{
    using std::exception::exception;
};

class encoding_failure: public stringify::stringify_error
{
    using stringify::v0::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: encoding conversion error";
    }
};

inline void throw_encoding_failure()
{
    throw stringify::encoding_failure();
}

class tr_string_syntax_error: public stringify::stringify_error
{
    using stringify::v0::stringify_error::stringify_error;

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

    virtual void print_to(stringify::v0::basic_outbuf<CharOut>& ob) const = 0;
};

namespace detail {

template<std::size_t CharSize>
void write_fill_continuation
    ( stringify::v0::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename stringify::v0::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename stringify::v0::underlying_outbuf<CharSize>::char_type;

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
    ( stringify::v0::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename stringify::v0::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename stringify::v0::underlying_outbuf<CharSize>::char_type;
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
    ( stringify::v0::basic_outbuf<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = typename stringify::v0::underlying_outbuf<sizeof(CharT)>::char_type;
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

template <bool Active>
class width_preview;

template <>
class width_preview<true>
{
public:

    explicit width_preview(stringify::v0::width_t initial_width) noexcept
        : _width(initial_width)
    {}

    width_preview(const width_preview&) = delete;

    constexpr void subtract_width(stringify::v0::width_t w)
    {
        _width -= w;
    }

    constexpr void checked_subtract_width(stringify::v0::width_t w)
    {
        if (w < _width)
        {
            _width -= w;
        }
        else
        {
            _width = 0;
        }
    }

    constexpr void checked_subtract_width(std::ptrdiff_t w)
    {
        if (w < _width.ceil())
        {
            _width -= static_cast<std::int16_t>(w);
        }
        else
        {
            _width = 0;
        }
    }

    constexpr void clear_remaining_width()
    {
        _width = 0;
    }

    constexpr stringify::v0::width_t remaining_width() const
    {
        return _width;
    }

private:

    stringify::v0::width_t _width;
};

template <>
class width_preview<false>
{
public:

    width_preview() noexcept = default;;
    width_preview(const width_preview&) = delete;

    constexpr void subtract_width(stringify::v0::width_t)
    {
    }

    constexpr void checked_subtract_width(stringify::v0::width_t)
    {
    }

    constexpr void checked_subtract_width(std::ptrdiff_t)
    {
    }

    constexpr void clear_remaining_width()
    {
    }

    constexpr stringify::v0::width_t remaining_width() const
    {
        return 0;
    }
};

template <bool Active>
class size_preview;

template <>
class size_preview<true>
{
public:
    explicit size_preview(std::size_t initial_size = 0) noexcept
        : _size(initial_size)
    {
    }

    size_preview(const size_preview&) = delete;

    constexpr void add_size(std::size_t s)
    {
        _size += s;
    }

    constexpr std::size_t get_size() const
    {
        return _size;
    }

private:

    std::size_t _size;
};

template <>
class size_preview<false>
{
public:

    size_preview() noexcept = default;
    size_preview(const size_preview&) = delete;

    constexpr void add_size(std::size_t)
    {
    }

    constexpr std::size_t get_size() const
    {
        return 0;
    }
};

template <bool SizeRequired, bool WidthRequired>
class print_preview
    : public stringify::v0::size_preview<SizeRequired>
    , public stringify::v0::width_preview<WidthRequired>
{
public:

    static constexpr bool size_required = SizeRequired;
    static constexpr bool width_required = WidthRequired;
    static constexpr bool nothing_required = ! SizeRequired && ! WidthRequired;

    template <bool W = WidthRequired>
    constexpr explicit print_preview
        ( std::enable_if_t<W, stringify::v0::width_t> initial_width ) noexcept
        : stringify::v0::width_preview<WidthRequired>{initial_width}
    {
    }

    constexpr print_preview() noexcept = default;
    constexpr print_preview(const print_preview&) = delete;
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_PRINTER_HPP

