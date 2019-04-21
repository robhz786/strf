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

constexpr std::size_t min_buff_size = 60;

struct tag
{
    explicit tag() = default;
};

template <typename CharOut>
class output_buffer
{
public:

    using char_type = CharOut;

    output_buffer(const output_buffer&) = delete;
    output_buffer(output_buffer&&) = delete;

    output_buffer& operator=(const output_buffer&) = delete;
    output_buffer& operator=(output_buffer&&) = delete;

    virtual ~output_buffer() = default;

    virtual bool recycle() = 0;

    void set_error(std::error_code ec);

    void set_error(std::errc e)
    {
        set_error(std::make_error_code(e));
    }

    void set_encoding_error()
    {
        set_error(std::errc::illegal_byte_sequence);
    }

    std::error_code get_error() const noexcept
    {
        return _ec;
    }

    bool has_error() const noexcept
    {
        return _has_error;
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

    output_buffer()
        : _pos(nullptr)
        , _end(nullptr)
    {
    }

    output_buffer(CharOut* buff_begin, CharOut* buff_end)
        : _pos(buff_begin)
        , _end(buff_end)
    {
        BOOST_ASSERT(buff_begin <= buff_end);
    }

    output_buffer(CharOut* buff_begin, std::size_t buff_size)
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
    std::error_code _ec;
    bool _has_error = false;
};

template <typename CharOut>
void output_buffer<CharOut>::set_error(std::error_code ec)
{
    if ( ! _has_error )
    {
        _ec = ec;
        _has_error = true;
        on_error();
    }
}

template <typename CharOut>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual bool write(stringify::v0::output_buffer<CharOut>& ob) const = 0;

    virtual std::size_t necessary_size() const = 0;

    virtual int remaining_width(int w) const = 0;
};

namespace detail {

template<typename CharT>
bool write_str_continuation
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
    while (ob.recycle())
    {
        space = ob.size();
        if (len <= space)
        {
            traits::copy(ob.pos(), str, len);
            ob.advance(len);
            return true;
        }
        traits::copy(ob.pos(), str, space);
        len -= space;
        str += space;
        ob.advance_to(ob.end());
    }
    return false;
}

template<typename CharT>
inline bool write_str
    ( stringify::v0::output_buffer<CharT>& ob
    , const CharT* str
    , std::size_t len )
{
    using traits = std::char_traits<CharT>;

    if (len <= ob.size()) // the common case
    {
        traits::copy(ob.pos(), str, len);
        ob.advance(len);
        return true;
    }
    return write_str_continuation(ob, str, len);
}

template<typename CharT>
bool write_fill_continuation
    ( stringify::v0::output_buffer<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    std::size_t space = ob.size();
    BOOST_ASSERT(space < count);
    std::char_traits<CharT>::assign(ob.pos(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    while (ob.recycle())
    {
        space = ob.size();
        if (count <= space)
        {
            std::char_traits<CharT>::assign(ob.pos(), count, ch);
            ob.advance(count);
            return true;
        }
        std::char_traits<CharT>::assign(ob.pos(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
    }
    return false;
}

template<typename CharT>
inline bool write_fill
    ( stringify::v0::output_buffer<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    if (count <= ob.size()) // the common case
    {
        std::char_traits<CharT>::assign(ob.pos(), count, ch);
        ob.advance(count);
        return true;
    }
    return write_fill_continuation(ob, count, ch);
}

template <typename T, std::size_t N>
struct simple_array;
template <typename T>
struct simple_array<T,1> { T obj0; };
template <typename T>
struct simple_array<T,2> { T obj0;  T obj1; };
template <typename T>
struct simple_array<T,3> { T obj0;  T obj1; T obj2; };
template <typename T>
struct simple_array<T,4> { T obj0;  T obj1; T obj2; T obj3; };


template <typename CharT, std::size_t N>
inline void do_repeat_sequence
    ( CharT* dest
    , std::size_t count
    , simple_array<CharT, N> seq )
{
    std::fill_n(reinterpret_cast<simple_array<CharT, N>*>(dest), count, seq);
}

template <typename CharT, std::size_t N>
bool repeat_sequence_continuation
    ( stringify::v0::output_buffer<CharT>& ob
    , std::size_t count
    , simple_array<CharT, N> seq )
{
    std::size_t space = ob.size() / N;
    BOOST_ASSERT(space < count);

    stringify::v0::detail::do_repeat_sequence(ob.pos(), space, seq);
    count -= space;
    ob.advance_to(ob.end());
    while (ob.recycle())
    {
        std::size_t space = ob.size() / N;
        if (count <= space)
        {
            stringify::v0::detail::do_repeat_sequence(ob.pos(), count, seq);
            ob.advance(count * N);
            return true;
        }
        stringify::v0::detail::do_repeat_sequence(ob.pos(), space, seq);
        count -= space;
        ob.advance_to(ob.end());
    }
    return false;
}


template <typename CharT, std::size_t N>
inline bool repeat_sequence
    ( stringify::v0::output_buffer<CharT>& ob
    , std::size_t count
    , simple_array<CharT, N> seq )
{
    if (count * N <= ob.size())
    {
        stringify::v0::detail::do_repeat_sequence(ob.pos(), count, seq);
        ob.advance(count * N);
        return true;
    }
    return stringify::v0::detail::repeat_sequence_continuation(ob, count, seq);
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
struct asm_string_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct is_asm_string_of
{
    template <typename T>
    using fn = std::is_same<stringify::v0::asm_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_asm_string: std::false_type
{
};

template <typename CharIn>
struct is_asm_string<stringify::v0::is_asm_string_of<CharIn>> : std::true_type
{
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_PRINTER_HPP

