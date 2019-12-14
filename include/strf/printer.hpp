#ifndef STRF_PRINTER_HPP
#define STRF_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuf.hpp>
#include <strf/width_t.hpp>
#include <strf/detail/standard_lib_functions.hpp>

STRF_NAMESPACE_BEGIN

class stringify_error: public std::exception
{
    using std::exception::exception;
};

class encoding_failure: public strf::stringify_error
{
    using strf::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: encoding conversion error";
    }
};

namespace detail {

inline STRF_HD void handle_encoding_failure()
{
#if defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)
    throw strf::encoding_failure();
#else // defined(__cpp_exceptions)
#ifndef __CUDA_ARCH__
    std::abort();
#else
    asm("trap;");
#endif
#endif // defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)
}


} // namespace detail


template <typename CharOut>
class printer
{
public:

    using char_type = CharOut;

    STRF_HD virtual ~printer()
    {
    }

    STRF_HD virtual void print_to(strf::basic_outbuf<CharOut>& ob) const = 0;
};

namespace detail {

template<std::size_t CharSize>
void STRF_HD write_fill_continuation
    ( strf::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuf<CharSize>::char_type;

    std::size_t space = ob.size();
    STRF_ASSERT(space < count);
    strf::detail::char_assign<char_type>(ob.pos(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    ob.recycle();
    while (ob.good())
    {
        space = ob.size();
        if (count <= space)
        {
            strf::detail::char_assign<char_type>(ob.pos(), count, ch);
            ob.advance(count);
            break;
        }
        strf::detail::char_assign(ob.pos(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
        ob.recycle();
    }
}

template <std::size_t CharSize>
inline STRF_HD void write_fill
    ( strf::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuf<CharSize>::char_type;
    if (count <= ob.size()) // the common case
    {
        strf::detail::char_assign<char_type>(ob.pos(), count, ch);
        ob.advance(count);
    }
    else
    {
        write_fill_continuation(ob, count, ch);
    }
}

template<typename CharT>
inline STRF_HD void write_fill
    ( strf::basic_outbuf<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = typename strf::underlying_outbuf<sizeof(CharT)>::char_type;
    write_fill(ob.as_underlying(), count, static_cast<u_char_type>(ch));
}

} // namespace detail

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
struct tr_string_input_tag: strf::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: strf::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct is_tr_string_of
{
    template <typename T>
    using fn = std::is_same<strf::tr_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_tr_string: std::false_type
{
};

template <typename CharIn>
struct is_tr_string<strf::is_tr_string_of<CharIn>> : std::true_type
{
};

template <bool Active>
class width_preview;

template <>
class width_preview<true>
{
public:

    explicit constexpr STRF_HD width_preview(strf::width_t initial_width) noexcept
        : _width(initial_width)
    {}

    STRF_HD width_preview(const width_preview&) = delete;

    constexpr STRF_HD void subtract_width(strf::width_t w)
    {
        _width -= w;
    }

    constexpr STRF_HD void checked_subtract_width(strf::width_t w)
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

    constexpr STRF_HD void checked_subtract_width(std::ptrdiff_t w)
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

    constexpr STRF_HD void clear_remaining_width()
    {
        _width = 0;
    }

    constexpr STRF_HD strf::width_t remaining_width() const
    {
        return _width;
    }

private:

    strf::width_t _width;
};

template <>
class width_preview<false>
{
public:

    constexpr STRF_HD width_preview() noexcept = default;;
    STRF_HD width_preview(const width_preview&) = delete;

    constexpr STRF_HD void subtract_width(strf::width_t)
    {
    }

    constexpr STRF_HD void checked_subtract_width(strf::width_t)
    {
    }

    constexpr STRF_HD void checked_subtract_width(std::ptrdiff_t)
    {
    }

    constexpr STRF_HD void clear_remaining_width()
    {
    }

    constexpr STRF_HD strf::width_t remaining_width() const
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
    explicit constexpr STRF_HD size_preview(std::size_t initial_size = 0) noexcept
        : _size(initial_size)
    {
    }

    STRF_HD size_preview(const size_preview&) = delete;

    constexpr STRF_HD void add_size(std::size_t s)
    {
        _size += s;
    }

    constexpr STRF_HD std::size_t get_size() const
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

    constexpr STRF_HD size_preview() noexcept = default;
    size_preview(const size_preview&) = delete;

    constexpr STRF_HD void add_size(std::size_t)
    {
    }

    constexpr STRF_HD std::size_t get_size() const
    {
        return 0;
    }
};

template <bool SizeRequired, bool WidthRequired>
class print_preview
    : public strf::size_preview<SizeRequired>
    , public strf::width_preview<WidthRequired>
{
public:

    static constexpr bool size_required = SizeRequired;
    static constexpr bool width_required = WidthRequired;
    static constexpr bool nothing_required = ! SizeRequired && ! WidthRequired;

    template <bool W = WidthRequired>
    STRF_HD constexpr explicit print_preview
        ( std::enable_if_t<W, strf::width_t> initial_width ) noexcept
        : strf::width_preview<WidthRequired>{initial_width}
    {
    }

    constexpr STRF_HD print_preview() noexcept = default;
    constexpr STRF_HD print_preview(const print_preview&) = delete;
};

STRF_NAMESPACE_END

#endif // STRF_PRINTER_HPP
