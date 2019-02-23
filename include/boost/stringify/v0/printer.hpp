#ifndef BOOST_STRINGIFY_V0_PRINTER_HPP
#define BOOST_STRINGIFY_V0_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

constexpr std::size_t min_buff_size = 60;

struct tag {};

template <typename CharOut>
class output_buffer
{
public:

    using char_type = CharOut;

    virtual ~output_buffer()
    {
    }

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

    output_buffer(const output_buffer&) = default;

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

inline const std::pair<char32_t*, char32_t*> global_mini_buffer32()
{
    thread_local static char32_t buff[16];
    return {buff, buff + sizeof(buff) / sizeof(buff[0])};
}

template<typename CharIn, typename CharOut>
bool transcode
    ( stringify::v0::output_buffer<CharOut>& ob
    , const CharIn* src
    , const CharIn* src_end
    , const stringify::v0::transcoder<CharIn, CharOut>& tr
    , stringify::v0::encoding_policy epoli )
{
    auto err_hdl = epoli.err_hdl();
    bool allow_surr = epoli.allow_surr();
    stringify::v0::cv_result res;
    do
    {
        auto pos = ob.pos();
        res = tr.transcode(&src, src_end, &pos, ob.end(), err_hdl, allow_surr);
        ob.advance_to(pos);
        if (res == stringify::v0::cv_result::success)
        {
            return true;
        }
        if (res == stringify::v0::cv_result::invalid_char)
        {
            ob.set_encoding_error();
            return false;
        }
    } while(ob.recycle());
    return false;
}

template<typename CharIn, typename CharOut>
bool decode_encode
    ( stringify::v0::output_buffer<CharOut>& ob
    , const CharIn* src
    , const CharIn* src_end
    , stringify::v0::encoding<CharIn> src_encoding
    , stringify::v0::encoding<CharOut> dest_encoding
    , stringify::v0::encoding_policy epoli )
{
    auto err_hdl = epoli.err_hdl();
    bool allow_surr = epoli.allow_surr();
    const auto buff32 = global_mini_buffer32();
    char32_t* const buff32_begin = buff32.first;
    char32_t* const buff32_end = buff32.second;
    stringify::v0::cv_result res1;
    do
    {
        char32_t* buff32_it = buff32_begin;
        res1 = src_encoding.to_u32().transcode( &src, src_end
                                              , &buff32_it, buff32_end
                                              , err_hdl, allow_surr );
        if (res1 == stringify::v0::cv_result::invalid_char)
        {
            ob.set_encoding_error();
            return false;
        }
        auto pos = ob.pos();
        const char32_t* buff32_it2 = buff32_begin;
        auto res2 = dest_encoding.from_u32().transcode( &buff32_it2, buff32_it
                                                      , &pos, ob.end()
                                                      , err_hdl, allow_surr );
        ob.advance_to(pos);
        while (res2 == stringify::v0::cv_result::insufficient_space)
        {
            if ( ! ob.recycle())
            {
                return false;
            }
            pos = ob.pos();
            res2 = dest_encoding.from_u32().transcode( &buff32_it2, buff32_it
                                                     , &pos, ob.end()
                                                     , err_hdl, allow_surr );
            ob.advance_to(pos);
        }
        if (res2 == stringify::v0::cv_result::invalid_char)
        {
            ob.set_encoding_error();
            return false;
        }
    } while (res1 == stringify::v0::cv_result::insufficient_space);

    return true;
}

template<typename CharIn, typename CharOut>
inline std::size_t decode_encode_size
    ( const CharIn* src
    , const CharIn* src_end
    , stringify::v0::encoding<CharIn> src_encoding
    , stringify::v0::encoding<CharOut> dest_encoding
    , stringify::v0::encoding_policy epoli )
{
    auto err_hdl = epoli.err_hdl();
    bool allow_surr = epoli.allow_surr();
    auto buff32 = global_mini_buffer32();
    char32_t* const buff32_begin = buff32.first;
    std::size_t count = 0;
    stringify::v0::cv_result res_dec;
    do
    {
        buff32.first = buff32_begin;
        res_dec = src_encoding.to_u32().transcode( &src, src_end
                                                 , &buff32.first, buff32.second
                                                 , err_hdl, allow_surr );
        count += dest_encoding.from_u32().necessary_size( buff32_begin, buff32.first
                                                        , err_hdl, allow_surr );
    } while(res_dec == stringify::v0::cv_result::insufficient_space);

    return count;
}

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

template<typename CharT>
bool do_write_fill
    ( stringify::v0::encoding<CharT> encoding
    , stringify::v0::output_buffer<CharT>& ob
    , std::size_t count
    , char32_t ch
    , stringify::v0::encoding_policy epoli )
{
    auto err_hdl = epoli.err_hdl();
    bool allow_surr = epoli.allow_surr();
    do
    {
        auto pos = ob.pos();
        auto res = encoding.encode_fill
            (&pos, ob.end(), count, ch, err_hdl, allow_surr);
        if (res == stringify::v0::cv_result::success)
        {
            ob.advance_to(pos);
            return true;
        }
        ob.advance_to(pos);
        if (res == stringify::v0::cv_result::invalid_char)
        {
            ob.set_encoding_error();
            return false;
        }
        BOOST_ASSERT(res == stringify::v0::cv_result::insufficient_space);
    } while (ob.recycle());
    return false;
}

template<typename CharT>
inline bool write_fill
    ( stringify::v0::encoding<CharT> encoding
    , stringify::v0::output_buffer<CharT>& buff
    , std::size_t count
    , char32_t ch
    , stringify::v0::encoding_policy epoli )
{
    return ( ch >= encoding.u32equivalence_begin()
          && ch < encoding.u32equivalence_end()
          && (epoli.allow_surr() || (ch >> 11 != 0x1B)) )
        ? stringify::v0::detail::write_fill( buff
                                           , count
                                           , (CharT)ch )
        : stringify::v0::detail::do_write_fill( encoding
                                              , buff
                                              , count
                                              , ch
                                              , epoli );
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

