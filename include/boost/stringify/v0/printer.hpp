#ifndef BOOST_STRINGIFY_V0_PRINTER_HPP
#define BOOST_STRINGIFY_V0_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/expected.hpp>
#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

constexpr std::size_t min_buff_size = 60;

struct tag {};

template <typename CharOut>
struct output_buffer
{
    output_buffer& operator=(const output_buffer& other)
    {
        it = other.it;
        end = other.end;
        return *this;
    }

    CharOut* it;
    CharOut* end;
};

template <typename CharOut>
class buffer_recycler
{
public:

    using char_type = CharOut;

    virtual ~buffer_recycler()
    {
    }

    virtual bool recycle(stringify::v0::output_buffer<CharOut>&) = 0;

    void set_error(std::error_code ec)
    {
        if ( ! _has_error )
        {
            _ec = ec;
            _has_error = true;
        }
    }

    void set_error(std::errc e)
    {
        set_error(std::make_error_code(e));
    }

    void set_encoding_error()
    {
        set_error(std::errc::illegal_byte_sequence);
    }

    std::error_code get_error() const
    {
        return _ec;
    }

    bool has_error() const
    {
        return _has_error;
    }

private:

    std::error_code _ec;
    bool _has_error = false;
};

template <typename CharOut>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual bool write
        ( stringify::v0::output_buffer<CharOut>& buff
        , stringify::v0::buffer_recycler<CharOut>& recycler ) const = 0;

    virtual std::size_t necessary_size() const = 0;

    virtual int remaining_width(int w) const = 0;
};

// inline std::error_code encoding_error()
// {
//     return std::make_error_code(std::errc::illegal_byte_sequence);
// }

namespace detail {

inline const output_buffer<char32_t> global_mini_buffer32()
{
    thread_local static char32_t buff[16];
    return {buff, buff + sizeof(buff) / sizeof(buff[0])};
}


template<typename CharIn, typename CharOut>
bool transcode
    ( stringify::v0::output_buffer<CharOut>& buff
    , stringify::v0::buffer_recycler<CharOut>& recycler
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
        res = tr.transcode(&src, src_end, &buff.it, buff.end, err_hdl, allow_surr);
        if (res == stringify::v0::cv_result::success)
        {
            return true;
        }
        if (res == stringify::v0::cv_result::invalid_char)
        {
            recycler.set_encoding_error();
            return false;
        }
    } while(recycler.recycle(buff));
    return false;
}

template<typename CharIn, typename CharOut>
bool decode_encode
    ( stringify::v0::output_buffer<CharOut>& buff
    , stringify::v0::buffer_recycler<CharOut>& recycler
    , const CharIn* src
    , const CharIn* src_end
    , stringify::v0::encoding<CharIn> src_encoding
    , stringify::v0::encoding<CharOut> dest_encoding
    , stringify::v0::encoding_policy epoli )
{
    auto err_hdl = epoli.err_hdl();
    bool allow_surr = epoli.allow_surr();
    auto buff32 = global_mini_buffer32();
    char32_t* const buff32_begin = buff32.it;
    stringify::v0::cv_result res1;
    do
    {
        res1 = src_encoding.to_u32().transcode( &src, src_end
                                              , &buff32.it, buff32.end
                                              , err_hdl, allow_surr );
        if (res1 == stringify::v0::cv_result::invalid_char)
        {
            recycler.set_error(std::make_error_code(std::errc::result_out_of_range));
            return false;
        }
        const char32_t* buff32_it2 = buff32_begin;
        auto res2 = dest_encoding.from_u32().transcode( &buff32_it2, buff32.it
                                                      , &buff.it, buff.end
                                                      , err_hdl, allow_surr );
        while (res2 == stringify::v0::cv_result::insufficient_space)
        {
            if ( ! recycler.recycle(buff))
            {
                return false;
            }
            res2 = dest_encoding.from_u32().transcode( &buff32_it2, buff32.it
                                                     , &buff.it, buff.end
                                                     , err_hdl, allow_surr );
        }
        if (res2 == stringify::v0::cv_result::invalid_char)
        {
            recycler.set_encoding_error();
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
    char32_t* const buff32_begin = buff32.it;
    std::size_t count = 0;
    stringify::v0::cv_result res_dec;
    do
    {
        buff32.it = buff32_begin;
        res_dec = src_encoding.to_u32().transcode( &src, src_end
                                                 , &buff32.it, buff32.end
                                                 , err_hdl, allow_surr );
        count += dest_encoding.from_u32().necessary_size( buff32_begin, buff32.it
                                                        , err_hdl, allow_surr );
    } while(res_dec == stringify::v0::cv_result::insufficient_space);

    return count;
}

template<typename CharT>
inline bool write_str
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::buffer_recycler<CharT>& recycler
    , const CharT* str
    , std::size_t len )
{
    auto ob_ = ob;
    using traits = std::char_traits<CharT>;
    do
    {
        std::size_t space = ob_.end - ob_.it;
        if (len <= space)
        {
            traits::copy(ob_.it, str, len);
            ob.it = ob_.it + len;
            ob.end = ob_.end;
            return true;
        }
        traits::copy(ob_.it, str, space);
        len -= space;
        str += space;
        ob_.it += space;
    } while (recycler.recycle(ob_));
    ob = ob_;
    return false;
}

template<typename CharT>
inline bool write_fill
    ( stringify::v0::output_buffer<CharT>& buff
    , stringify::v0::buffer_recycler<CharT>& recycler
    , std::size_t count
    , CharT ch )
{
    auto ob = buff;
    do
    {
        std::size_t space = buff.end - buff.it;
        if (count <= space)
        {
            std::char_traits<CharT>::assign(buff.it, count, ch);
            buff.it = ob.it + count;
            buff.end = ob.end;
            return true;
        }
        std::char_traits<CharT>::assign(buff.it, space, ch);
        count -= space;
        ob.it += space;
    } while (recycler.recycle(ob));
    buff = ob;
    return false;
}

template<typename CharT>
bool do_write_fill
    ( stringify::v0::encoding<CharT> encoding
    , stringify::v0::output_buffer<CharT>& out_buff
    , stringify::v0::buffer_recycler<CharT>& recycler
    , std::size_t count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    auto ob = out_buff;
    do
    {
        auto res = encoding.encode_fill(&ob.it, ob.end, count, ch, err_hdl);
        if (res == stringify::v0::cv_result::success)
        {
            out_buff.it = ob.it;
            out_buff.end = ob.end;
            return true;
        }
        if (res == stringify::v0::cv_result::invalid_char)
        {
            recycler.set_encoding_error();
            return false;
        }
        BOOST_ASSERT(res == stringify::v0::cv_result::insufficient_space);
    } while (recycler.recycle(ob));
    out_buff = ob;
    return false;
}

template<typename CharT>
inline bool write_fill
    ( stringify::v0::encoding<CharT> encoding
    , stringify::v0::output_buffer<CharT>& buff
    , stringify::v0::buffer_recycler<CharT>& recycler
    , std::size_t count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    return  ( ch >= encoding.u32equivalence_begin()
           && ch < encoding.u32equivalence_end()
            ? stringify::v0::detail::write_fill( buff
                                               , recycler
                                               , count
                                               , (CharT)ch )
            : stringify::v0::detail::do_write_fill( encoding
                                                  , buff
                                                  , recycler
                                                  , count
                                                  , ch
                                                  , err_hdl ) );
}

} // namespace detail

// template <typename CharOut>
// class printers_receiver
// {
// public:

//     virtual ~printers_receiver()
//     {
//     }

//     virtual bool put(const stringify::v0::printer<CharOut>& ) = 0;
// };

// namespace detail {

// template <typename CharOut>
// class width_subtracter: public printers_receiver<CharOut>
// {
// public:

//     width_subtracter(int w)
//         : _width(w)
//     {
//     }

//     bool put(const stringify::v0::printer<CharOut>& p) override;

//     int remaining_width() const
//     {
//         return _width;
//     }

// private:

//     int _width;
// };

// template <typename CharOut>
// bool width_subtracter<CharOut>::put(const stringify::v0::printer<CharOut>& p)
// {
//     _width = p.remaining_width(_width);
//     return _width > 0;
// }

// template <typename CharOut>
// class necessary_size_sum: public printers_receiver<CharOut>
// {
// public:

//     necessary_size_sum() = default;

//     bool put(const stringify::v0::printer<CharOut>& p) override;

//     std::size_t accumulated_length() const
//     {
//         return _len;
//     }

// private:

//     std::size_t _len = 0;
// };

// template <typename CharOut>
// bool necessary_size_sum<CharOut>::put(const stringify::v0::printer<CharOut>& p)
// {
//     _len += p.necessary_size();
//     return true;
// }

// template <typename CharOut>
// class serial_writer: public printers_receiver<CharOut>
// {
// public:

//     serial_writer() = default;

//     bool put(const stringify::v0::printer<CharOut>& p) override;
// };

// template <typename CharOut>
// bool serial_writer<CharOut>::put(const stringify::v0::printer<CharOut>& p)
// {
//     p.write();
//     return true;
// }

// } // namespace detail

// template <typename CharOut>
// class dynamic_join_printer: public stringify::v0::printer<CharOut>
// {
// public:

//     dynamic_join_printer(stringify::v0::output_writer<CharOut>& out)
//         : m_out(out)
//     {
//     }

//     std::size_t necessary_size() const override;

//     void write() const override;

//     int remaining_width(int w) const override;

// protected:

//     virtual stringify::v0::alignment_format::fn<void> formatting() const;

//     virtual void compose(stringify::v0::printers_receiver<CharOut>& out) const = 0;

// private:

//     void write_with_fill(int fillcount) const;

//     void write_without_fill() const;

//     stringify::v0::output_writer<CharOut>& m_out;
// };

// template <typename CharOut>
// stringify::v0::alignment_format::fn<void>
// dynamic_join_printer<CharOut>::formatting() const
// {
//     return {};
// }

// template <typename CharOut>
// std::size_t dynamic_join_printer<CharOut>::necessary_size() const
// {
//     std::size_t fill_len = 0;
//     const auto fmt = formatting();
//     if(fmt.width() > 0)
//     {
//         stringify::v0::detail::width_subtracter<CharOut> wds{fmt.width()};
//         compose(wds);
//         std::size_t fillcount = wds.remaining_width();
//         fill_len = m_out.necessary_size(fmt.fill()) * fillcount;
//     }

//     stringify::v0::detail::necessary_size_sum<CharOut> s;
//     compose(s);
//     return s.accumulated_length() + fill_len;
// }

// template <typename CharOut>
// int dynamic_join_printer<CharOut>::remaining_width(int w) const
// {
//     const auto fmt_width = formatting().width();
//     if (fmt_width > w)
//     {
//         return 0;
//     }

//     stringify::v0::detail::width_subtracter<CharOut> s{w};
//     compose(s);
//     int rw = s.remaining_width();
//     return (w - rw < fmt_width) ? (w - fmt_width) : rw;
// }

// template <typename CharOut>
// void dynamic_join_printer<CharOut>::write() const
// {
//     auto fmt = formatting();
//     auto fillcount = fmt.width();
//     if(fillcount > 0)
//     {
//         stringify::v0::detail::width_subtracter<CharOut> wds{fillcount};
//         compose(wds);
//         fillcount = wds.remaining_width();
//     }
//     if(fillcount > 0)
//     {
//         write_with_fill(fillcount);
//     }
//     else
//     {
//         write_without_fill();
//     }
// }

// template <typename CharOut>
// void dynamic_join_printer<CharOut>::write_without_fill() const
// {
//     stringify::v0::detail::serial_writer<CharOut> s;
//     compose(s);
// }

// template <typename CharOut>
// void dynamic_join_printer<CharOut>::write_with_fill(int fillcount) const
// {
//     auto fmt = formatting();
//     switch (fmt.alignment())
//     {
//         case stringify::v0::alignment::left:
//         {
//             write_without_fill();
//             m_out.put32(fillcount, fmt.fill());
//             break;
//         }
//         case stringify::v0::alignment::center:
//         {
//             auto halfcount = fillcount / 2;
//             m_out.put32(halfcount, fmt.fill());
//             write_without_fill();
//             m_out.put32(fillcount - halfcount, fmt.fill());
//             break;
//         }
//         //case stringify::v0::alignment::internal:
//         //case stringify::v0::alignment::right:
//         default:
//         {
//             m_out.put32(fillcount, fmt.fill());
//             write_without_fill();
//         }
//     }
// }


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

