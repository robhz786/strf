#ifndef BOOST_STRINGIFY_V0_DETAIL_TRANSCODER_HPP
#define BOOST_STRINGIFY_V0_DETAIL_TRANSCODER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharOut>
class length_accumulator: public stringify::v0::u32output
{
public:

    length_accumulator
        ( const stringify::v0::encoder<CharOut>& encoder )
        : m_encoder(encoder)
    {
    }

    bool put(char32_t ch) override
    {
        m_length += m_encoder.length(ch);
        return true;
    }

    void set_error(std::error_code) override
    {
    }

    std::size_t get_length() const
    {
        return m_length;
    }

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    std::size_t m_length = 0;
};


template <typename CharOut>
class encoder_adapter: public stringify::v0::u32output
{
public:
    encoder_adapter
        ( const stringify::v0::encoder<CharOut>& encoder
        , stringify::v0::output_writer<CharOut>& destination
        )
        : m_encoder(encoder)
        , m_destination(destination)
    {
    }

    bool put(char32_t ch) override
    {
        return m_encoder.encode(m_destination, 1, ch);
    }

    void set_error(std::error_code err) override
    {
        m_destination.set_error(err);
    }

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    stringify::v0::output_writer<CharOut>& m_destination;
};

template <typename CharOut>
class u32writer: public stringify::v0::u32output
{
public:

    u32writer(stringify::v0::output_writer<CharOut>& destination)
        : m_destination(destination)
    {
    }

    bool put(char32_t ch) override
    {
        return m_destination.put(static_cast<CharOut>(ch));
    }

    void set_error(std::error_code err) override
    {
        m_destination.set_error(err);
    }

private:

    stringify::v0::output_writer<CharOut>& m_destination;
};



template <typename CharIn, typename CharOut>
class transcoder
{
public:

    transcoder
        ( const stringify::v0::width_calculator& wcalc
        , const stringify::v0::decoder<CharIn>& decoder
        , const stringify::v0::encoder<CharOut>& encoder
        ) noexcept
        : m_wcalc(wcalc)
        , m_decoder(decoder)
        , m_encoder(encoder)
    {
    }

    void write
        ( stringify::v0::output_writer<CharOut>& dest
        , const CharIn* begin
        , const CharIn* end
        ) const
    {
        if (shall_skip_encoder_and_decoder())
        {
            dest.put(reinterpret_cast<const CharOut*>(begin), end - begin);
        }
        else if(shall_skip_decoder())
        {
            for(auto it = begin; it < end; ++it)
            {
                if( ! m_encoder.encode(dest, 1, *it))
                {
                    return;
                }
            }
        }
        else if(shall_skip_encoder())
        {
            stringify::v0::detail::u32writer<CharOut> writer{dest};
            m_decoder.decode(writer, begin, end);
        }
        else
        {
            //decode and encode
            stringify::v0::detail::encoder_adapter<CharOut> c32w{m_encoder, dest};
            m_decoder.decode(c32w, begin, end);
        }
    }

    auto remaining_width(int w, const CharIn* begin, const CharIn* end) const
    {
        if (shall_skip_encoder())
        {
            auto b = reinterpret_cast<const char32_t*>(begin);
            auto e = reinterpret_cast<const char32_t*>(end);
            return m_wcalc.remaining_width(w, b, e);
        }
        return m_wcalc.remaining_width(w, begin, end, m_decoder);
    }

    std::size_t length(const CharIn* begin, const CharIn* end) const
    {

        if(shall_skip_encoder_and_decoder())
        {
            return end - begin;
        }
        else if(shall_skip_decoder())
        {
            std::size_t len = 0;
            for(auto it = begin; it != end; ++it)
            {
                len += m_encoder.length(*it);
            }
            return len;
        }
        else if(shall_skip_encoder())
        {
            stringify::v0::u32encoder<char32_t> dummy_encoder;
            stringify::v0::detail::length_accumulator<char32_t> acc{dummy_encoder};
            m_decoder.decode(acc, begin, end);
            return acc.get_length();
        }
        else
        {
            stringify::v0::detail::length_accumulator<CharOut> acc{m_encoder};
            m_decoder.decode(acc, begin, end);
            return acc.get_length();
        }
    }

    const stringify::v0::width_calculator& m_wcalc;
    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::encoder<CharOut>& m_encoder;

private:

    constexpr static bool m_assume_wchar_encoding = true;

    constexpr bool shall_skip_encoder_and_decoder() const
    {
        return
            std::is_same<CharIn, CharOut>::value
            || (m_assume_wchar_encoding && sizeof(CharIn) == sizeof(CharOut));
    }

    constexpr bool shall_skip_decoder() const
    {
        return std::is_same<CharIn, char32_t>::value
            || (m_assume_wchar_encoding && sizeof(CharIn) == 4);
    }

    constexpr bool shall_skip_encoder() const
    {
        return std::is_same<CharOut, char32_t>::value
            || (m_assume_wchar_encoding && sizeof(CharOut) == 4);
    }
};


} // namespace detail
BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_TRANSCODER_HPP

