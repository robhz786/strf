#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template<typename CharOut>
class char_ptr_writer: public output_buffer<CharOut>
{
    using Traits = std::char_traits<CharOut>;

public:

    using char_type = CharOut;

    char_ptr_writer(CharOut* dest, CharOut* end )
        : output_buffer<CharOut>{dest, end - 1}
        , _begin(dest)
    {
        BOOST_ASSERT(dest < end);
    }

    ~char_ptr_writer()
    {
       * this->pos() = char_type{};
    }

    void recycle() override;

    std::size_t finish()
    {
        return this->pos() - _begin;
    }

private:

    char_type* _begin;
};

template<typename CharOut>
void char_ptr_writer<CharOut>::recycle()
{
    throw std::out_of_range("boost::stringify::write: destination string too small."); 
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char8_t>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<wchar_t>;

#endif


} // namespace detail

#if defined(__cpp_char8_t)

template<std::size_t N>
inline auto write(char8_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char8_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char8_t*, char8_t*>
        (dest, dest + N);
}

#endif

template<std::size_t N>
inline auto write(char (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char*, char*>
        (dest, dest + N);
}

inline auto write(char* dest, char* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char*, char* >
        (dest, end);
}

inline auto write(char* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char*, char*>
        (dest, dest + count);
}


template<std::size_t N>
inline auto write(char16_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char16_t*, char16_t*>
        (dest, dest + N);
}

inline auto write(char16_t* dest, char16_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char16_t*, char16_t*>
        (dest, end);
}

inline auto write(char16_t* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                     , writer, char16_t*, char16_t* >
        (dest, dest + count);
}

template<std::size_t N>
inline auto write(char32_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char32_t*, char32_t*>
        (dest, dest + N);
}

inline auto write(char32_t* dest, char32_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char32_t*, char32_t*>
        (dest, end);
}

inline auto write(char32_t* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char32_t*, char32_t* >
        (dest, dest + count);
}

template<std::size_t N>
inline auto write(wchar_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, wchar_t*, wchar_t*>
        (dest, dest + N);
}

inline auto write(wchar_t* dest, wchar_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                     , writer, wchar_t*, wchar_t*>
        (dest, end);
}


inline auto write(wchar_t* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, wchar_t*, wchar_t* >
        (dest, dest + count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP */

