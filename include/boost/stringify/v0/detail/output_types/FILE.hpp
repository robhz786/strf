#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT>
class narrow_file_writer final: public stringify::v0::output_buffer<CharT>
{
public:
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;

private:
    CharT _buff[_buff_size];

public:

    using char_type = CharT;

    narrow_file_writer(std::FILE* file, std::size_t* count)
        : output_buffer<CharT>{ _buff, _buff + _buff_size }
        , _file(file)
        , _count_ptr(count)
    {
    }

    ~narrow_file_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    void recycle() override;

    std::size_t finish();

private:

    std::FILE* _file;
    std::size_t _count = 0;
    std::size_t* _count_ptr = nullptr;
};


template <typename CharT>
void narrow_file_writer<CharT>::recycle()
{
    auto it = this->pos();
    BOOST_ASSERT(_buff <= it && it <= _buff + _buff_size);
    this->set_pos(_buff);

    std::size_t count = it - _buff;
    auto count_inc = std::fwrite(_buff, sizeof(CharT), count, _file);
    _count += count_inc;
    if (count_inc < count)
    {
        throw std::system_error{errno, std::generic_category()};
    }
}


template <typename CharT>
inline std::size_t narrow_file_writer<CharT>::finish()
{
    if (this->pos() != _buff)
    {
        std::size_t count = this->pos() - _buff;
        auto count_inc = std::fwrite(_buff, sizeof(CharT), count, _file);
        _count += count_inc;
    }
    return _count;
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char8_t>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<wchar_t>;

#endif

class wide_file_writer final: public stringify::v0::output_buffer<wchar_t>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    wchar_t _buff[_buff_size];

public:

    using char_type = wchar_t;

    wide_file_writer(std::FILE* file, std::size_t* count)
        : output_buffer<wchar_t>{ _buff, _buff + _buff_size }
        , _file(file)
        , _count_ptr(count)
    {
    }

    ~wide_file_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    void recycle() override;

    std::size_t finish()
    {
        recycle();
        return _count;
    }

private:

    std::FILE* _file;
    std::size_t _count = 0;
    std::size_t* _count_ptr = nullptr;
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE
void wide_file_writer::recycle()
{
    auto end = this->pos();
    BOOST_ASSERT(_buff <= end && end <= _buff + _buff_size);
    this->set_pos(_buff);

    for(auto it = _buff ; it != end; ++it, ++_count)
    {
        auto ret = std::fputwc(*it, _file);
        if(ret == WEOF)
        {
            throw std::system_error{errno, std::generic_category()};
        }
    }
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

template <typename CharT = char>
inline auto write(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::narrow_file_writer<CharT>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE*, std::size_t* >
        (destination, count);
}

inline auto wwrite(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE*, std::size_t* >
        (destination, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

