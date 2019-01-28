#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/make_destination.hpp>

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
        if (_count_ptr != nullptr)
        {
            *_count_ptr = 0;
        }
    }

    ~narrow_file_writer()
    {
    }

    bool recycle() override;

    stringify::v0::expected<void, std::error_code> finish();

protected:

    std::FILE* _file;
    std::size_t* _count_ptr = nullptr;
};


template <typename CharT>
bool narrow_file_writer<CharT>::recycle()
{
    auto it = this->pos();
    BOOST_ASSERT(_buff <= it && it <= _buff + _buff_size);
    std::size_t count = it - _buff;
    auto count_inc = std::fwrite(_buff, sizeof(CharT), count, _file);

    if (_count_ptr != nullptr)
    {
        *_count_ptr += count_inc;
    }
    this->set_pos(_buff);
    return count == count_inc;
}


template <typename CharT>
stringify::v0::expected<void, std::error_code>
inline narrow_file_writer<CharT>::finish()
{
    if (! this->has_error() && recycle())
    {
        return {};
    }
    return { stringify::v0::unexpect_t{}, this->get_error() };
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

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
        if (_count_ptr != nullptr)
        {
            *_count_ptr = 0;
        }
    }

    bool recycle() override;

    stringify::v0::expected<void, std::error_code> finish()
    {
        if ( ! this->has_error() && recycle())
        {
            return {};
        }
        return { stringify::v0::unexpect_t{}, this->get_error() };
    }

private:

    std::FILE* _file;
    std::size_t* _count_ptr = nullptr;
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE
bool wide_file_writer::recycle()
{
    auto end = this->pos();
    BOOST_ASSERT(_buff <= end && end <= _buff + _buff_size);

    std::size_t count = 0;
    bool good = true;
    for(auto it = _buff ; it != end; ++it, ++count)
    {
        auto ret = std::fputwc(*it, _file);
        if(ret == WEOF)
        {
            good = false;
            this->set_error(std::error_code{errno, std::generic_category()});
            break;
        }
    }
    if (_count_ptr != nullptr)
    {
        *_count_ptr += count;
    }

    this->set_pos(_buff);
    return good;
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

template <typename CharT = char>
inline auto write(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::narrow_file_writer<CharT>;
    return stringify::v0::make_destination<writer>(destination, count);
}

inline auto wwrite(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return stringify::v0::make_destination<writer>(destination, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

