#ifndef STRF_DETAIL_PRINTER_VARIANT_HPP
#define STRF_DETAIL_PRINTER_VARIANT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <type_traits>
#include <algorithm>

STRF_NAMESPACE_BEGIN

namespace detail {

template <typename Printer0, typename ... Printers>
class printer_variant
{
    static constexpr std::size_t _max_size
    = std::max({sizeof(Printer0), sizeof(Printers)...});

public:

    using char_type = typename Printer0::char_type;

    template <typename P, typename ... Args>
    printer_variant(strf::tag<P>, Args&& ... args)
    {
        static_assert( std::is_base_of<strf::printer<char_type>, P>::value
                     , "Invalid printer type" );
        static_assert(sizeof(P) <= _max_size, "Invalid printer type");
        new ((void*)&_pool) P{std::forward<Args>(args)...};
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    printer_variant(printer_variant&&);

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    ~printer_variant()
    {
        _get_ptr()->~printer();
    }

    operator const strf::printer<char_type>& () const
    {
        return *_get_ptr();
    }

    const strf::printer<char_type>& get() const
    {
        return *_get_ptr();
    }

private:

    const strf::printer<char_type>* _get_ptr() const
    {
        return reinterpret_cast<const strf::printer<char_type>*>(&_pool);
    }

    using _storage_type = typename std::aligned_storage_t
        < _max_size, alignof(strf::printer<char_type>)>;

    _storage_type _pool;
};

} // namespace detail

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_PRINTER_VARIANT_HPP

