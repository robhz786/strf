#ifndef STRF_DETAIL_PRINTER_VARIANT_HPP
#define STRF_DETAIL_PRINTER_VARIANT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <type_traits>
#include <algorithm>

namespace strf {

namespace detail {

template <typename Printer0, typename ... Printers>
class printer_variant
{
    static constexpr std::size_t _max_size
    = std::max({sizeof(Printer0), sizeof(Printers)...});

public:

    static constexpr std::size_t char_size = Printer0::char_size;

    template <typename P, typename ... Args>
    printer_variant(strf::tag<P>, Args&& ... args)
    {
        static_assert( std::is_base_of<strf::printer<char_size>, P>::value
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

    operator const strf::printer<char_size>& () const
    {
        return *_get_ptr();
    }

    const strf::printer<char_size>& get() const
    {
        return *_get_ptr();
    }

private:

    const strf::printer<char_size>* _get_ptr() const
    {
        return reinterpret_cast<const strf::printer<char_size>*>(&_pool);
    }

    using _storage_type = typename std::aligned_storage_t
        < _max_size, alignof(strf::printer<char_size>)>;

    _storage_type _pool;
};

} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_PRINTER_VARIANT_HPP

