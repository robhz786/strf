//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

class reservation_tester : public strf::basic_outbuf<char>
{
    constexpr static std::size_t _buff_size = strf::min_size_after_recycle<char>();
    char _buff[_buff_size];

public:

    reservation_tester()
        : strf::basic_outbuf<char>{ _buff, _buff + _buff_size }
        , _buff{0}
    {
    }

    reservation_tester(std::size_t size)
        : strf::basic_outbuf<char>{ _buff, _buff + _buff_size }
        , _buff{0}
        , _reserved_size{size}
    {
    }

    reservation_tester(const reservation_tester&) = delete;

#if defined(BOOST_STRINGIFY_NO_CXX17_COPY_ELISION)

    reservation_tester(reservation_tester&& r)
        : reservation_tester(r._reserved_size)
    {
    }

#else

    reservation_tester(reservation_tester&&) = delete;

#endif
    
    void recycle() override
    {
        this->set_pos(_buff);
    }

    std::size_t finish()
    {
        return _reserved_size;
    }

private:

    std::size_t _reserved_size = 0;
};


class reservation_tester_creator
{
public:

    using char_type = char;

    template <typename ... Printers>
    std::size_t write(const Printers& ... printers) const
    {
        reservation_tester ob;
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }

    template <typename ... Printers>
    std::size_t sized_write(std::size_t size, const Printers& ... printers) const
    {
        reservation_tester ob{size};
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }
};

constexpr auto reservation_test()
{
    return strf::dispatcher_no_reserve<reservation_tester_creator>();
}


int main()
{
    // on non-const rval ref
    constexpr std::size_t not_reserved = 0;

    {
        auto size = reservation_test()  ("abcd");
        BOOST_TEST_EQ(size, not_reserved);
    }
    {
        auto size = reservation_test() .reserve(5555) ("abcd");
        BOOST_TEST_EQ(size, 5555);
    }
    {
        auto size = reservation_test() .reserve_calc() ("abcd");
        BOOST_TEST_EQ(size, 4);
    }

    // // on non-const ref

    {
        auto tester = reservation_test();
        auto size = tester ("abcd");
        BOOST_TEST_EQ(size, not_reserved);
    }
    {
        auto tester = reservation_test();
        auto size = tester.reserve(5555) ("abcd");
        BOOST_TEST_EQ(size, 5555);
    }
    {
        auto tester = reservation_test();
        auto size = tester.reserve_calc() ("abcd");
        BOOST_TEST_EQ(size, 4);
    }

    // // on const ref

    {
        const auto tester = reservation_test();
        auto size = tester ("abcd");
        BOOST_TEST_EQ(size, not_reserved);
    }
    {
        const auto tester = reservation_test();
        auto size = tester.reserve(5555) ("abcd");
        BOOST_TEST_EQ(size, 5555);
    }
    {
        const auto tester = reservation_test() .reserve(5555);
        auto size = tester.reserve_calc() ("abcd");
        BOOST_TEST_EQ(size, 4);
    }

    // // on const rval ref

    {
        const auto tester = reservation_test();
        auto size = std::move(tester) ("abcd");
        BOOST_TEST_EQ(size, not_reserved);
    }
    {
        const auto tester = reservation_test();
        auto size = std::move(tester).reserve(5555) ("abcd");
        BOOST_TEST_EQ(size, 5555);
    }
    {
        const auto tester = reservation_test() .reserve(5555);
        auto size = std::move(tester).reserve_calc() ("abcd");
        BOOST_TEST_EQ(size, 4);
    }

    return boost::report_errors();
}
