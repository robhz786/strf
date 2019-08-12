//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

class reservation_tester : public boost::basic_outbuf<char>
{
    constexpr static std::size_t _buff_size = boost::min_size_after_recycle<char>();
    char _buff[_buff_size];

public:

    reservation_tester()
        : boost::basic_outbuf<char>{ _buff, _buff + _buff_size }
        , _buff{0}
    {
    }

    reservation_tester(const reservation_tester&) = delete;

    reservation_tester(reservation_tester&&) = delete;

    void reserve(std::size_t s)
    {
        _reserve_size = s;
    }

    void recycle() override
    {
        this->set_pos(_buff);
    }

    std::size_t finish()
    {
        return _reserve_size;
    }

private:

    std::size_t _reserve_size = std::numeric_limits<std::size_t>::max();
};


constexpr auto reservation_test()
{
    return strf::dispatcher<strf::facets_pack<>, reservation_tester>();
}


int main()
{
    // on non-const rval ref
    constexpr std::size_t not_reserved = std::numeric_limits<std::size_t>::max();

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

    // on non-const ref

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

    // on const ref

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

    // on const rval ref

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
