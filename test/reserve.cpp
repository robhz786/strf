//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

class reservation_tester : public strf::output_buffer<char>
{
    constexpr static std::size_t _buff_size = strf::min_buff_size;
    char _buff[_buff_size];

public:

    reservation_tester()
        : output_buffer<char>{ _buff, _buff + _buff_size }
    {
    }

    void reserve(std::size_t s)
    {
        m_reserve_size = s;
    }

    bool recycle() override
    {
        this->set_pos(_buff);
        return true;
    }

    std::size_t finish()
    {
        return m_reserve_size;
    }

private:

    std::size_t m_reserve_size = std::numeric_limits<std::size_t>::max();
};


auto reservation_test()
{
    return strf::make_destination<reservation_tester>();
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
