//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

class reservation_tester : public strf::output_writer<char>
{
public:

    reservation_tester(std::size_t& reserve_size)
        : m_reserve_size(reserve_size)
    {
    }
    
    virtual void set_error(std::error_code) override
    {
    }

    virtual bool good() const override
    {
        return true;
    }

    virtual bool put(const char*, std::size_t) override
    {
        return true;
    }

    virtual bool put(char) override
    {
        return true;
    }

    virtual bool repeat(std::size_t, char) override
    {
        return true;
    }

    virtual bool repeat(std::size_t, char, char) override
    {
        return true;
    }

    virtual bool repeat(std::size_t, char, char, char) override
    {
        return true;
    }

    virtual bool repeat(std::size_t, char, char, char, char) override
    {
        return true;
    }

    void reserve(std::size_t s)
    {
        m_reserve_size = s;
    }

    std::error_code finish()
    {
        return {};
    }

    void finish_throw()
    {
    }
    
private:

    std::size_t & m_reserve_size;
};


auto reservation_test(std::size_t & s)
{
    return strf::make_args_handler<reservation_tester, std::size_t&>(s);
}


int main()
{
    constexpr std::size_t initial_value = std::numeric_limits<std::size_t>::max();

    // on non-const rval ref
   
    {
        std::size_t size{initial_value};
        reservation_test(size) .no_reserve() &= {"abcd"};
        BOOST_TEST(size == initial_value); 
    }
    {
        std::size_t size{initial_value};
        reservation_test(size) .reserve(5555) &= {"abcd"};
        BOOST_TEST(size == 5555); 
    }
    {
        std::size_t size{initial_value};
        reservation_test(size) &= {"abcd"};
        BOOST_TEST(size == 4); 
    }
    {
        std::size_t size{initial_value};
        reservation_test(size) .reserve(5555) .reserve_auto() &= {"abcd"};
        BOOST_TEST(size == 4); 
    }

    // on non-const ref
    
    {
        std::size_t size{initial_value};
        auto tester = reservation_test(size);
        tester.no_reserve() &= {"abcd"};
        BOOST_TEST(size == initial_value); 
    }
    {
        std::size_t size{initial_value};
        auto tester = reservation_test(size);
        tester.reserve(5555) &= {"abcd"};
        BOOST_TEST(size == 5555); 
    }
    {
        std::size_t size{initial_value};
        auto tester = reservation_test(size);
        tester &= {"abcd"};
        BOOST_TEST(size == 4); 
    }
    {
        std::size_t size{initial_value};
        auto tester = reservation_test(size) .reserve(5555);
        tester.reserve_auto() &= {"abcd"};
        BOOST_TEST(size == 4); 
    }

    // on const ref

    {
        std::size_t size{initial_value};
        const auto tester = reservation_test(size);
        tester.no_reserve() &= {"abcd"};
        BOOST_TEST(size == initial_value); 
    }
    {
        std::size_t size{initial_value};
        const auto tester = reservation_test(size);
        tester.reserve(5555) &= {"abcd"};
        BOOST_TEST(size == 5555); 
    }
    {
        std::size_t size{initial_value};
        const auto tester = reservation_test(size) .reserve(5555);
        tester.reserve_auto() &= {"abcd"};
        BOOST_TEST(size == 4); 
    }

    // on const rval ref
    
    {
        std::size_t size{initial_value};
        const auto tester = reservation_test(size);
        std::move(tester).no_reserve() &= {"abcd"};
        BOOST_TEST(size == initial_value); 
    }
    {
        std::size_t size{initial_value};
        const auto tester = reservation_test(size);
        std::move(tester).reserve(5555) &= {"abcd"};
        BOOST_TEST(size == 5555); 
    }
    {
        std::size_t size{initial_value};
        const auto tester = reservation_test(size) .reserve(5555);
        std::move(tester).reserve_auto() &= {"abcd"};
        BOOST_TEST(size == 4); 
    }


    
    return report_errors() || boost::report_errors();
}
