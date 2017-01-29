//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include <vector>

template <int I> struct t
{
    static constexpr int value = I;
};

template<typename T> struct le1
{
    static constexpr bool value = (T::value <= 1); 
};

template<typename T> struct le2
{
    static constexpr bool value = (T::value <= 2); 
};

template<typename T> struct le3
{
    static constexpr bool value = (T::value <= 3); 
};

template<typename T> struct le4
{
    static constexpr bool value = (T::value <= 4); 
};

template<typename T> struct le5
{
    static constexpr bool value = (T::value <= 5); 
};

auto f1 = boost::stringify::width_if<le1>(1);
auto f2 = boost::stringify::width_if<le2>(2);
auto f3 = boost::stringify::width_if<le3>(3);
auto f4 = boost::stringify::width_if<le4>(4);
auto f5 = boost::stringify::width_if<le5>(5);

template <typename FTuple>
std::vector<int> digest(const FTuple& fmt)
{
    namespace strf = boost::stringify;
    return std::vector<int> 
    {
        strf::get<strf::width_tag, t<1>>(fmt).width(),
        strf::get<strf::width_tag, t<2>>(fmt).width(),
        strf::get<strf::width_tag, t<3>>(fmt).width(),
        strf::get<strf::width_tag, t<4>>(fmt).width(),
        strf::get<strf::width_tag, t<5>>(fmt).width()
    };
}

std::vector<int> expected = {1, 2, 3, 4, 5};
    

int main()
{
    namespace strf = boost::stringify;

    {
        auto fmt = strf::make_ftuple(f1, f2, f3, f4, f5);
        BOOST_TEST(digest(fmt) == expected);
    }

    {
        auto fmt = strf::make_ftuple(f1, f1, f2, f1, f2, f3, f4, f5, f5);
        BOOST_TEST(digest(fmt) == expected);
    }
    
    {
        auto fmt = strf::make_ftuple
            (strf::make_ftuple(f1), f2, f3, f4, f5);
        BOOST_TEST(digest(fmt) == expected);
    }

    {
        auto fmt = strf::make_ftuple
            (f1, f2, f3, f4, strf::make_ftuple(f5));
        BOOST_TEST(digest(fmt) == expected);
    }

    {
        auto fmt = strf::make_ftuple
            (f1, strf::make_ftuple(f2, f3), f4, f5);
        BOOST_TEST(digest(fmt) == expected);
    }

    {
        auto fmt = strf::make_ftuple
            ( strf::make_ftuple()
            , strf::make_ftuple(strf::make_ftuple(f1))
            , strf::make_ftuple(f2, f3)
            , strf::make_ftuple(f2, f3)
            , f4
            , strf::make_ftuple(strf::make_ftuple())  
            , f5
            , strf::make_ftuple()  
            );
        BOOST_TEST(digest(fmt) == expected);
    }

    
    return  boost::report_errors();
}


