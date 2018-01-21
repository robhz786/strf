//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>

template <int N> struct facet_category;

template <int N> struct facet_type
{
    using category = facet_category<N>;
    int value = 0;
};

template <int N> struct facet_category
{
    static const auto& get_default() noexcept
    {
        static const facet_type<N> f {};
        return f;
    }
};

class class_x{};

class class_xa: public class_x {};
class class_xb: public class_x {public: int i;};
class class_c {};

template <typename T>
using derives_from_x
= std::integral_constant<bool, ! std::is_base_of<class_x, T>::value>;

namespace strf = boost::stringify;

int main()
{
    auto f1_10 = facet_type<1>{10};
    auto f2_20 = facet_type<2>{20};
    auto f2_21 = facet_type<2>{21};
    auto f2_22 = facet_type<2>{22};
    auto f3_30 = facet_type<3>{30};
    struct dummy_type{};

    {
        auto ft = strf::make_ftuple(
            f1_10,
            f2_20,
            f2_21,
            strf::constrain<std::is_integral>(f2_22),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<facet_category<1>, int>(ft);
        decltype(auto) f2d = strf::get_facet<facet_category<2>, double>(ft);
        decltype(auto) f2i = strf::get_facet<facet_category<2>, int>(ft);
        decltype(auto) f3i = strf::get_facet<facet_category<3>, int>(ft);

        static_assert(std::is_same_v<decltype(f1i), const facet_type<1>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2d), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2i), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f3i), const facet_type<3>&>, "wrong type");

        BOOST_TEST(f1i.value == 10);
        BOOST_TEST(f2d.value == 21);
        BOOST_TEST(f2i.value == 22);
        BOOST_TEST(f3i.value == 30);
    }

    {   // using std::reference_wrapper<Facet>
        // and constrained_faced<Filter, std::reference_wrapper<F>>

        auto ft = strf::make_ftuple(
            std::ref(f1_10),
            std::cref(f2_20),
            strf::constrain<std::is_arithmetic>(std::ref(f2_21)),
            strf::constrain<std::is_integral>(std::cref(f2_22)),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<facet_category<1>, int>(ft);
        decltype(auto) f2t = strf::get_facet<facet_category<2>, dummy_type>(ft);
        decltype(auto) f2d = strf::get_facet<facet_category<2>, double>(ft);
        decltype(auto) f2i = strf::get_facet<facet_category<2>, int>(ft);
        decltype(auto) f3i = strf::get_facet<facet_category<3>, int>(ft);

        static_assert(std::is_same_v<decltype(f1i), const facet_type<1>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2t), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2d), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2i), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f3i), const facet_type<3>&>, "wrong type");

        BOOST_TEST(f1i.value == 10);
        BOOST_TEST(f2t.value == 20);
        BOOST_TEST(f2d.value == 21);
        BOOST_TEST(f2i.value == 22);
        BOOST_TEST(f3i.value == 30);

        BOOST_TEST(&f1i == &f1_10);
        BOOST_TEST(&f2t == &f2_20);
        BOOST_TEST(&f2d == &f2_21);
        BOOST_TEST(&f2i == &f2_22);
    }

    {   // std::reference_wrapper< constrained_faced<Filter, Facet> >

        auto constrained_f2_21 = strf::constrain<std::is_arithmetic>(f2_21);
        auto constrained_f2_22 = strf::constrain<std::is_integral>(f2_22);

        auto ft = strf::make_ftuple(
            std::cref(f1_10),
            std::ref(f2_20),
            std::ref(constrained_f2_21),
            std::cref(constrained_f2_22),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<facet_category<1>, int>(ft);
        decltype(auto) f2t = strf::get_facet<facet_category<2>, dummy_type>(ft);
        decltype(auto) f2d = strf::get_facet<facet_category<2>, double>(ft);
        decltype(auto) f2i = strf::get_facet<facet_category<2>, int>(ft);
        decltype(auto) f3i = strf::get_facet<facet_category<3>, int>(ft);

        static_assert(std::is_same_v<decltype(f1i), const facet_type<1>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2t), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2d), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2i), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f3i), const facet_type<3>&>, "wrong type");

        BOOST_TEST(f1i.value == 10);
        BOOST_TEST(f2t.value == 20);
        BOOST_TEST(f2d.value == 21);
        BOOST_TEST(f2i.value == 22);
        BOOST_TEST(f3i.value == 30);

        BOOST_TEST(&f1i == &f1_10);
        BOOST_TEST(&f2t == &f2_20);
    }

    {  //std::reference_wrapper
        //    < constrained_faced
        //        < Filter
        //        , std::reference_wrapper<Facet> > >

        auto constrained_ref_f2_21 = strf::constrain<std::is_arithmetic>(std::ref(f2_21));
        auto constrained_ref_f2_22 = strf::constrain<std::is_integral>(std::cref(f2_22));

        auto ft = strf::make_ftuple(
            std::cref(f1_10),
            std::ref(f2_20),
            std::ref(constrained_ref_f2_21),
            std::cref(constrained_ref_f2_22),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<facet_category<1>, int>(ft);
        decltype(auto) f2t = strf::get_facet<facet_category<2>, dummy_type>(ft);
        decltype(auto) f2d = strf::get_facet<facet_category<2>, double>(ft);
        decltype(auto) f2i = strf::get_facet<facet_category<2>, int>(ft);
        decltype(auto) f3i = strf::get_facet<facet_category<3>, int>(ft);

        static_assert(std::is_same_v<decltype(f1i), const facet_type<1>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2t), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2d), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f2i), const facet_type<2>&>, "wrong type");
        static_assert(std::is_same_v<decltype(f3i), const facet_type<3>&>, "wrong type");

        BOOST_TEST(f1i.value == 10);
        BOOST_TEST(f2t.value == 20);
        BOOST_TEST(f2d.value == 21);
        BOOST_TEST(f2i.value == 22);
        BOOST_TEST(f3i.value == 30);

        BOOST_TEST(&f1i == &f1_10);
        BOOST_TEST(&f2t == &f2_20);
        BOOST_TEST(&f2d == &f2_21);
        BOOST_TEST(&f2i == &f2_22);
    }

    {   // constrain<Filter1>(constrain<Filter2>(facet))

        auto f2_20_empty = strf::constrain<std::is_empty>(std::ref(f2_20));
        auto f2_20_empty_and_derives_from_x
            = strf::constrain<derives_from_x>(f2_20_empty);

        auto ft = strf::make_ftuple
            (
                std::cref(f2_20_empty_and_derives_from_x),
                strf::constrain<std::is_empty>(f2_21),
                strf::constrain<derives_from_x>(f2_22)
            );

        decltype(auto) xf2_20 = strf::get_facet<facet_category<2>, class_xa>(ft);
        static_assert(std::is_same_v<decltype(xf2_20), const facet_type<2>&>, "wrong type");
        BOOST_TEST(&xf2_20 == &f2_20);
    }

    {   // constrain<Filter1>(std::ref(a_contrained_facet))

        auto f2_20_empty = strf::constrain<std::is_empty>(std::ref(f2_20));
        auto f2_20_empty_and_derives_from_x
            = strf::constrain<derives_from_x>(std::ref(f2_20_empty));

        auto ft = strf::make_ftuple
            (
                std::cref(f2_20_empty_and_derives_from_x),
                strf::constrain<std::is_empty>(f2_21),
                strf::constrain<derives_from_x>(f2_22)
            );

        decltype(auto) xf2_20 = strf::get_facet<facet_category<2>, class_xa>(ft);
        static_assert(std::is_same_v<decltype(xf2_20), const facet_type<2>&>, "wrong type");
        BOOST_TEST(&xf2_20 == &f2_20);
    }


    return boost::report_errors();
}
