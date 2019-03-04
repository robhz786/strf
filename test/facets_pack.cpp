//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

template <int N> struct fcategory;

template <int N> struct facet
{
    using category = fcategory<N>;
    int value = 0;
};

template <int N> struct fcategory
{
    constexpr static bool constrainable = true;

    static facet<N> get_default() noexcept
    {
        return facet<N>{};
    }
};

class class_x{};

class class_xa: public class_x {};
class class_xb: public class_x {public: int i;};
class class_c {};

template <typename T>
using derives_from_x = std::is_base_of<class_x, T>;

namespace strf = boost::stringify;

int main()
{
    auto f1_10 = facet<1>{10};
    auto f2_20 = facet<2>{20};
    auto f2_21 = facet<2>{21};
    auto f2_22 = facet<2>{22};
    auto f3_30 = facet<3>{30};
    struct dummy_type{};


    {   // strip_facet
        // auto cf2_22 = strf::constrain<std::is_integral>(f2_22);
        // auto ccf2_22 = strf::constrain<std::is_signed>(cf2_22);
        // auto rccf2_22 = std::cref(ccf2_22);
        
        // const auto& stripped = strf::strip_facet(rccf2_22);
        // int x = stripped;
        // static_assert( std::is_same<decltype(stripped), const facet<2>&>::value
        //              , "strip_facet failed");
    }

    {
        auto fp = strf::pack(
            f1_10,
            f2_20,
            f2_21,
            strf::constrain<std::is_integral>(f2_22),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<fcategory<1>, int>(fp);
        decltype(auto) f2d = strf::get_facet<fcategory<2>, double>(fp);
        decltype(auto) f2i = strf::get_facet<fcategory<2>, int>(fp);
        decltype(auto) f3i = strf::get_facet<fcategory<3>, int>(fp);

        static_assert(std::is_same<decltype(f1i), const facet<1>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2d), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2i), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f3i), const facet<3>&>::value, "wrong type");

        BOOST_TEST(f1i.value == 10);
        BOOST_TEST(f2d.value == 21);
        BOOST_TEST(f2i.value == 22);
        BOOST_TEST(f3i.value == 30);
    }

    {   // using std::reference_wrapper<Facet>
        // and constrained_faced<Filter, std::reference_wrapper<F>>

        auto fp = strf::pack(
            std::ref(f1_10),
            std::cref(f2_20),
            strf::constrain<std::is_arithmetic>(std::ref(f2_21)),
            strf::constrain<std::is_integral>(std::cref(f2_22)),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<fcategory<1>, int>(fp);
        decltype(auto) f2t = strf::get_facet<fcategory<2>, dummy_type>(fp);
        decltype(auto) f2d = strf::get_facet<fcategory<2>, double>(fp);
        decltype(auto) f2i = strf::get_facet<fcategory<2>, int>(fp);
        decltype(auto) f3i = strf::get_facet<fcategory<3>, int>(fp);

        static_assert(std::is_same<decltype(f1i), const facet<1>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2t), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2d), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2i), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f3i), const facet<3>&>::value, "wrong type");

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

        auto fp = strf::pack(
            std::cref(f1_10),
            std::ref(f2_20),
            std::ref(constrained_f2_21),
            std::cref(constrained_f2_22),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<fcategory<1>, int>(fp);
        decltype(auto) f2t = strf::get_facet<fcategory<2>, dummy_type>(fp);
        decltype(auto) f2d = strf::get_facet<fcategory<2>, double>(fp);
        decltype(auto) f2i = strf::get_facet<fcategory<2>, int>(fp);
        decltype(auto) f3i = strf::get_facet<fcategory<3>, int>(fp);

        static_assert(std::is_same<decltype(f1i), const facet<1>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2t), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2d), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2i), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f3i), const facet<3>&>::value, "wrong type");

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

        auto fp = strf::pack(
            std::cref(f1_10),
            std::ref(f2_20),
            std::ref(constrained_ref_f2_21),
            std::cref(constrained_ref_f2_22),
            f3_30
        );

        decltype(auto) f1i = strf::get_facet<fcategory<1>, int>(fp);
        decltype(auto) f2t = strf::get_facet<fcategory<2>, dummy_type>(fp);
        decltype(auto) f2d = strf::get_facet<fcategory<2>, double>(fp);
        decltype(auto) f2i = strf::get_facet<fcategory<2>, int>(fp);
        decltype(auto) f3i = strf::get_facet<fcategory<3>, int>(fp);

        static_assert(std::is_same<decltype(f1i), const facet<1>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2t), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2d), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f2i), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(f3i), const facet<3>&>::value, "wrong type");

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

        auto f2_20_empty = strf::constrain<std::is_empty>(f2_20);
        auto f2_20_empty_and_derives_from_x
            = strf::constrain<derives_from_x>(f2_20_empty);

        auto fp = strf::pack
            (
                strf::constrain<std::is_empty>(f2_21),
                strf::constrain<derives_from_x>(f2_22),
                f2_20_empty_and_derives_from_x
            );

        decltype(auto) xf2_20 = strf::get_facet<fcategory<2>, class_xa>(fp);
        decltype(auto) xf2_22 = strf::get_facet<fcategory<2>, class_xb>(fp);
        decltype(auto) xf2_21 = strf::get_facet<fcategory<2>, class_c>(fp);
        static_assert(std::is_same<decltype(xf2_20), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(xf2_21), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(xf2_22), const facet<2>&>::value, "wrong type");
        BOOST_TEST(xf2_20.value == 20);
        BOOST_TEST(xf2_21.value == 21);
        BOOST_TEST(xf2_22.value == 22);
    }

    {   // constrain<Filter1>(std::ref(a_contrained_facet))

        auto f2_20_empty = strf::constrain<std::is_empty>(std::ref(f2_20));
        auto f2_20_empty_and_derives_from_x
            = strf::constrain<derives_from_x>(std::ref(f2_20_empty));

        auto fp = strf::pack
            (
                strf::constrain<std::is_empty>(std::cref(f2_21)),
                strf::constrain<derives_from_x>(std::ref(f2_22)),
                std::cref(f2_20_empty_and_derives_from_x)
            );
        decltype(auto) xf2_20 = strf::get_facet<fcategory<2>, class_xa>(fp);
        decltype(auto) xf2_22 = strf::get_facet<fcategory<2>, class_xb>(fp);
        decltype(auto) xf2_21 = strf::get_facet<fcategory<2>, class_c>(fp);
        static_assert(std::is_same<decltype(xf2_20), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(xf2_21), const facet<2>&>::value, "wrong type");
        static_assert(std::is_same<decltype(xf2_22), const facet<2>&>::value, "wrong type");
        BOOST_TEST(&xf2_20 == &f2_20);
        BOOST_TEST(&xf2_21 == &f2_21);
        BOOST_TEST(&xf2_22 == &f2_22);

    }


    return boost::report_errors();
}
