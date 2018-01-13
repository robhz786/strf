#include <boost/assert.hpp>
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;


void sample1()
{
    //[trivial_facets_sample

    // some facets
    auto f1 = strf::fill_t<U'.'>;
    auto f2 = strf::hex;
    auto f3 = strf::showbase;

    auto result = strf::make_string .with(f1, f2, f3) = {{255, 8}, {255, {8, /*<<
        `The 'd'` flag implies decimal base. Local formatting has priority over facets>>*/"d"}}};
    BOOST_ASSERT(result.value() == "....0xff.....255");
    //]
}


 /*< This char32_t will be converted to utf-8
                                  unless another encoding is specified. Such
                                  convertion can also be customized with facets. >*/


void sample2()
{
//[facet_filters
    auto uhex = strf::constrain<std::is_unsigned>(strf::hex);
    auto result = strf::make_string.with(uhex) = {255, " ", 255u};
        
    BOOST_ASSERT(result.value() == "255 ff");
//]
}

void sample3()
{
    
//[facet_overriding
    auto f1 = strf::oct;
    auto f2 = strf::constrain<std::is_unsigned>(strf::hex);

    // f2 overrides f1, but only for unsigned types:
    auto result = strf::make_string.with(f1, f2) = {255, " ", 255u};
    BOOST_ASSERT(result.value() == "377 ff");


    // Since f1 applies to all input types, f1 completely overrides f2 here.
    // So the presence of f2 has no effect:
    auto result_2 = strf::make_string.with(f2, f1) = {255, " ", 255u};
    BOOST_ASSERT(result_2.value() == "377 377");
//]

}


template <typename ... Args>
void sample(Args ... args)
{
    // some facets:
    auto f1 = strf::showbase;
    auto f2 = strf::showpos;
    auto f3 = strf::uppercase;
        
//[basic_ftuple_sample
    auto my_default_formatting = strf::make_ftuple(f1, f2, f3);

    auto result = strf::make_string.with(my_default_formatting) = {args ...};
    // just the same as strf::make_string.with(f1, f2, f3) (args ...)
//]
}



void sample4()
{
    auto my_default_formatting = strf::make_ftuple
        ( strf::showbase
        , strf::showpos
        , strf::uppercase
        );

    auto result = strf::make_string /*<<
    Just the same as .with(strf::showbase, strf::showpos, strf::uppercase)
    >>*/.with(my_default_formatting) = {15, ' ', {15, "x"}};
    BOOST_ASSERT(result.value() == "+15 0XF");
}

// void sample5()
// {
// //[custom_make_string
//     auto my_make_string = strf::make_string.with(strf::showbase, strf::showpos);
//     auto result = my_make_string = {15, ' ', {15, "x"}};
//     BOOST_ASSERT(result.value() == "+15 0xf");
// 
//     result = my_make_string /*<<
//     Yes, you can call `with()` again and again:
// 
//     `strf::make_string.with(f1).with(f2).with(f3) `
//     is equivalent to `strf::make_string.with(f1, f2, f3)`
//     >>*/ .with(strf::uppercase) = {15, ' ', {15, "x"}};
//     BOOST_ASSERT(result.value() == "+15 0XF");
//     
// //]
//     
//     //Note, all the following are equivalent:
// 
//     my_make_string
//         .with(strf::uppercase);
// 
//     strf::make_string
//         .with(my_default_formatting)
//         .with(strf::uppercase);
// 
//     strf::make_string
//         .with(my_default_formatting, strf::uppercase);
// 
//     strf::make_string
//         .with(strf::showbase, strf::showpos, strf::uppercase);
// 
//     strf::make_string
//         .with(strf::showbase)
//         .with(strf::showpos)
//         .with(strf::uppercase);
// }



void sample6()
{
    //[ftuple_of_ftuple
    auto facet_1 = strf::showbase;
    auto facet_2 = strf::fill_t<U'.'>;
    auto facet_3 = strf::hex;
   
    auto ftuple_1 = strf::make_ftuple(strf::left, facet_1);
    auto ftuple_2 = strf::make_ftuple(facet_2, facet_3);
    auto ftuple_3 = strf::make_ftuple(ftuple_1, ftuple_2, strf::fill_t<U'~'>);
    // ftuple_3 is equivalent to strf::make_ftuple(strf::left, facet_1, facet_2, facet_3, strf::fill<U'~'>)
    // which is equivalent to    strf::make_ftuple(strf::left, facet_1,          facet_3, strf::fill<U'~'>)
    // since facet_2 is overriden by strf::fill<U'~'>

    auto result = strf::make_string.with(ftuple_3) = {{255, 8}, {255, {8, "d"}}};
    BOOST_ASSERT(result.value() == "0xff~~~~255~~~~~");

    //]
    
    //you can also create you own version of make_string
    auto my_make_string = strf::make_string.with(ftuple_3);

    result = my_make_string = {{255, 8}, {255, {8, "d"}}};
    BOOST_ASSERT(result.value() == "0xff~~~~255~~~~~");

    result = my_make_string.with(ftuple_3)/*<<
      Here `facet_2` overrides `strf::fill<U'~'>` in `ftuple_3`. 
      You can call `.with()` again and again, like `make_string.with(facet_1, facet_2).with(facet_3).with(facet_4)`.
      The facets passed in a `.with()` call overrides the ones passed in the previous calls.
    >>*/.with(facet_2) = {{255, 8}, {255, {8, "d"}}};
    BOOST_ASSERT(result.value() == "0xff....255.....");
    
}





// template <typename T>
// struct is_long_long: public std::is_same<T, long long>
// {
// };
// 
// void facet_filters()
// {
//     std::string result = strf::make_string .with
//         ( strf::hex_if<std::is_unsigned>
//         , strf::with_if<strf::is_long_long>(20) /*<
//           first match wins, hence this facets must come before the next,
//           otherwise it would have no effect >*/
//         , strf::with_if<strf::ic_int>(10) /*<
//           `ic_int` matches the input types that are treated an integer.
//            If `std::is_integral` was used istead,
//            then it would also match `char`, which is not the intention here.
//           >*/
//         )
//         = {'a', "bc", 255, 255u, 255LL};
//         
//     BOOST_ASSERT(result == "abc       255        ff                 255");
// }



int main()
{
    sample1();
    sample2();
    sample3();
    sample4();
    //sample5();
    sample6();

    return 0;
}


