#include <boost/stringify.hpp>
#include <boost/assert.hpp>


void sample1()
{
    //[trivial_facets_sample
    namespace strf = boost::stringify::v0;

    // some facets
    auto f1 = strf::fill_t<U'.'>;
    auto f2 = strf::hex;
    auto f3 = strf::showbase;

    std::string result = strf::make_string .with(f1, f2, f3) ({255, 8}, {255, {8, /*<<
        `'d'` = decimal base. >>*/"d"}});
    BOOST_ASSERT(result == "....0xff.....255");
    //]
}


 /*< This char32_t will be converted to utf-8
                                  unless another encoding is specified. Such
                                  convertion can also be customized with facets. >*/


void sample2()
{
    //[ftuple_sample
    namespace strf = boost::stringify::v0;

    auto f1 = strf::showbase;
    auto f2 = strf::fill_t<U'.'>;
    auto f3 = strf::hex;

   
    auto fmt1 = strf::make_ftuple(strf::left, f1);
    auto fmt2 = strf::make_ftuple(f2, f3);
    auto fmt3 = strf::make_ftuple( /*<< `strf::fill<U'~'>` overrides `f2`
       since it comes after. When two facets of the same category are present,
       the one that comes after ( at the right in the argument list ) wins.
       >>*/fmt1, fmt2, strf::fill_t<U'~'>);
    // fmt3 is equivalent to strf::make_ftuple(strf::left, f1, f2, f3, strf::fill<U'~'>)


    auto result = strf::make_string.with(fmt3) [{{255, 8}, {255, {8, "d"}}}];
    BOOST_ASSERT(result == "0xff~~~~255~~~~~");

    //]
    
    //you can also create you own version of make_string
    auto my_make_string = strf::make_string.with(fmt3);

    result = my_make_string({255, 8}, {255, {8, "d"}});
    BOOST_ASSERT(result == "0xff~~~~255~~~~~");

    result = my_make_string.with(fmt3)/*<<
      Here `f2` overrides `strf::fill<U'~'>` in `fmt3`. 
      You can call `.with()` again and again, like `make_string.with(f1, f2).with(f3).with(f4)`.
      The facets passed in a `.with()` call overrides the ones passed in the previous calls.
    >>*/.with(f2) ({255, 8}, {255, {8, "d"}});
    BOOST_ASSERT(result == "0xff....255.....");
    
}



void sample3()
{
    
//[facet_filters
    namespace strf = boost::stringify::v0;

    auto result = strf::make_string.with(strf::hex_if<std::is_unsigned>)(255, " ", 255u);
        
    BOOST_ASSERT(result == "255 ff");

//]

}


// template <typename T>
// struct is_long_long: public std::is_same<T, long long>
// {
// };
// 
// void facet_filters()
// {
//     namespace strf = boost::stringify::v0;
// 
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
//         ('a', "bc", 255, 255u, 255LL);
//         
//     BOOST_ASSERT(result == "abc       255        ff                 255");
// }




int main()
{
    sample1();
    sample2();
    sample3();

    return 0;
}


