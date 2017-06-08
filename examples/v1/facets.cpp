#include <boost/stringify.hpp>
#include <boost/assert.hpp>


void sample()
{
    //[trivial_facets_sample
    namespace strf = boost::stringify::v1;

    // some facets
    auto f1 = strf::fill<U'.'>;
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
    namespace strf = boost::stringify::v1;

    auto f1 = strf::hex;
    auto f2 = strf::fill<U'.'>;
    auto f3 = strf::showbase;
   
    auto fmt1 = strf::make_ftuple(f1, f2);
    auto fmt2 = strf::make_ftuple(f3, strf::left);
    auto fmt3 = strf::make_ftuple( /*<< `strf::fill<U'~'>` overrides `f1`
       since it comes first. When two facets of the same category are present,
       the first one wins.
       >>*/strf::fill<U'~'>, fmt1, fmt2);
    // fmt3 is equivalent to strf::make_ftuple(strf::fill<U'~'>, f1, f2, f3, strf::left)


    auto result = strf::make_string.with(fmt3) ({255, 8}, {255, {8, "d"}});
    BOOST_ASSERT(result == "0xff~~~~255~~~~~");

    //]
    
    // you can also create you own version of make_string
    auto my_make_string = strf::make_string.with(fmt3);

    result = my_make_string({255, 8}, {255, {8, "d"}});
    BOOST_ASSERT(result == "0xff~~~~255~~~~~");

    result = my_make_string/*<<
      Here `f1` overrides `strf::fill<U'~'>` in `fmt3`. 
      You can call `.with()` again and again, like `make_string.with(f1, f2).with(f3).with(f4)`.
      The facets passed in a `.with()` call overrides the ones passed in the previous calls.
    >>*/.with(f1) ({255, 8}, {255, {8, "d"}});
    BOOST_ASSERT(result == "0xff....255.....");
}




void sample3()
{
    //[ftuple_sub_args
    namespace strf = boost::stringify::v1;

    auto outer_ftuple = strf::make_ftuple(strf::witdh(3), strf::left);
    auto inner_ftuple = strf::make_ftuple(strf::hex, strf::right);

    auto result = strf::make_string.with(outer_ftuple)
        (10, 11, {inner_ftuple, /*<<
        This sublist of arguments could contain just anything that can be inserted
        in the outer argument list ( including another ftuple with its own
        argument sublist ).
        >>*/{"~~~", 12, 13, {14, "d"}, 15, "~~~"}}, 16, 17);

    BOOST_ASSERT(result == "10 11 ~~~  c  d 14  f~~~16 17 "); /*< Note that `outer_ftuple`
        still has effect inside the sublist ( which is why width is 3 there ),
        but the facets of `inner_ftuple` are prefered
        ( which is why arguments are aligned to the right )>*/
    //]
}






void sample4()
{
    
//[facet_filters
    namespace strf = boost::stringify::v1;

    auto result = strf::make_string.with(strf::hex_if<std::is_unsigned>)(255, " ", 255u);
        
    BOOST_ASSERT(result == "255 ff");

//]

}









template <typename T>
struct is_long_long: public std::is_same<T, long long>
{
};

void facet_filters()
{
    namespace strf = boost::stringify::v1;

    std::string result = strf::make_string .with
        ( strf::hex_if<std::is_unsigned>
        , strf::with_if<strf::is_long_long>(20) /*<
          first match wins, hence this facets must come before the next,
          otherwise it would have no effect >*/
        , strf::with_if<strf::ic_int>(10) /*<
          `ic_int` matches the input types that are treated an integer.
           If `std::is_integral` was used istead,
           then it would also match `char`, which is not the intention here.
          >*/
        )
        ('a', "bc", 255, 255u, 255LL);
        
    BOOST_ASSERT(result == "abc       255        ff                 255");
}




int main()
{
    sample1();
    sample2();
    sample3();
    sample4();

    return 0;
}


