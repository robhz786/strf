#include <boost/detail/lightweight_test.hpp>
#include <boost/listf.hpp>
#include <limits>
#include <locale>
#include <sstream>


template <typename charT, 
          typename traits,
          typename intT>

void test(intT value)
{
  std::basic_string<charT, traits> result;
  result += boost::basic_listf<charT, traits>{value};

  std::basic_ostringstream<charT, traits> oss;
  oss.imbue(std::locale("C"));
  oss << value;
  
  BOOST_TEST(result == oss.str());
}

template <typename charT, 
          typename traits=std::char_traits<charT> >
void test_many_values()
{
  test<charT, traits>(0);
  test<charT, traits>(123);
  test<charT, traits>(-123);
  test<charT, traits>(std::numeric_limits<int>::min());
  test<charT, traits>(std::numeric_limits<int>::max());
  test<charT, traits>(std::numeric_limits<long long>::min());
  test<charT, traits>(std::numeric_limits<long long>::max());
  test<charT, traits>(std::numeric_limits<unsigned int>::max());
  test<charT, traits>(std::numeric_limits<unsigned long long>::max());
}


int main()
{
  test_many_values<char>();
  test_many_values<wchar_t>();
 
  // gcc 4.8 does not support well std::basic_ostringstream<char16_t> 
  // and std::basic_ostringstream<char32_t> 

  //test_many_values<char16_t>();
  //test_many_values<char32_t>();

  {
    std::u16string output;
    output += boost::listf16{ 123 };
    BOOST_TEST(output == u"123");
  }
  {
    std::u16string output;
    output += boost::listf16{ -123 };
    BOOST_TEST(output == u"-123");
  }
  {
    std::u32string output;
    output += boost::listf32{ 123 };
    BOOST_TEST(output == U"123");
  }
  {
    std::u32string output;
    output += boost::listf32{ -123 };
    BOOST_TEST(output == U"-123");
  }



  return  boost::report_errors();
}
