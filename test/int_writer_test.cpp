#include <boost/detail/lightweight_test.hpp>
#include <boost/rose.hpp>
#include <limits>
#include <locale>
#include <sstream>


template <typename charT, typename intT>
void test(intT value)
{
  std::basic_string<charT> result;
  result << boost::rose::basic_listf<charT>{value};

  std::basic_ostringstream<charT> oss;
  oss.imbue(std::locale("C"));
  oss << value;
  
  BOOST_TEST(result == oss.str());
}

template <typename charT>
void test_many_values()
{
  test<charT>(0);
  test<charT>(123);
  test<charT>(-123);
  test<charT>(std::numeric_limits<int>::min());
  test<charT>(std::numeric_limits<int>::max());
  test<charT>(std::numeric_limits<long long>::min());
  test<charT>(std::numeric_limits<long long>::max());
  test<charT>(std::numeric_limits<unsigned int>::max());
  test<charT>(std::numeric_limits<unsigned long long>::max());
}


int main()
{
  test_many_values<char>();
  test_many_values<wchar_t>();
 
  // gcc 4.8 does not support well std::basic_ostringstream
  //  for char16_t char32_t 
  //test_many_values<char16_t>();
  //test_many_values<char32_t>();

  {
    std::u16string output;
    output << boost::rose::listf16{ 123 };
    BOOST_TEST(output == u"123");
  }
  {
    std::u16string output;
    output << boost::rose::listf16{ -123 };
    BOOST_TEST(output == u"-123");
  }
  {
    std::u32string output;
    output << boost::rose::listf32{ 123 };
    BOOST_TEST(output == U"123");
  }
  {
    std::u32string output;
    output << boost::rose::listf32{ -123 };
    BOOST_TEST(output == U"-123");
  }



  return  boost::report_errors();
}
