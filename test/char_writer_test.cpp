#include <boost/detail/lightweight_test.hpp>
#include <boost/rose.hpp>


void test_char32_to_utf8()
{
  using namespace boost::rose;

  {
    std::string output;
    output << listf
    {
      U'\u007F', U'\u0080', U'\u07ff', U'\u0800',
      U'\uFFFF', U'\U00010000', U'\U0010ffff'
    }; 

    BOOST_TEST(
      output == u8"\u007F\u0080\u07ff\u0800\uFFFF\U00010000\U0010ffff");
  }

  {
    // test invalid codepoints
    std::string output;
    output << listf{static_cast<char32_t>(0x110000)};
    BOOST_TEST(output.empty());
  }
}


void test_char32_to_utf16()
{
  using namespace boost::rose;

  {
    std::u16string output;
    output << listf16
    {
      U'\ud7ff', U'\ue000', U'\uffff', U'\U00010000', U'\U0010ffff'
    };

    BOOST_TEST(output == u"\ud7ff\ue000\uffff\U00010000\U0010ffff");
  }
  {
    // invalid codepoints
    std::u16string output;
    output << listf16{static_cast<char32_t>(0xd800), static_cast<char32_t>(0xdfff)};
    BOOST_TEST(output.empty());
  }
}


void test_char32_to_wstring()
{
  using namespace boost::rose;
  std::wstring output;

  output << wlistf
  {
    U'\ud7ff', U'\ue000', U'\uffff', U'\U00010000', U'\U0010ffff'
  };

  BOOST_TEST(output == L"\ud7ff\ue000\uffff\U00010000\U0010ffff");
}



int main()
{
  test_char32_to_utf16();
  test_char32_to_utf8();
  test_char32_to_wstring();

  return  boost::report_errors();
}













