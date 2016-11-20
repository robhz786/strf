#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>


template <typename charT>
struct to_upper_char_traits : public std::char_traits<charT>
{
  static charT*
  copy(charT* to, const charT* from, std::size_t n)
  {
    charT* it = to;
    while(n--)
      *it++ = std::toupper(*from++);
    return to;
  }

  static void
  assign(charT& c1, const charT& c2)
  {
    c1 = std::toupper(c2);
  }

  static charT*
  assign(charT* dest, std::size_t n, charT a)
  {
    std::fill_n(dest, n, std::toupper(a));
    return dest;
  }
};


template <typename charT>
struct weird_char_traits : public std::char_traits<charT>
{
  static charT*
  copy(charT* to, const charT* from, std::size_t n)
  {
    charT* it = to;
    while(n--)
    {
        assign(*it++, *from++);
    }
    return to;
  }

  static void
  assign(charT& c1, const charT& c2)
  {
      if (c2 == charT())
      {
          c1 = c2;
      }
      else
      {
          c1 = c2 | ( 1 << (sizeof(charT) * 8 - 1));
      }
  }

  static charT*
  assign(charT* dest, std::size_t n, charT a)
  {
      charT b;
      assign(b, a);
      std::fill_n(dest, n, b);
      return dest;
  }
};


template <int LINE_NUM, typename charT>
class tester
{
    typedef
    boost::stringify::detail::char_ptr_writer<charT, std::char_traits<charT> >
    writer_type;
    
public:
    typedef charT char_type;
    
    tester(std::basic_string<charT> expected)
        : m_expected_size(0)
        , m_expected(std::move(expected))
    {
    }

    tester(tester&&) = default;
    
    ~tester()
    {
    }
          
    void reserve(std::size_t size)
    {
        m_writer.reserve(size);
        m_expected_size = size;
    }
    
    bool finish()
    {
        auto result = m_writer.finish();
        BOOST_TEST(m_expected == result);
        BOOST_TEST(m_expected_size == 1 + m_expected.length());
        return m_expected == result
            && m_expected_size == 1 + m_expected.length();
    }

    void put(charT character) noexcept
    {
        m_writer.put(character);
    }

    void put(charT character, std::size_t repetitions) noexcept
    {
        m_writer.put(character, repetitions);
    }

    void put(const charT* str, std::size_t count) noexcept
    {
        m_writer.put(str, count);
    }
    
    bool set_pos(std::size_t pos) noexcept
    {
        return m_writer.set_pos(pos);
    }

    decltype(auto) get_pos() noexcept
    {
        return m_writer.get_pos();
    }
    
    void rput(charT character) noexcept
    {
        m_writer.rput(character);
    }

private:

    std::size_t m_expected_size;
    std::basic_string<charT> m_expected;
    boost::stringify::detail::string_maker<std::basic_string<charT>> m_writer;
};
      
template<int LINE_NUM, typename charT>
decltype(auto)
    testf(const charT* expected)    
{
    return boost::stringify::writef_helper<tester<LINE_NUM, charT>>(expected);
}



#endif


