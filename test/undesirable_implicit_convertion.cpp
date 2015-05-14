#include <boost/detail/lightweight_test.hpp>
#include <boost/rose/listf.hpp>


struct aaa
{
};

class bbb_base
{
};

template <class T>
class bbb: public bbb_base
{
  operator aaa() const
  {
    return aaa();
  } 
};

struct ccc
{
  ccc()
  {
  }

  ccc(const ccc&)
  {
  }

  ccc(const bbb_base&)
  {
  }

  template <class T>
  ccc(const bbb<T>&)
  {
  }
};


struct aaa_traits
{
  template <typename charT, typename traits>
  struct writer: public boost::rose::listf_writer_base<charT>
  {
    writer(const aaa&)
    {
    }

    virtual std::size_t minimal_length() const
    {
      return 3;
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      traits::copy(output, BOOST_STRING_LITERAL(charT, "aaa"), 3);
      return output + 3;
    }
  }; 
};


struct bbb_base_traits
{
  template <typename charT, typename traits>
  struct writer: public boost::rose::listf_writer_base<charT>
  {
    writer(const bbb_base&)
    {
    }

    virtual std::size_t minimal_length() const
    {
      return 8;
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      traits::copy(output, BOOST_STRING_LITERAL(charT, "bbb_base"), 8);
      return output + 8;
    }
  };
};

template <class T>
struct bbb_traits
{
  template <typename charT, typename traits>
  struct writer: public boost::rose::listf_writer_base<charT>
  {
    writer(bbb<T>)
    {
    }

    virtual std::size_t minimal_length() const
    {
      return 3;
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      traits::copy(output, BOOST_STRING_LITERAL(charT, "bbb"), 3);
      return output + 3;
    }
  }; 
};


struct ccc_traits
{
  template <typename charT, typename traits>
  struct writer: public boost::rose::listf_writer_base<charT>
  {
    writer(const ccc&)
    {
    }

    virtual std::size_t minimal_length() const
    {
      return 3;
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      traits::copy(output, BOOST_STRING_LITERAL(charT, "ccc"), 3);
      return output + 3;
    }
  }; 
};

inline aaa_traits listf_argument_traits(aaa)
{
  return aaa_traits();
}

inline bbb_base_traits listf_argument_traits(const bbb_base&)
{
  return bbb_base_traits();
}

template <class T>
inline bbb_traits<T> listf_argument_traits(bbb<T>)
{
  return bbb_traits<T>();
}

inline ccc_traits listf_argument_traits(ccc)
{
  return ccc_traits();
}

int main()
{
  bbb<double> bbb_instance;
  std::string output;
  output << boost::rose::listf{bbb_instance};
  BOOST_TEST(output == "bbb");

  return boost::report_errors();
}
