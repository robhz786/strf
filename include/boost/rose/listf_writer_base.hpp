#ifndef BOOST_LISTF_WRITER_BASE_HPP_INCLUDED
#define BOOST_LISTF_WRITER_BASE_HPP_INCLUDED

#include <cstddef>

namespace boost
{
namespace rose 
{

  template <typename charT>
  class listf_writer_base
  {
  public:
    virtual ~listf_writer_base()=0;

    /**
       return position after last written character
     */
    virtual charT* write_without_termination_char(charT* output) const =0;

    charT* write(charT* output) const
    {
      charT* end = write_without_termination_char(output);
      *end = charT();
      return ++end;
    }

    /**
       return the amount of character that needs to be allocated,
       not counting the termination character. The implementer is
       not required to calculate the exact value if this is 
       difficult. But it must be greater or equal. And should not
       be much greater.
     */
    virtual std::size_t minimal_length() const = 0;

    std::size_t minimal_size() const
    {
      return 1 + minimal_length();
    }

  };

  template <typename charT>
  listf_writer_base<charT>::~listf_writer_base() 
  {
  }

  //--------------------------------------------------

  template <typename T, typename charT, typename traits>
  class listf_writer : public listf_writer_base<charT>
  {
    //This template class must be specialized for each type T.

    listf_writer(const T&){}
  };

} //namespace rose
} //namespace boost


#endif
