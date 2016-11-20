#ifndef BOOST_STRINGIFY_INPUT_BASE_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_BASE_HPP_INCLUDED

#include <boost/assert.hpp>
#include <boost/stringify/custom_fill.hpp>
#include <boost/stringify/custom_width.hpp>
#include <boost/stringify/custom_width_calculator.hpp>
#include <cstddef>

namespace boost
{
namespace stringify 
{

template <typename CharT, typename Output, typename Formating>
class input_base
{
public:
  virtual ~input_base()
    {}

    /**
       return the amount of character that needs to be allocated,
       not counting the termination character. The implementer is
       not required to calculate the exact value if this is 
       difficult. But it must be greater or equal. And should not
       be much greater.
    */
    virtual std::size_t length(const Formating&) const noexcept = 0;

    static constexpr bool random_access_output = true; // todo
    static constexpr bool noexcept_output = true; // todo
    
    virtual void write(Output&, const Formating& fmt)
        const noexcept(noexcept_output) {} // todo make pure virtual
};

} //namespace stringify
} //namespace boost


#endif






