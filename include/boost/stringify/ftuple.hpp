#ifndef BOOST_STRINGIFY_FTUPLE_HPP
#define BOOST_STRINGIFY_FTUPLE_HPP

#include <type_traits>

namespace boost {
namespace stringify {


template <typename ... Formaters>
class ftuple;


template <>
class ftuple<>
{
private:
    
    template <typename FmtImpl, typename InputArg>
    using input_permitted = typename FmtImpl::template accept_input_type<InputArg>;

protected:

    struct matching_preference {};
    
    template <typename InputArg, typename FmtType>
    typename FmtType::default_impl
    do_get(const matching_preference&, FmtType) const
    {
        return typename FmtType::default_impl();
    }

public:
   
    template <typename FmtType, typename InputArg>
    decltype(auto) get() const
    {
        return typename FmtType::default_impl();
    }
};


template <typename FmtImpl, typename ... OtherFmtImpls>
class ftuple<FmtImpl, OtherFmtImpls ...>
    : public ftuple<OtherFmtImpls ...>
    , private FmtImpl
{
    
    typedef ftuple<OtherFmtImpls ...> parent;

protected:
    
    struct matching_preference: public parent::matching_preference
    {
    };

    using parent::do_get;

    template <typename InputArg>
    typename std::enable_if
         < FmtImpl::template accept_input_type<InputArg>::value
         , const FmtImpl&
         > :: type
    do_get(const matching_preference&, typename FmtImpl::fmt_type) const
    {
        return *this;
    }

public:
 
    ftuple(const FmtImpl& fmt_impl, const OtherFmtImpls& ... otherfmtimpls)
        : parent(otherfmtimpls ...)
        , FmtImpl(fmt_impl)
    {
    }

    template <typename FmtType, typename InputArg>
    decltype(auto) get() const
    {
        return this->do_get<InputArg>(matching_preference(), FmtType());
    }

};


template <typename FTuple, typename FmtType, typename InputArg>
using ftuple_get_return_type
= decltype(std::declval<FTuple>().template get<FmtType, InputArg>());


template <typename ... Fmts>
boost::stringify::ftuple<Fmts ...> make_formating(const Fmts& ... fmts)
{
    return boost::stringify::ftuple<Fmts ...>(fmts ...);
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_FTUPLE_HPP */

