//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[QStringAppender_def

#include <QString>
#include <boost/stringify.hpp>
#include <climits>

namespace strf = boost::stringify::v0;

class QStringAppender: public boost::basic_outbuf<char16_t>
{
public:

    QStringAppender(QString& str);

    void reserve(std::size_t size);

    void recycle() override;

    std::size_t finish();

private:

    QString& _str;
    std::size_t _count = 0;
    std::exception_ptr _eptr = nullptr;

    constexpr static std::size_t _buffer_size = boost::min_size_after_recycle<char16_t>();
    char16_t _buffer[_buffer_size];
};

//]

//[QStringAppender_ctor
QStringAppender::QStringAppender(QString& str)
    : boost::basic_outbuf<char16_t>(_buffer, _buffer_size)
    , _str(str)
{
}
//]

//[QStringAppender_recycle
void QStringAppender::recycle()
{
    if (this->good())
    {
        // Flush the content:
        std::size_t count = /*<<ouput_buffer::pos() returns the immediate position
                              after the last character the library wrote in the buffer>>*/this->pos() - _buffer;
        const QChar * qchar_buffer = reinterpret_cast<QChar*>(_buffer);
        try
        {
            _str.append(qchar_buffer, count);
            _count += count;
        }
        catch(...)
        {
            _eptr = std::current_exception();
            this->set_good(false);
        }
    }
    // Reset the buffer position:
    this->set_pos(_buffer);

    // Not necessary to set the buffer's end since it's the same as before:
    // this->set_end(_buffer + _buffer_size);
}
//]


//[QStringAppender_reserve
void QStringAppender::reserve(std::size_t size)
{
    Q_ASSERT(_str.size() + size < static_cast<std::size_t>(INT_MAX));
    _str.reserve(_str.size() + static_cast<int>(size));
}
//]

//[QStringAppender_finish
std::size_t QStringAppender::finish()
{
    recycle();
    if (_eptr != nullptr)
    {
        std::rethrow_exception(_eptr);
    }
    return _count;
}
//]

//[QStringAppender_dispatcher
inline auto append(QString& str)
{
    return strf::dispatcher<strf::facets_pack<>, QStringAppender, QString&> {str};
}
//]



//[QStringAppender_use
int main()
{
    QString str = "....";
    int initial_length = str.length();

    int x = 255;
    std::size_t append_count = append(str) (x, u" in hexadecimal is ", ~strf::hex(x));

    BOOST_ASSERT(str == "....255 in hexadecimal is 0xff");

    // append_count is equal to the value returned by QStringAppender::finish()
    BOOST_ASSERT(str.length() == (int)append_count + initial_length);

    return 0;
}
//]
