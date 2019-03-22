//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[QStringAppender_def

#include <QString>
#include <boost/stringify.hpp>
#include <climits>

namespace strf = boost::stringify::v0;

class QStringAppender: public strf::output_buffer<char16_t>
{
public:

    QStringAppender(QString& str);

    void reserve(std::size_t size);

    bool recycle() override;

    std::size_t finish();

private:

    QString& _str;
    std::size_t _count = 0;

    constexpr static std::size_t _buffer_size = strf::min_buff_size;
    char16_t _buffer[_buffer_size];
};

//]

//[QStringAppender_ctor
QStringAppender::QStringAppender(QString& str)
    : strf::output_buffer<char16_t>(_buffer, _buffer_size)
    , _str(str)
{
}
//]

//[QStringAppender_recycle
bool QStringAppender::recycle()
{
    // Flush the content:
    std::size_t count = /*<<ouput_buffer::pos() returns the immediate position
    after the last character the library wrote in the buffer>>*/this->pos() - _buffer;
    const QChar * qchar_buffer = reinterpret_cast<QChar*>(_buffer);
    _str.append(qchar_buffer, count);
    _count += count;

    // Reset the buffer position:
    this->set_pos(_buffer);

    // Not necessary to set the buffer's end since it's the same as before:
    // this->set_end(_buffer + _buffer_size);

    return true;
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
    if (this->has_error())
    {
        throw strf::stringify_error(this->get_error());
    }

    recycle();
    return _count;
}
//]

//[QStringAppender_dispatcher
inline auto append(QString& str)
{
    using dispatcher_type = strf::dispatcher< strf::facets_pack<>
                                            , QStringAppender
                                            , QString& >;
    return dispatcher_type(strf::pack(), str);
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
