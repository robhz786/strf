//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <QString>
#include <boost/stringify.hpp>
#include <climits>

namespace strf = boost::stringify::v0;

class QStringCreator: public strf::basic_outbuf<char16_t>
{
public:

    QStringCreator() : strf::basic_outbuf<char16_t>(_buffer, _buffer_size)
    {
    }

    void reserve(std::size_t size)
    {
        Q_ASSERT(size < static_cast<std::size_t>(INT_MAX));
        _str.reserve(static_cast<int>(size));
    }

    void recycle() override;

    QString finish();

private:

    QString _str;
    std::exception_ptr _eptr = nullptr;
    constexpr static std::size_t _buffer_size = strf::min_size_after_recycle<char16_t>();
    char16_t _buffer[_buffer_size];
};

void QStringCreator::recycle()
{
    if (this->good())
    {
        const QChar * qchar_buffer = reinterpret_cast<QChar*>(_buffer);
        std::size_t count = this->pos() - _buffer;
        try
        {
            _str.append(qchar_buffer, count);
        }
        catch(...)
        {
            _eptr = std::current_exception();
            this->set_good(false);
        }
    }
    this->set_pos(_buffer);
}

QString QStringCreator::finish()
{
    recycle();
    this->set_good(false);
    if (_eptr != nullptr)
    {
        std::rethrow_exception(_eptr);
    }
    return std::move(_str);
}

constexpr strf::dispatcher<strf::facets_pack<>, QStringCreator> toQString{};

int main()
{
    int x = 255;
    QString str = toQString(x, u" in hexadecimal is ", ~strf::hex(x));
    BOOST_ASSERT(str == "255 in hexadecimal is 0xff");

    return 0;
}
