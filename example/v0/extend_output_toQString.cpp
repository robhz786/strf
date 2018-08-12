#include <QString>
#include <boost/stringify.hpp>
#include <climits>

namespace strf = boost::stringify::v0;

class QStringCreator: public strf::buffered_writer<char16_t>
{
public:

    QStringCreator(strf::output_writer_init<char16_t> init)
        : strf::buffered_writer<char16_t>(init, buffer, buffer_size)
    {
    }

    ~QStringCreator()
    {
        this->flush();
    }

    void reserve(std::size_t size)
    {
        Q_ASSERT(size < static_cast<std::size_t>(INT_MAX));
        m_str.reserve(static_cast<int>(size));
    }

    strf::expected<QString, std::error_code> finish()
    {
        auto x = strf::buffered_writer<char16_t>::finish();
        if(x)
        {
            return {boost::stringify::v0::in_place_t{}, std::move(m_str)};
        }
        return {strf::unexpect_t{}, x.error()};
    }

protected:

    bool do_put(const char16_t* str, std::size_t count) override;

private:

    QString m_str;

    constexpr static std::size_t buffer_size = 60;
    char16_t buffer[buffer_size];
    QChar    qchar_buffer[buffer_size];
};

bool QStringCreator::do_put(const char16_t* str, std::size_t count)
{
    Q_ASSERT(count <= buffer_size);
    std::copy(str, str + count, qchar_buffer);
    m_str.append(qchar_buffer, static_cast<int>(count));
    return true;
}

constexpr auto toQString = strf::make_destination<QStringCreator>();


int main()
{
    int x = 255;
    QString str = toQString(x, u" in hexadecimal is ", ~strf::hex(x)).value();
    BOOST_ASSERT(str == "255 in hexadecimal is 0xff");

    return 0;
}
