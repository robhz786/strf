#ifndef SRC_ID
#define SRC_ID 0
#endif

#define STR(s) XSTR(s)
#define XSTR(X) # X

static const char* const   arg_a0 = STR(SRC_ID) "-aaaa";
constexpr unsigned         arg_a1 = 1000u + SRC_ID;
constexpr int              arg_a2 = SRC_ID;
constexpr int              arg_a3 = 5 + SRC_ID;
constexpr int              arg_a4 = 10l + SRC_ID;
constexpr int              arg_a5 = 9;

static const char* const   arg_b0 = "blah";
static const char* const   arg_b1 = STR(SRC_ID) "-bbb";
constexpr char             arg_b2 = 'b';
constexpr unsigned long    arg_b3 = 1000uL + SRC_ID;
constexpr int              arg_b4 = SRC_ID;
constexpr long long        arg_b5 = 5LL + SRC_ID;
constexpr int              arg_b6 = 10 + SRC_ID;
constexpr char             arg_b7 = '9';

static const char* const   arg_c0 = "BLAH";
static const char* const   arg_c1 = STR(SRC_ID) "-cccc";
constexpr int              arg_c2 = -1000 - SRC_ID;
constexpr int              arg_c3 = SRC_ID;
static const char* const   arg_c4 = "Blah";
constexpr long             arg_c5 = 5L + SRC_ID;
constexpr long             arg_c6 = 10L + SRC_ID;
constexpr unsigned long long  arg_c7 = 9;

    
    
    
