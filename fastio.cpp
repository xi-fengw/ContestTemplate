#define FREAD_OPTION
 
#ifdef FREAD_OPTION
#define MAXBUFFERSIZE 1000000
inline char fgetc() {
    static char buf[MAXBUFFERSIZE + 5], *p1 = buf, *p2 = buf;
    return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, MAXBUFFERSIZE, stdin), p1 == p2) ? EOF : *p1++;
}
#undef MAXBUFFERSIZE
#define getchar fgetc
#endif
#define gc getchar
struct IOReader {
    template <typename T>
    inline IOReader &operator>>(T &a) {
        a = 0;
        bool flg = false;
        char ch = gc();
        while (ch < '0' || ch > '9') {
            if (ch == '-')
                flg ^= 1;
            ch = gc();
        }
        while (ch >= '0' && ch <= '9') {
            a = (a << 3) + (a << 1) + (ch ^ '0');
            ch = gc();
        }
        if (flg)
            a = -a;
        return *this;
    }
    inline IOReader &operator>>(std::string &a) {
        a.clear();
        char ch = gc();
        while (isspace(ch) && ch != EOF)
            ch = gc();
        while (!isspace(ch) && ch != EOF)
            a += ch, ch = gc();
        return *this;
    }
    inline IOReader &operator>>(char *a) {
#ifdef FREAD_OPTION
        char ch = gc();
        while (isspace(ch) && ch != EOF)
            ch = gc();
        while (!isspace(ch) && ch != EOF)
            *(a++) = ch, ch = gc();
        *a = '\0';
#else
        scanf(" %s", a);
#endif
        return *this;
    }
    inline IOReader &operator>>(char &a) {
        a = gc();
        while (isspace(a))
            a = gc();
        return *this;
    }
#define importRealReader(type)                      \
    inline IOReader &operator>>(type &a) {          \
        a = 0;                                      \
        bool flg = false;                           \
        char ch = gc();                             \
        while ((ch < '0' || ch > '9') && ch != '.') {\
            if (ch == '-')                          \
                flg ^= 1;                           \
            ch = gc();                              \
        }                                           \
        while (ch >= '0' && ch <= '9'){             \
            a = a * 10 + (ch ^ '0');                \
            ch = gc();                              \
        }                                           \
        if (ch == '.') {                            \
            ch = gc();                              \
            type p = 0.1;                           \
            while (ch >= '0' && ch <= '9')          \
            {                                       \
                a += p * (ch ^ '0');                \
                ch = gc();                          \
                p *= 0.1;                           \
            }                                       \
        }                                           \
        if (flg)                                    \
            a = -a;                                 \
        return *this;                               \
    }
    importRealReader(float) importRealReader(double) importRealReader(long double)
#undef importRealReader
} iocin;
#define cin iocin
#define importReadInteger(type, name)             \
    type name()                                   \
    {                                             \
        type a = 0;                               \
        bool flg = false;                         \
        char ch = gc();                           \
        while (ch < '0' || ch > '9')              \
        {                                         \
            if (ch == '-')                        \
                flg ^= 1;                         \
            ch = gc();                            \
        }                                         \
        while (ch >= '0' && ch <= '9')            \
        {                                         \
            a = (a << 3) + (a << 1) + (ch ^ '0'); \
            ch = gc();                            \
        }                                         \
        if (flg)                                  \
            a = -a;                               \
        return a;                                 \
    }
importReadInteger(int, readInt) importReadInteger(unsigned int, readUInt) importReadInteger(long long, readLL) importReadInteger(unsigned long long, readULL) importReadInteger(short, readShort) importReadInteger(unsigned short, readUShort)
#undef importReadInteger
#define importReadDecimal(type, name)               \
    type name()                                     \
    {                                               \
        type a = 0;                                 \
        bool flg = false;                           \
        char ch = gc();                             \
        while ((ch < '0' || ch > '9') && ch != '.') \
        {                                           \
            if (ch == '-')                          \
                flg ^= 1;                           \
            ch = gc();                              \
        }                                           \
        while (ch >= '0' && ch <= '9')              \
        {                                           \
            a = a * 10 + (ch ^ '0');                \
            ch = gc();                              \
        }                                           \
        if (ch == '.')                              \
        {                                           \
            ch = gc();                              \
            type p = 0.1;                           \
            while (ch >= '0' && ch <= '9')          \
            {                                       \
                a += p * (ch ^ '0');                \
                ch = gc();                          \
                p *= 0.1;                           \
            }                                       \
        }                                           \
        if (flg)                                    \
            a = -a;                                 \
        return a;                                   \
    }
    importReadDecimal(float, readFL) importReadDecimal(double, readDB) importReadDecimal(long double, readLDB)
#undef importReadDecimal
#undef gc