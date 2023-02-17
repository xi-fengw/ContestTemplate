// 0-indexed Fenwick (binary indexed tree / Fenwick tree) (i : [0, len))
template <class T> struct Fenwick {
    int n;
    std::vector<T> data;
    Fenwick(int len = 0) : n(len), data(len) {}
    void reset() { std::fill(data.begin(), data.end(), T(0)); }
    void add(int pos, T v) { // a[pos] += v
        pos++;
        while (pos > 0 && pos <= n) data[pos - 1] += v, pos += pos & -pos;
    }
    T sum(int k) const { // a[0] + ... + a[k - 1]
        T res = 0;
        while (k > 0) res += data[k - 1], k -= k & -k;
        return res;
    }
 
    T sum(int l, int r) const { return sum(r) - sum(l); } // a[l] + ... + a[r - 1]
 
    int kth(T k) {
        int ret = 0;
        for (int i = 1 << std::__lg(n); i; i /= 2) 
            if (ret + i <= n && k >= data[ret + i - 1]) 
                ret += i, k -= data[ret - 1];
        return ret;
    }

    template <class OStream> friend OStream &operator<<(OStream &os, const Fenwick &bit) {
        T prv = 0;
        os << '[';
        for (int i = 1; i <= bit.n; i++) {
            T now = bit.sum(i);
            os << now - prv << ',', prv = now;
        }
        return os << ']';
    }
};

// range modify
template<typename T>
class fenwick {
private:
    vector<T> f1, f2;
    int n;
 
    void modify(int x, T v) {
        for (int i = x; i < n; i |= i + 1)
            f1[i] += v, f2[i] += v * x;
    }
 
public:
    explicit fenwick(int _n) : n(_n), f1(_n), f2(_n) {}
 
    void modify(int l, int r, T v) {
        modify(l, v), modify(r + 1, -v);
    }
 
    T get(int x) {
        T v{};
        for (int i = x; i >= 0; i = (i & (i + 1)) - 1)
            v += f1[i] * (x + 1) - f2[i];
        return v;
    }
 
    T get(int l, int r) {
        if (l > r) return 0;
        return get(r) - get(l - 1);
    }
};