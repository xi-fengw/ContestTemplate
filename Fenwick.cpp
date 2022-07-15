template <typename T>
struct Fenwick {
    const int n;
    std::vector<T> tr;
    Fenwick(int n) : n(n), tr(n) {}
 
    void add(int x, T v) {
        for (int i = x; i <= n; i += i & -i) {
            tr[i] += v;
        }
    }
 
    T sum(int x) {
        T ans = 0;
        for (int i = x; i; i -= i & -i) {
            ans += tr[i];
        }
        return ans;
    }
    
    T rangeSum(int l, int r) {
        return sum(r) - sum(l);
    }
};
