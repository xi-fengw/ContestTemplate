// jiangly
template<class Info, class Merge = std::plus<Info>>
struct SegmentTree {
    const int n;
    const Merge merge;
    std::vector<Info> info;
    SegmentTree(int n) : n(n), merge(Merge()), info(4 << std::__lg(n)) {}
    SegmentTree(std::vector<Info> init) : SegmentTree(init.size()) {
        std::function<void(int, int, int)> build = [&] (int u, int l, int r) {
            if (l == r) {
                info[u] = init[l];
                return ;
            }
            int mid = (l + r) >> 1;
            build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
            pull(u);
        };
        build(1, 1, n);
    }
 
    void pull(int u) {
        info[u] = merge(info[u << 1], info[u << 1 | 1]);
    }
 
    void modify(int u, int l, int r, int pos, const Info& x) {
        if (l == r) return void(info[u] = x);
        int mid = (l + r) >> 1;
        if (pos <= mid) modify(u << 1, l, mid, pos, x);
        else modify(u << 1 | 1, mid + 1, r, pos, x);
        pull(u);
    }
 
    Info query(int u, int l, int r, int ln, int rn) {
        if (l >= ln && r <= rn) return info[u];
        if (l >= rn || r <= ln) return Info();
        int mid = (l + r) >> 1;
        return merge(query(u << 1, l, mid, ln, rn), query(u << 1 | 1, mid + 1, r, ln, rn));
 
    }
};
 
struct Max {
    int x;
    Max(int x = 0) : x(x) {}
};
 
Max operator+(const Max& a, const Max& b) {
    return std::max(a.x, b.x);
}

//***************************

struct Seg {
    i64 val[N << 2], tag[N << 2];

    void pull(int u) {val[u] = val[u << 1] + val[u << 1 | 1];}

    void build(int u, int l, int r) {
        if (l == r) return void(val[u] = 1);
        int mid = (l + r) >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pull(u);
    }

    void push(int u, int l, int r) {
        if (tag[u]) {
            int mid = (l + r) >> 1;
            tag[u << 1] = tag[u];
            tag[u << 1 | 1] = tag[u];
            val[u << 1] = tag[u] * (mid - l + 1);
            val[u << 1 | 1] = tag[u] * (r - mid);
            tag[u] = 0;
        }
    }

    void change(int u, int l, int r, int ln, int rn, int v) {
        if (l >= ln && r <= rn) {
            val[u] = v * (r - l + 1);
            tag[u] = v;
            return ;
        }

        int mid = (l + r) >> 1;
        push(u, l, r);
        if (mid >= ln) change(u << 1, l, mid, ln, rn, v);
        if (mid < rn) change(u << 1 | 1, mid + 1, r, ln, rn, v);
        pull(u);
    }

    i64 query(int u, int l, int r, int ln, int rn) {
        if (l >= ln && r <= rn) return val[u];
        int mid = (l + r) >> 1;
        push(u, l, r);

        i64 ans = 0;
        if (mid >= ln) ans += query(u << 1, l, mid, ln, rn);
        if (mid < rn) ans += query(u << 1 | 1, mid + 1, r, ln, rn);
        pull(u);
        return ans;
    }

}SGT;
