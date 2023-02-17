template<class Info, class Tag>
struct SegTree {
    int n;
    vector<Info> tr;
    vector<Tag> tag;
    SegTree() = default;
    SegTree(int n) : n(n), tr(n + 5 << 2), tag(n + 5 << 2) {}
    SegTree(int n, const vector<Info> a) : SegTree(n) {
        function<void(int, int, int)> build = [&](int u, int l, int r) {
            if (l == r) {
                tr[u] = a[r];
                return;
            }
            int mid = l + r >> 1;
            build(u << 1, l, mid);
            build(u << 1 | 1, mid + 1, r);
            pushup(u);
        };
        build(1, 1, n);
    }
    void pushup(int u) {
        tr[u] = tr[u << 1] + tr[u << 1 | 1];
    }
    void apply(int u, const Tag &v) {
        tr[u].apply(v);
        tag[u].merge(v);
    }
    void pushdown(int u) {
        apply(u << 1, tag[u]);
        apply(u << 1 | 1, tag[u]);
        tag[u] = Tag();
    }
    void modify(int u, int l, int r, int x, const Info &v) {
        if (l > x || r < x) {
            return;
        }
        if (l == r) {
            tr[u] = v;
            return;
        }
        pushdown(u);
        int mid = l + r >> 1;
        modify(u << 1, l, mid, x, v);
        modify(u << 1 | 1, mid + 1, r, x, v);
        pushup(u);
    }
    void rangeModify(int u, int l, int r, int ql, int qr, const Tag &v) {
        if (l > qr || r < ql) {
            return;
        }
        if (ql <= l && r <= qr) {
            return apply(u, v);
        }
        pushdown(u);
        int mid = l + r >> 1;
        rangeModify(u << 1, l, mid, ql, qr, v);
        rangeModify(u << 1 | 1, mid + 1, r, ql, qr, v);
        pushup(u);
    }
    Info query(int u, int l, int r, int ql, int qr) {
        if (l > qr || r < ql) {
            return Info();
        }
        if (ql <= l && r <= qr) {
            return tr[u];
        }
        pushdown(u);
        int mid = l + r >> 1;
        return query(u << 1, l, mid, ql, qr) + query(u << 1 | 1, mid + 1, r, ql, qr);
    }
    template<class F>
    int find(int u, int l, int r, int ql, int qr, const F &f) {
        if (l > qr || r < ql || f(tr[u])) {
            return 0;
        }
        if (l == r) {
            return r;
        }
        pushdown(u);
        int mid = l + r >> 1;
        int res = find(u << 1, l, mid, ql, qr, f);
        return res ? res : find(u << 1 | 1, mid + 1, r, ql, qr, f);
    }
 
    Info query(int l, int r) {
        return query(1, 1, n, l, r);
    }
    void modify(int x, const Info &v) {
        return modify(1, 1, n, x, v);
    }
    void rangeModify(int l, int r, const Tag &o) {
        return rangeModify(1, 1, n, l, r, o);
    }
    template<class F>
    int find(int l, int r, const F &f) {
        return find(1, 1, n, l, r, f);
    }
};
 
struct Tag {
    int add = 0;
    void merge(const Tag &o) {
        add += o.add;
    }
};
 
struct Info {

    void apply(const Tag &o) {
    }
};
 
Info operator+(const Info &l, const Info &r) {
    return {
    };
}