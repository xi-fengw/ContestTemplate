struct FHQ_Treap {
    const int n;
    struct Node {
        int l, r, val, rnd, siz;
    };

    std::vector<Node> tr;
    int idx = 0, root = 0, x = 0, y = 0, z = 0;

    Treap(int n) : n(n), tr(n) {}

    void pull(int u) {  //传递
        tr[u].siz = tr[tr[u].l].siz + tr[tr[u].r].siz + 1;
    }

    void split(int u, int a, int& x, int& y) {  //分裂
        if (!u) return void(x = y = 0);
        if (tr[u].val <= a) x = u, split(tr[u].r, a, tr[u].r, y);
        if (tr[u].val > a) y = u, split(tr[u].l, a, x, tr[u].l);

        pull(u);
    }

    int merge(int u, int v) {   //合并
        if (!u || !v) return u | v;
        if (tr[u].rnd <= tr[v].rnd) {
            tr[u].r = merge(tr[u].r, v);
            pull(u);
            return u;
        }
        tr[v].l = merge(u, tr[v].l);
        pull(v);
        return v;
    }

    int build(int v) {  //创建新节点
        tr[++ idx].val = v, tr[idx].siz = 1, tr[idx].rnd = rand();
        return idx;
    }

    void insert(int a) {    //插入操作
        split(root, a, x, y);
        root = merge(merge(x, build(a)), y);
    }

    void del(int a) {   //删除操作
        split(root, a, x, z);
        split(x, a - 1, x, y);
        y = merge(tr[y].l, tr[y].r);
        root = merge(merge(x, y), z);
    }

    int kth(int now, int k) {   //第k大的数
        while(true) {
            if (k <= tr[tr[now].l].siz) now = tr[now].l;
            else if (k == tr[tr[now].l].siz + 1) return now;
            else k = k - tr[tr[now].l].siz - 1, now = tr[now].r;
        }
        // if (k <= tr[tr[now].l].siz) return kth(tr[now].l, k);
        // else if (k == tr[tr[now].l].siz + 1) return now;
        // else {
        //  k = k - tr[tr[now].l].siz - 1;
        //  return kth(tr[now].r, k);
        // }
    }

    int rankbykey(int rank) {    //获得排名为rank的数
        split(root, rank - 1, x, y);
        int k = tr[x].siz + 1;
        root = merge(x, y);
        return k;
    }

    int findpre(int a) {    //前缀
        split(root, a - 1, x, y);
        int res = tr[kth(x, tr[x].siz)].val;
        root = merge(x, y);
        return res;
    }

    int findsuf(int a) {    //后缀
        split(root, a, x, y);
        int res = tr[kth(y, 1)].val;
        root = merge(x, y);
        return res;
    }
};
