struct Treap {
    struct Node {
        int l, r;
        int key, val, cnt, size;
    };
    
    const int n;
    std::vector<Node> tr;
    int root = 0, idx = 0;
    
    Treap(int n) : n(n), tr(n) {
        auto build = [&] () -> void {   //初始化
            get_Node(-1E8), get_Node(1E8);
            tr[1].r = 2, root = 1;
            pull(root);
            
            if (tr[1].val < tr[2].val) zag(root);
        };
        build();
    }
    
    int get_Node(int key) {     //创建新节点
        tr[++ idx].key = key;
        tr[idx].val = rand();
        tr[idx].cnt = tr[idx].size = 1;
        return idx;
    }
    
    void pull(int p) {  // 传递
        tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
    }
    
    void zig(int& p){   //左旋
        int q = tr[p].l;
        tr[p].l = tr[q].r, tr[q].r = p, p = q;
        pull(tr[p].r), pull(p);
    }
    
    void zag(int& p) {  //右旋
        int q = tr[p].r;
        tr[p].r = tr[q].l, tr[q].l = p, p = q;
        pull(tr[p].l), pull(p);
    }
    
    void insert(int& p, int key) {  //插入元素
        if (!p) p = get_Node(key);
        else if (tr[p].key == key) tr[p].cnt ++;
        else if (tr[p].key > key) {
            insert(tr[p].l, key);
            if (tr[tr[p].l].val > tr[p].val) zig(p);
        } else {
            insert(tr[p].r, key);
            if (tr[tr[p].r].val > tr[p].val) zag(p);
        }
        pull(p);
    }
    
    void remove(int& p, int key) {      //删除操作
        if (!p) return ;
        if (tr[p].key == key) {
            if (tr[p].cnt > 1) tr[p].cnt --;
            else if (tr[p].l || tr[p].r) {
                if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val) 
                    zig(p), remove(tr[p].r, key);
                else 
                    zag(p), remove(tr[p].l, key);
            } else p = 0;
        } else if (tr[p].key > key) {
            remove(tr[p].l, key);
        } else remove(tr[p].r, key);
        
        pull(p);
    }
    
    int get_rank(int p, int key) {      //获得排名
        if (!p) return 0;
        if (tr[p].key == key) return tr[tr[p].l].size + 1;
        else if (tr[p].key > key) return get_rank(tr[p].l, key);
        return tr[tr[p].l].size + tr[p].cnt + get_rank(tr[p].r, key);
    }
    
    int get_key(int p, int rank) {      //获得键值
        if (!p) return 1E8;
        if (tr[tr[p].l].size >= rank) return get_key(tr[p].l, rank);
        else if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
        return get_key(tr[p].r, rank - tr[p].cnt - tr[tr[p].l].size);
    }
    
    int getpre(int p, int key) {    //前缀
        if (!p) return -1E8;
        if (tr[p].key >= key) return getpre(tr[p].l, key);
        return std::max(tr[p].key, getpre(tr[p].r, key));
    }
    
    int getnext(int p, int key) {   //后缀
        if (!p) return 1E8;
        if (tr[p].key > key) return std::min(getnext(tr[p].l, key), tr[p].key);
        return getnext(tr[p].r, key);
    }
};
