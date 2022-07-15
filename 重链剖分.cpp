    // siz是子树大小，son是重儿子，dep是这个节点的深度，dfn是dfs序，top是链的端点
    std::vector<int> parent(n + 1), siz(n + 1), son(n + 1), dep(n + 1);
    std::vector<int> dfn(n + 1), top(n + 1);
    int idx = 0;

    std::function<void(int, int, int)> dfs1 = [&] (int u, int fa, int depth) -> void {  //预处理出来轻重链
        parent[u] = fa;
        dep[u] = depth;
        siz[u] = 1;
        for (auto v : G[u]) {
            if (v == fa) continue;
            dfs1(v, u, depth + 1);
            siz[u] += siz[v];
            if (siz[v] > siz[son[u]])
                son[u] = v;
        }
    };

    std::function<void(int, int)> dfs2 = [&] (int u, int t) -> void {   //剖分
        dfn[u] = ++ idx;
        top[u] = t;
        w[idx] = a[u];

        if (!son[u]) return ;
        dfs2(son[u], t);

        for (auto v : G[u]) {
             if (v == parent[u] || v == son[u]) continue;
             dfs2(v, v);
        }
    };

    auto ask = [&] (int u, int v) -> int {  //查询树上两个节点之间的最短距离中所有节点
        int ans = 0;
        while(top[u] != top[v]){
            if (dep[top[u]] < dep[top[v]]) std::swap(u, v);
            ans += SGT.query(1, dfn[top[u]], dfn[u]);
            ans %= p;
            u = parent[top[u]];
        }
        if (dep[u] > dep[v]) std::swap(u, v);
        ans += SGT.query(1, dfn[u], dfn[v]);
        ans %= p;
        return ans;
    };

    auto update = [&] (int u, int v, int x) -> void {   //修改两个节点最短路径上
        x %= p;
        while(top[u] != top[v]) {
            if (dep[top[u]] < dep[top[v]]) std::swap(u, v);
            SGT.modify(1, dfn[top[u]], dfn[u], x);
            u = parent[top[u]];
        }
        if (dep[u] > dep[v]) std::swap(u, v);
        SGT.modify(1, dfn[u], dfn[v], x);
    };
