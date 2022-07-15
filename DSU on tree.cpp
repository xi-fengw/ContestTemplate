    std::vector<int> siz(n + 1), son(n + 1);

    std::function<void(int, int)> dfs1 = [&] (int u, int fa) -> void {
        siz[u] = 1;
        for (auto v : G[u]) {
            if (v == fa) continue;
            dfs1(v, u);
            
            siz[u] += siz[v];
            if (siz[v] > siz[son[u]]) {
                son[u] = v;
            }
        }
    };

    dfs1(1, 0);

    std::vector<bool> vis(n + 1);
    std::vector<int> cnt(n + 1), ans(n + 1);
    int top = 0;
    std::function<void(int, int, int)> calc = [&] (int u, int fa, int val) {
        if (val > 0) {
            if (!cnt[c[u]]) top ++;
            cnt[c[u]] ++;
        } else {
            if (cnt[c[u]] <= 1) top --;
            cnt[c[u]] --;
        }
        for (auto v : G[u]) {
            if (v == fa || vis[v]) continue;
            calc(v, u, val);
        }
    };

    std::function<void(int, int, bool)> dfs2 = [&] (int u, int fa, bool keep) {
        for (auto v : G[u]) {
            if (v == fa || son[u] == v)
                continue;
            dfs2(v, u, false);
        }

        if (son[u]) {
            dfs2(son[u], u, true);
            vis[son[u]] = true;
        }
        calc(u, fa, 1);
        vis[son[u]] = false;
        ans[u] = top;
        if (!keep) {
            calc(u, fa, -1);
            // top = 0;
        }
    };

    dfs2(1, 0, 0);
