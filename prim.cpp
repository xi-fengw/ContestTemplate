int prim(int n) {
    static int dist[N]{};
    memset(dist, 0x3f, sizeof dist);

    int res = 0;
    for (int i = 0; i <= n; i++) {
        int p = -1;
        for (int j = 1; j <= n; j++)
            if (vis[j] == 0 && (!~p || dist[j] < dist[p]))
                p = j;
        vis[p] = true;

        res += dist[p];
        for (int j = 1; j <= n; j++) {
            dist[j] = min(dist[j], g[p][j]);
        }
    }
    return res;
}
