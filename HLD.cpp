template <typename G> struct HLDecomposition {
  G &g;
  vector<int> sz, in, out, head, rev, par, d, vis, roots;

  HLDecomposition(G &g, vector<int> r = vector<int>())
      : g(g), d(g.size()), sz(g.size()), in(g.size()), out(g.size()),
        head(g.size()), rev(g.size()), par(g.size()) {
    if (empty(r)) {
      vector<bool> vis(g.size());
      for (int i = 0; i < g.size(); i++) {
        if (vis[i])
          continue;
        roots.emplace_back(i);
        queue<int> q;
        q.emplace(i);
        while (!empty(q)) {
          int x = q.front();
          vis[x] = true;
          q.pop();
          for (auto e : g[x]) {
            if (!vis[e])
              q.emplace(e);
          }
        }
      }
    } else
      roots = r;
  }

  void dfs_sz(int idx, int p) {
    par[idx] = p;
    sz[idx] = 1;
    if (g[idx].size() and g[idx][0] == p)
      swap(g[idx][0], g[idx].back());
    for (auto &to : g[idx]) {
      if (to == p)
        continue;
      d[to] = d[idx] + 1;
      dfs_sz(to, idx);
      sz[idx] += sz[to];
      if (sz[g[idx][0]] < sz[to])
        swap(g[idx][0], to);
    }
  }

  void dfs_hld(int idx, int par, int &times) {
    in[idx] = times++;
    rev[in[idx]] = idx;
    for (auto &to : g[idx]) {
      if (to == par)
        continue;
      head[to] = (g[idx][0] == to ? head[idx] : to);
      dfs_hld(to, idx, times);
    }
    out[idx] = times;
  }

  template <typename T>
  void dfs_hld(int idx, int par, int &times, vector<T> &v) {
    in[idx] = times++;
    rev[in[idx]] = idx;
    for (auto &to : g[idx]) {
      if (to == par) {
        v[in[idx]] = to.cost;
        continue;
      }
      head[to] = (g[idx][0] == to ? head[idx] : to);
      dfs_hld(to, idx, times, v);
    }
    out[idx] = times;
  }

  template <typename T>
  void dfs_hld(int idx, int par, int &times, vector<T> &v, vector<T> &a) {
    in[idx] = times++;
    rev[in[idx]] = idx;
    v[in[idx]] = a[idx];
    for (auto &to : g[idx]) {
      if (to == par)
        continue;
      head[to] = (g[idx][0] == to ? head[idx] : to);
      dfs_hld(to, idx, times, v, a);
    }
    out[idx] = times;
  }

  void build() {
    int t = 0;
    for (auto root : roots) {
      head[root] = root;
      dfs_sz(root, -1);
      dfs_hld(root, -1, t);
    }
  }

  template <typename T> vector<T> build(int root = 0) {
    vector<T> res(g.size());
    int t = 0;
    for (auto root : roots) {
      head[root] = root;
      dfs_sz(root, -1);
      dfs_hld(root, -1, t, res);
    }
    return res;
  }

  template <typename T> vector<T> build(vector<T> &a, int root = 0) {
    vector<T> res(g.size());
    for (auto root : roots) {
      head[root] = root;
      dfs_sz(root, -1);
      int t = 0;
      dfs_hld(root, -1, t, res, a);
    }
    return res;
  }

  int la(int v, int k) {
    while (1) {
      int u = head[v];
      if (in[v] - k >= in[u])
        return rev[in[v] - k];
      k -= in[v] - in[u] + 1;
      v = par[u];
    }
  }

  int lca(int u, int v) {
    for (;; v = par[head[v]]) {
      if (in[u] > in[v])
        swap(u, v);
      if (head[u] == head[v])
        return u;
    }
  }

  template <typename T, typename Q, typename F>
  T query(int u, int v, const T &e, const Q &q, const F &f, bool edge = false) {
    T l = e, r = e;
    for (;; v = par[head[v]]) {
      if (in[u] > in[v])
        swap(u, v), swap(l, r);
      if (head[u] == head[v])
        break;
      l = f(q(in[head[v]], in[v] + 1), l);
    }
    return f(f(q(in[u] + edge, in[v] + 1), l), r);
  }

  template <typename T, typename Q, typename Q2, typename F>
  T query(int u, int v, const T &e, const Q &q1, const Q2 &q2, const F &f,
          bool edge = false) {
    T l = e, r = e;
    for (;;) {
      if (head[u] == head[v])
        break;
      if (in[u] > in[v]) {
        l = f(l, q2(in[head[u]], in[u] + 1));
        u = par[head[u]];
      } else {
        r = f(q1(in[head[v]], in[v] + 1), r);
        v = par[head[v]];
      }
    }
    if (in[u] > in[v])
      return f(f(l, q2(in[v] + edge, in[u] + 1)), r);
    return f(f(l, q1(in[u] + edge, in[v] + 1)), r);
  }

  template <typename Q> void add(int u, int v, const Q &q, bool edge = false) {
    for (;; v = par[head[v]]) {
      if (in[u] > in[v])
        swap(u, v);
      if (head[u] == head[v])
        break;
      q(in[head[v]], in[v] + 1);
    }
    q(in[u] + edge, in[v] + 1);
  }

  constexpr int operator[](int k) { return in[k]; }

  constexpr int dist(int u, int v) { return d[u] + d[v] - 2 * d[lca(u, v)]; }

  // u -> v の unique path
  vector<int> road(int u, int v) {
    int l = lca(u, v);
    vector<int> a, b;
    for (; v != l; v = par[v])
      b.eb(v);
    for (; u != l; u = par[u])
      a.eb(u);
    a.eb(l);
    per(i, si(b), 0) a.eb(b[i]);
    return a;
  }

  int jump(int s, int t, int i) {
    if (!i)
      return s;
    int l = lca(s, t);
    int dst = d[s] + d[t] - d[l] * 2;
    if (dst < i)
      return -1;
    if (d[s] - d[l] >= i)
      return la(s, i);
    i -= d[s] - d[l];
    return la(t, d[t] - d[l] - i);
  }
};