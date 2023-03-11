# Trick

## MOD
```cpp
template<const int T>
struct ModInt {
    const static int mod = T;
    int x;
    ModInt(int x = 0) : x(x % mod) {}
    int val() { return x; }
    ModInt operator + (const ModInt &a) const { int x0 = x + a.x; return ModInt(x0 < mod ? x0 : x0 - mod); }
    ModInt operator - (const ModInt &a) const { int x0 = x - a.x; return ModInt(x0 < mod ? x0 + mod : x0); }
    ModInt operator * (const ModInt &a) const { return ModInt(1LL * x * a.x % mod); }
    ModInt operator / (const ModInt &a) const { return *this * a.inv(); }
    void operator += (const ModInt &a) { x += a.x; if (x >= mod) x -= mod; }
    void operator -= (const ModInt &a) { x -= a.x; if (x < 0) x += mod; }
    void operator *= (const ModInt &a) { x = 1LL * x * a.x % mod; }
    void operator /= (const ModInt &a) { *this = *this / a; }
    bool operator == (const ModInt &p) const {return x == p.x;}
    bool operator!=(const ModInt &p) const {return x != p.x;}
    friend std::istream &operator>>(std::istream &is, ModInt &a) { i64 v; is >> v; a = ModInt(v); return is;}
    friend std::ostream &operator<<(std::ostream &os, const ModInt &a) { return os << a.x;}

    ModInt pow(i64 n) const {
        ModInt res(1), mul(x);
        for (; n; n >>= 1, mul *= mul)
            if (n & 1) res *= mul;
        return res;
    }

    ModInt inv() const {
        int a = x, b = mod, u = 1, v = 0;
        while (b) {
            int t = a / b;
            std::swap(a -= t * b, b), std::swap(u -= t * v, v);
        }
        return (u < 0 ? u + mod : u);
    }
};

constexpr int P = 998244353;
using mint = ModInt<P>;
```
## Frac
```cpp
struct Frac {
  Frac(long long p_ = 0, long long q_ = 1) {
    auto g = std::__gcd(p_, q_);
    p = p_ / g;
    q = q_ / g;
  }
 
  Frac add() const { return Frac(p + q, q); }
 
  long long p, q;
};
 
Frac operator/(const Frac &a, const Frac &b) {
  return Frac(a.p * b.q, a.q * b.p);
}
 
bool operator<(const Frac &a, const Frac &b) {
  return a.p != b.p ? a.p < b.p : a.q < b.q;
}
```

## quick read

```cpp
inline int read() {  
    int x = 0, f = 0;
    char ch = getchar();
    while(ch > '9'|| ch < '0') {
        f |= (ch == '-');
        ch = getchar();
    }
    while(ch <= '9' && ch >= '0') {
        x = (x << 1) + (x << 3) + (ch ^ 48);
        ch = getchar();
    }
    return f ? -x : x;
}
```
## fastread
```cpp
namespace fast_IO {
    static const int N = 1000010;
    char buf[N], *s, *t;
    char getchar() {
        return (s == t) && (t = (s = buf) + fread(buf, 1, N, stdin)), s == t ? -1 : *s++;
    }
    int read() {
        int num = 0;
        char c;
        while ((c = getchar()) != '-' && c != '+' && (c < '0' || c > '9') && ~c);
        num = c ^ 48;
        while ((c = getchar()) >= '0' && c <= '9')
            num = (num << 1) + (num << 3) + (c ^ 48);
        return num;
    }
}
```

## Merge_sort

```cpp
void merge_sort(int q[], int l, int r)  
{  
    if (l >= r) return;  
    int mid = l + r >> 1;  
    merge_sort(q, l, mid);  
    merge_sort(q, mid + 1, r);  
    int k = 0, i = l, j = mid + 1;  
    while (i <= mid && j <= r)  
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];  
        else tmp[k ++ ] = q[j ++ ];  
    while (i <= mid) tmp[k ++ ] = q[i ++ ];  
    while (j <= r) tmp[k ++ ] = q[j ++ ];  
    for (i = l, j = 0; i <= r; i ++ , j ++ ) q[i] = tmp[j];  
}  
```

## 龟速乘

```cpp
ll quick_cheng(ll a1, ll a2 ){  
     ll sum = 0;  
     ll q = a2;  
     while(a1){  
         if(a1 & 1) sum = (sum + q) % c;  
        q = (q + q) % c;  
        a1 = a1 >> 1;  
     }  
     return sum % c;  
}
```

## __int128_t
```cpp
using i128 = __int128_t;  

std::istream &operator>>(std::istream &is, i128 &n) {  
  n = 0;  
  std::string s;  
  is >> s;  
  for (auto c : s) {  
    n = 10 * n + c - '0';  
  }  
  return is;  
}  

std::ostream &operator<<(std::ostream &os, i128 n) {  
  if (n == 0) {  
    return os << 0;  
  }  
  std::string s;  
  while (n > 0) {  
    s += '0' + n % 10;  
    n /= 10;  
  }  
  std::reverse(s.begin(), s.end());  
  return os << s;  
}  

i128 gcd(i128 a, i128 b) { return b ? gcd(b, a % b) : a; }
```

## fast read
```cpp
namespace io {
    constexpr int SIZE = 1 << 23;
    char buf[SIZE], *head, *tail;
    char get_char() {
        if (head == tail) tail = (head = buf) + fread(buf, 1, SIZE, stdin);
        return *head++;
    }
    template <typename T>
    T read() {
        T x = 0, f = 1;
        char c = get_char();
        for (; !isdigit(c); c = get_char()) (c == '-') && (f = -1);
        for (; isdigit(c); c = get_char()) x = x * 10 + c - '0';
        return x * f;
    }
    std::string read_s() {
        std::string str;
        char c = get_char();
        while (c == ' ' || c == '\n' || c == '\r') c = get_char();
        while (c != ' ' && c != '\n' && c != '\r') str += c, c = get_char();
        return str;
    }
    void print(i64 x) {
        if (x > 9) print(x / 10);
        putchar(x % 10 | '0');
    }
    void println(int x) { print(x), putchar('\n'); }
    struct Read {
        Read& operator>>(int& x) { return x = read<int>(), *this; }
        Read& operator>>(i64& x) { return x = read<i64>(), *this; }
        Read& operator>>(long double& x) { return x = stold(read_s()), *this; }
        Read& operator>>(std::string& x) { return x = read_s(), *this; }
    } in;
}  // namespace io;
```

# DP

## 二进制拆位优化多重背包
```cpp
constexpr int N = 20010;
int v[N], w[N];
i64 dp[N];
void solve() {
    int n, V;
    std::cin >> n >> V;
    int cnt = 0;
    for (int i = 1; i <= n; i ++ ) {
        int s, a, b, k = 1;
        std::cin >> a >> b >> s;
        while(k <= s)
            v[++cnt] = a * k, w[cnt] = b * k,s -= k, k <<= 1;
        if (s)
            v[++cnt] = a * s, w[cnt] = b * s, s = 0;
    }

    n = cnt;
    for (int i = 1; i <= cnt; i ++ )
        for (int j = V; j >= v[i]; j -- )
            dp[j] = std::max(dp[j], dp[j - v[i]] + w[i]);
    std::cout << dp[V] << "\n";
    return 0 ^ 0;
}
```
# String
## Manacher
```cpp
char s[N], g[N];
int n, ans, p[N];

void manacher() {
    int r = 0, mid = 0;
    for (int i = 1; i <= n; i++) {
        p[i] = i <= r ? std::min(r - i + 1, p[2 * mid - i]) : 1;
        while (g[i - p[i]] == g[i + p[i]]) ++p[i];
        if (i + p[i] - 1 > r) mid = i, r = i + p[i] - 1;
        ans = std::max(ans, p[i] - 1);
    }
}

void change() {
    n = strlen(s + 1) * 2;
    g[0] = 0;
    for (int i = 1; i <= n; i++) {
        if (i % 2) g[i] = 1;
        else g[i] = s[i >> 1];
    }
    g[++n] = 1, g[n + 1] = 2;
    manacher();
}
```
## PAM
```cpp
const int S = 26;
struct PAM {
    char s[N];
    bool vis[N];
    int all, son[N][S], fail[N], cnt[N], len[N], text[N], last, tot;
    int newnode(int l) {
        for (int i = 0; i < S; i++)son[tot][i] = 0;
        cnt[tot] = 0, len[tot] = l;
        return tot++;
    }
    void init() {
        last = all = tot = 0;
        newnode(0), newnode(-1);
        text[0] = -1, fail[0] = 1;
    }
    int getfail(int x) {
        while (text[all - len[x] - 1] != text[all])x = fail[x];
        return x;
    }
    void add(int w) {
        text[++all] = w;
        int x = getfail(last);
        if (!son[x][w]) {
            int y = newnode(len[x] + 2);
            fail[y] = son[getfail(fail[x])][w];
            son[x][w] = y;
            // printf("son[%d][%d]=%d\n", x, w, son[x][w]);
        }
        cnt[last = son[x][w]]++;
    }
    void count() {for (int i = tot - 1; ~i; i--)cnt[fail[i]] += cnt[i];}
    void dfs(int u) {
        rep(i, 0, 25)if (son[u][i]) {
            // printf("%d->%d\n", u, son[u][i]);
            dfs(son[u][i]);
        }
    }
    void build() {
        init();
        scanf("%s", s + 1);
        // printf("%s\n", s + 1);
        int m = strlen(s + 1);
        rep(i, 1, m)add(s[i] - 'a');
        // printf("tot=%d\n", tot);
        dfs(1);
    }
} P[5];

```
# Graph
## MST
```cpp
struct Edge {
    int a, b, w;
    bool operator<(const Edge &W) const {
        return w < W.w;
    }
} edges[M];
int find(int x) { return p[x] == x ? p[x] : p[x] = find(p[x]); }
int kruskal() {
    std::sort(edges, edges + m); std::iota(all(p), 0);
    int res = 0, cnt = 0; // res 表示的是最小生成树的边权之和， cnt表示的是边数
    for (int i = 0; i < m; i++) {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;
        //判断是不是在连通块里面，在连通块里面的话就更新边权和，更新数量
        a = find(a), b = find(b);
        if (a != b) p[a] = b, res += w, cnt++;
    }
    return (cnt < n - 1 ? INF : res);
}
```

## Prim
```cpp
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
```

## 严格次小生成树
```cpp
class SMST {
private:
    struct Edge {
        int v, nxt, w;
    } e[M * 2];

    int cnt, h[N];
    int par[N][22], dpth[N], maxx[N][22], minn[N][22];

public:
    void addedge(int u, int v, int val) {
        e[++cnt] = (Edge){v, h[u], val};
        h[u] = cnt;
    }

    void connect(int u, int v, int val) {
        addedge(u, v, val), addedge(v, u, val);
    }

    void dfs(int now, int fa) {
        dpth[now] = dpth[fa] + 1, par[now][0] = fa;
        minn[now][0] = -(1 << 30);
        for (int i = 1; (1 << i) <= dpth[now]; i++) {
            par[now][i] = par[par[now][i - 1]][i - 1];
            int kk[4] = {maxx[now][i - 1], maxx[par[now][i - 1]][i - 1],
                        minn[now][i - 1], minn[par[now][i - 1]][i - 1]};
            std::sort(kk, kk + 4), maxx[now][i] = kk[3];
            int ptr = 2;
            while (ptr >= 0 && kk[ptr] == kk[3]) ptr--;
            minn[now][i] = (ptr == -1 ? -(1 << 30) : kk[ptr]);
        }

        for (int i = h[now]; i; i = e[i].nxt) {
            if (e[i].v != fa) {
                maxx[e[i].v][0] = e[i].w;
                dfs(e[i].v, now);
            }
        }
    }

    int lca(int a, int b) {
        if (dpth[a] < dpth[b]) std::swap(a, b);

        for (int i = 21; i >= 0; i--) {
            if (dpth[par[a][i]] >= dpth[b])
                a = par[a][i];
        }
        if (a == b) return a;
        for (int i = 21; i >= 0; i--)
            if (par[a][i] != par[b][i])
                a = par[a][i], b = par[b][i];

        return par[a][0];
    }

    // 求a -> b的链上权值不等于val的最大边权
    int query(int a, int b, int val) {
        int res = -(1 << 30);
        for (int i = 21; i >= 0; i--) {
            if (dpth[par[a][i]] >= dpth[b]) {
                res = std::max(res, val == maxx[a][i] ? minn[a][i] : maxx[a][i]);
            a = par[a][i];
            }
        }
        return res;
    }
};
```

## Kruskal重构树
```cpp
void ex_kruskal() {
    for (int i = 1; i < N; i++) fa[i] = i;
    std::sort(edge + 1, edge + 1 + m);
    for (int i = 1; i <= m; i++) {
        int u = edge[i].u, v = edge[i].v, w = edge[i].w;
        if (find(u) == find(v)) continue;
        u = find(u), v = find(v);
        val[++n] = w;
        fa[n] = fa[u] = fa[v] = n;
        add(n, u), add(n, v);
    }
}
```

## 欧拉序
```cpp
    int tim = 0;
    std::vector<int> id(n + 1); // 欧拉序的序号
    std::function<void(int, int)> dfs=[&](int u, int fa) -> void {
        tim++;
        for (auto& v : adj[u]) {
            if (v == fa) continue;
            id[v] = tim;
            dfs(v, u);
        }
        tim++;
    };
    dfs(1, 0);

```

## SPFA
### 判负环
```cpp
bool spfa() {
    std::vector<bool> vis(n + 1);
    std::vector<int> cnt(n + 1), dist(n + 1);
    std::queue<int> q;
    for (int i = 1; i <= n; i++) {
        q.push(i);
        vis[i] = true;
    }

    while(q.size()) {
        int p = q.front();
        q.pop();
        vis[p] = false;

        for (auto& G : adj[p]) {
            int v = G.first, w = G.second;
            if (dist[v] > dist[p] + w);
            cnt[v] = cnt[p] + 1;
            if (cnt[v] >= n) return true;
            if (!vis[v]) {
                q.push(v);
                vis[v] = true;
            }
        }
    }
    return false;
}
```

## Dijkstra
```cpp
void dijkstra() {
    std::vector<bool> vis(n + 1);
    std::vector<int> dist(n + 1, 1E9);
    dist[1] = 0;
    #define arr std::pair<int, int>
    priority_queue<arr, vector<arr>, greater<arr>> heap;
    heap.push(make_pair(0, 1));
    while (heap.size()) {
        std::pair<int, int> t = heap.top();heap.pop();
        int ver = t.second, distance = t.first;
        if (vis[ver]) continue;
        vis[ver] = true;
        for (auto& G : adj[ver]) {
            int v = G.first, w = G.second;
            if (dist[v] > dist[ver] + w)
                dist[v] = dist[ver] + w, heap.push(make_pair(dist[v], v));
        }
    }
    #undef arr
}
```

## Tarjan
### 求LCA
```cpp
std::vector<std::vector<std::pair<int, int>>> adj(n + 1);//存边
std::vector<int> dist(n + 1); // 从lca到当前节点的距离
std::vector<int> p(n + 1); // dsu的操作
std::iota(p.begin(), p.end(), 0);
std::function<int(int)> find = [&] (int x) -> int {
    return p[x] == x ? p[x] : p[x] = find(p[x]);
};
std::vector<std::array<int, 2>> event[n + 1]; // 所连的节点,第几个询问
std::vector<bool> vis(n + 1); // 判断这个节点是不是访问过
for (int i = 1; i < n; i++) {
    int u, v, w;
    std::cin >> u >> v >> w;
    adj[u].emplace_back(v, w), adj[v].emplace_back(u, w);
}
// 预处理出所有的距离
std::function<void(int, int)> dfs = [&] (int u, int fa) -> void {
        for (auto& G : adj[u]) {
            int v = G.first, w = G.second;
            if (v == fa) continue;
            dist[v] = dist[u] + w;
            dfs(v, u);
        }
};
dfs(1, 0);

for (int i = 1; i <= m; i++) { // 离线
    int u, v;
    std::cin >> u >> v;
    event[u].push_back({v, i}), event[v].push_back({u, i});
}

std::vector<int> ans(m + 1);
std::function<void(int)> tarjan = [&] (int u) -> void {
    vis[u] = true;
    for (auto& G : adj[u]) { // 回u
        int v = G.first, w = G.second;
        if (vis[v]) continue;
        tarjan(v), p[v] = u;
    }
    for (auto& evt : event[u])  // 离u
        if (vis[evt[0]])
            ans[evt[1]]=dist[u]+dist[evt[0]]-dist[find(evt[0])]*2;
};
tarjan(1); // 从根节点开始
for (int i = 1; i <= m; i++)
    std::cout << ans[i] << "\n";
```

### 求强连通分量
```cpp
std::vector<std::vector<int>> adj(n + 1);

for (int i = 0; i < m; i++) {
    int u, v;
    std::cin >> u >> v;
    adj[u].push_back(v);
}

std::vector<int> dfn(n + 1), stk(n + 1), low(n + 1);
std::vector<int> scc(n + 1);
std::vector<bool> vis(n + 1);
int idx = 0, top = 0, cnt = 0;
// idx是时间戳，top是栈的指针，cnt是强连通分量的数量->缩点的个数

std::function<void(int)> tarjan = [&] (int u) -> void {
    dfn[u] = low[u] = ++ idx;
    stk[++top] = u;
    vis[u] = true;

    for (auto& v : adj[u]) {
        if (!dfn[v]) {
            tarjan(v);
            low[u] = std::min(low[u], low[v]);
        } else if (vis[v]) {
            low[u] = std::min(low[u], dfn[v]);
        }
    }

    if (dfn[u] == low[u]) {//时间戳和回溯点是同一个那就是缩点
        int v = 0;
        cnt++;
        do {
            v = stk[top--];
            vis[v] = false;
            scc[v] = cnt;
        } while(v != u);
    }
};

for (int i = 1; i <= n; i++) {
    if (!dfn[i])
        tarjan(i);
}
```

### 求割点
```cpp
std::vector<std::vector<int>> adj(n + 1);

for (int i = 1; i <= m; i++) {
    int u, v;
    std::cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
}

std::vector<int> dfn(n + 1), low(n + 1);
std::vector<bool> cut(n + 1);

int idx = 0;
int cnt = 0;
int root = 1;

std::function<void(int, int)> tarjan = [&] (int u, int fa) -> void {
    int tot = 0;
    dfn[u] = low[u] = ++idx;

    for (auto& v : adj[u]) {
        if (!dfn[v]) {
            tarjan(v, u);
            low[u] = std::min(low[v], low[u]);

            //不是根，但是回溯点大于等于时间戳，那就是割点
            if (low[v] >= dfn[u] && u != fa) {
                cut[u] = true;
            }

            if (u == fa) tot++;
        } else {
            low[u] = std::min(low[u], dfn[v]);
        }
    }

    // 如果是根并且有至少两个儿子满足性质，那它也是割点
    if (tot >= 2 && u == fa)
        cut[u] = true;
};


for (int i = 1; i <= n; i++) {
    if (!dfn[i]) {
        tarjan(i, i);
    }
}

```

### 割边
```cpp
std::vector<std::vector<int>> adj(n + 1);

for (int i = 0; i < m; i++) {
    int a, b;
    std::cin >> a >> b;
    adj[a].push_back(b);
    adj[b].push_back(a);
}

std::vector<std::pair<int, int>> edge;

int tot = 0;

std::vector<int> dfn(n + 1);
std::vector<int> low(n + 1);

std::function<void(int, int)> tarjan = [&] (int u, int fa) -> void {
    dfn[u] = low[u] = ++tot;

    for (auto& v : adj[u]) {
        if (dfn[v] && v != fa) {
            low[u] = std::min(low[u], dfn[v]);
        }

        if (!dfn[v]) {
            tarjan(v, u);
            if (dfn[u] < low[v]) {
                // 将割边存下来，u -> v, u 是标号小的， v是标号大的
                edge.emplace_back(std::min(u, v), std::max(u, v));
            }

            low[u] = std::min(low[u], low[v]);
        }
    }
};


for (int i = 1; i <= n; i++) {
    if (!dfn[i]) {
        tarjan(i, i);
    }
}

```

#### 求边双强连通分量
```cpp
std::vector<int> h(n + 1, -1), e(m * 2 + 1), ne(m * 2 + 1);
int idx = 0;
auto add = [&] (int u, int v) -> void {
    e[idx] = v, ne[idx] = h[u], h[u] = idx++;
};

for (int i = 0; i < m; i++) {
    int a, b;
    std::cin >> a >> b;
    add(a, b);
    add(b, a);
}

std::vector<int> stk(n + 1);
std::vector<int> dfn(n + 1);
std::vector<int> low(n + 1);
std::vector<int> dcc(n + 1);
std::vector<bool> vis(m * 2 + 1);
int cnt = 0, top = 0, p = 0;
std::function<void(int, int)> tarjan = [&] (int u, int edge) -> void {
    dfn[u] = low[u] = ++p;
    stk[++top] = u;

    for (int i = h[u]; ~i; i = ne[i]) {
        int v = e[i];
        if (!dfn[v]) {
            tarjan(v, i);
            low[u] = std::min(low[u], low[v]);

            if (low[v] > dfn[u]) {
                vis[i] = vis[i ^ 1] = true; // 防止走返祖边回去
            }
        } else if (i != (edge ^ 1)) {
            // 如果之前遍历过并且走的不是原来的反边，就更新回溯值
            low[u] = std::min(low[u], dfn[v]);
        }
    }

    if (dfn[u] == low[u]) {
        // 如果回溯值和自己出现的时间一致，就是一个边双强连通分量
        ++cnt;
        while(true) {
            int v = stk[top--];
            dcc[v] = cnt;
            if (u == v) break;
        }
    }
};

for (int i = 1; i <= n; i++) {
    if (!dfn[i]) tarjan(i, 0);
}
```

#### 求点双强连通分量
```cpp
std::vector<int> dfn(n + 1), low(n + 1);
std::vector<int> dcc[n + 1];
std::vector<int> stk(n + 1);
std::vector<bool> cut(n + 1);
int idx = 0, top = 0, cnt = 0;

std::function<void(int, int)> tarjan = [&] (int u, int fa) -> void {
    dfn[u] = low[u] = ++idx;
    stk[++top] = u;
    int son = 0;
    for (auto& v : adj[u]) {
        if (!dfn[v]) {
            son++;
            tarjan(v, u);

            low[u] = std::min(low[u], low[v]);
            if (low[v] >= dfn[u]) {
                if (u != fa || son > 1)
                    cut[u] = true;
                cnt ++;

                while(stk[top + 1] != v)
                    dcc[cnt].push_back(stk[top--]);
                dcc[cnt].push_back(u);
            }
        } else if (v != fa) {
            low[u] = std::min(low[u], dfn[v]);
        }
    }
    if (fa == u && son == 0)
        dcc[++cnt].push_back(u);
};

for (int i = 1; i <= n; i++) {
    if (!dfn[i])
        top = 0, tarjan(i, i);
}

```

##  Flow
###  maxflow
```cpp
template <class Cap> struct mf_graph {
  public:
    mf_graph() : _n(0) {}
    explicit mf_graph(int n) : _n(n), g(n) {}

    int add_edge(int from, int to, Cap cap) {
        assert(0 <= from && from < _n);
        assert(0 <= to && to < _n);
        assert(0 <= cap);
        int m = int(pos.size());
        pos.push_back({from, int(g[from].size())});
        int from_id = int(g[from].size());
        int to_id = int(g[to].size());
        if (from == to) to_id++;
        g[from].push_back(_edge{to, to_id, cap});
        g[to].push_back(_edge{from, from_id, 0});
        return m;
    }

    struct edge {
        int from, to;
        Cap cap, flow;
    };

    edge get_edge(int i) {
        int m = int(pos.size());
        assert(0 <= i && i < m);
        auto _e = g[pos[i].first][pos[i].second];
        auto _re = g[_e.to][_e.rev];
        return edge{pos[i].first, _e.to, _e.cap + _re.cap, _re.cap};
    }
    std::vector<edge> edges() {
        int m = int(pos.size());
        std::vector<edge> result;
        for (int i = 0; i < m; i++) {
            result.push_back(get_edge(i));
        }
        return result;
    }
    void change_edge(int i, Cap new_cap, Cap new_flow) {
        int m = int(pos.size());
        assert(0 <= i && i < m);
        assert(0 <= new_flow && new_flow <= new_cap);
        auto& _e = g[pos[i].first][pos[i].second];
        auto& _re = g[_e.to][_e.rev];
        _e.cap = new_cap - new_flow;
        _re.cap = new_flow;
    }

    Cap flow(int s, int t) {
        return flow(s, t, std::numeric_limits<Cap>::max());
    }
    Cap flow(int s, int t, Cap flow_limit) {
        assert(0 <= s && s < _n);
        assert(0 <= t && t < _n);
        assert(s != t);

        std::vector<int> level(_n), iter(_n);

        auto bfs = [&]() {
            std::fill(level.begin(), level.end(), -1);
            level[s] = 0;
            std::queue<int> que;
            que.push(s);
            while (!que.empty()) {
                int v = que.front();
                que.pop();
                for (auto e : g[v]) {
                    if (e.cap == 0 || level[e.to] >= 0) continue;
                    level[e.to] = level[v] + 1;
                    if (e.to == t) return;
                    que.push(e.to);
                }
            }
        };
        auto dfs = [&](auto self, int v, Cap up) {
            if (v == s) return up;
            Cap res = 0;
            int level_v = level[v];
            for (int& i = iter[v]; i < int(g[v].size()); i++) {
                _edge& e = g[v][i];
                if (level_v <= level[e.to] || g[e.to][e.rev].cap == 0) continue;
                Cap d =
                    self(self, e.to, std::min(up - res, g[e.to][e.rev].cap));
                if (d <= 0) continue;
                g[v][i].cap += d;
                g[e.to][e.rev].cap -= d;
                res += d;
                if (res == up) return res;
            }
            level[v] = _n;
            return res;
        };

        Cap flow = 0;
        while (flow < flow_limit) {
            bfs();
            if (level[t] == -1) break;
            std::fill(iter.begin(), iter.end(), 0);
            Cap f = dfs(dfs, t, flow_limit - flow);
            if (!f) break;
            flow += f;
        }
        return flow;
    }

    std::vector<bool> min_cut(int s) {
        std::vector<bool> visited(_n);
        std::queue<int> que;
        que.push(s);
        while (!que.empty()) {
            int p = que.front();
            que.pop();
            visited[p] = true;
            for (auto e : g[p]) {
                if (e.cap && !visited[e.to]) {
                    visited[e.to] = true;
                    que.push(e.to);
                }
            }
        }
        return visited;
    }

  private:
    int _n;
    struct _edge {
        int to, rev;
        Cap cap;
    };
    std::vector<std::pair<int, int>> pos;
    std::vector<std::vector<_edge>> g;
};
```

### MCMF
```cpp
const static int INF = std::numeric_limits<int>::max();
template <typename Cap, typename Cost>
struct primal_dual{
    struct edge{
        int to, rev, id;
        Cap cap;
        Cost cost;
        edge(int to, int rev, int id, Cap cap, Cost cost): to(to), rev(rev), id(id), cap(cap), cost(cost){}
    };

    int N, M;
    vector<vector<edge>> G;

    primal_dual(){}
    primal_dual(int N): N(N), M(0), G(N){}

    void add_edge(int from, int to, Cap cap, Cost cost){
        int id1 = G[from].size();
        int id2 = G[to].size();
        G[from].push_back(edge(to, id2, M, cap, cost));
        G[to].push_back(edge(from, id1, ~M, 0, -cost));
        M++;
    }

    pair<Cap, Cost> min_cost_flow(int s, int t, Cap F){
        Cap flow = 0;
        Cost cost = 0;
        vector<Cost> h(N, 0);
        while (flow < F){
            vector<Cap> m(N, INF);
            vector<Cost> d(N, INF);
            vector<int> pv(N, -1);
            vector<int> pe(N, -1);
            vector<bool> used(N, false);
            priority_queue<pair<Cost, int>, vector<pair<Cost, int>>, greater<pair<Cost, int>>> pq;
            pq.push(make_pair(0, s));
            d[s] = 0;
    
            while (!pq.empty()){
                int v = pq.top().second;
                pq.pop();
                if (!used[v]){
                    used[v] = true;
                    if (v == t) break;
                    int cnt = G[v].size();
                    for (int i = 0; i < cnt; i++){
                        int w = G[v][i].to;
                        if (!used[w] && G[v][i].cap > 0){
                            Cost tmp = G[v][i].cost - h[w] + h[v];
                            if (d[w] > d[v] + tmp){
                                d[w] = d[v] + tmp;
                                m[w] = min(m[v], G[v][i].cap);
                                pv[w] = v, pe[w] = i;
                                pq.push(make_pair(d[w], w));
                            }
                        }
                    }
                }
            }
            
            if (!used[t]) break;

            for (int i = 0; i < N; i++)
                if (used[i])
                    h[i] -= d[t] - d[i];

            Cap c = min(m[t], F - flow);
            for (int i = t; i != s; i = pv[i]){
                G[pv[i]][pe[i]].cap -= c;
                G[i][G[pv[i]][pe[i]].rev].cap += c;
            }
            flow += c;
            cost += c * (- h[s]);
        }

        return make_pair(flow, cost);
    }
};
```

# Data Structure

##  Sparse Table
```cpp
#pragma once

#include <cassert>
#include <limits>
#include <vector>
using namespace std;

template <typename T>
struct SparseTable {
  inline static constexpr T INF = numeric_limits<T>::max() / 2;
  int N;
  vector<vector<T> > table;
  T f(T a, T b) { return min(a, b); }
  SparseTable() {}
  SparseTable(const vector<T> &v) : N(v.size()) {
    int b = 1;
    while ((1 << b) <= N) ++b;
    table.push_back(v);
    for (int i = 1; i < b; i++) {
      table.push_back(vector<T>(N, INF));
      for (int j = 0; j + (1 << i) <= N; j++) {
        table[i][j] = f(table[i - 1][j], table[i - 1][j + (1 << (i - 1))]);
      }
    }
  }
  // [l, r)
  T query(int l, int r) {
    assert(0 <= l and l <= r and r <= N);
    if (l == r) return INF;
    int b = 31 - __builtin_clz(r - l);
    return f(table[b][l], table[b][r - (1 << b)]);
  }
};

/**
 * @brief Sparse Table
 */

```

##  Fenwick

```cpp
template <typename T>
struct Fenwick {
    const int n;
    std::vector<T> tr;
    Fenwick(int n) : n(n), tr(n + 1) {}

    void add(int x, T v) {for (int i=x;i<=n;i+=i&-i) tr[i] += v;}

    T sum(int x) {
        T ans = 0;
        for (int i = x; i; i-=i&-i) ans += tr[i];
        return ans;
    }

    T rangeSum(int l, int r) {return sum(r) - sum(l);}

    int query(T s) { // 查询1~pos的和小于等于s
        int pos = 0;
        for (int j = 18; j >= 0; j -- )
            if ((pos + (1ll << j) < n) && tr[pos + (1ll << j)] <= s )
                pos = (pos + (1 << j)), s -= tr[pos];
        return pos;
    }
};
```

### 二维数点
```cpp
void solve() {
    int n, q;
    std::cin >> n >> q;
    std::vector<int> vx;
    std::vector<std::array<int, 4>> event; // 将询问离线
    for (int i = 0; i < n; i++ ) {
        int x, y; std::cin >> x >> y;
        vx.push_back(x);
        event.push_back({y, 0, x});
    }
    for (int i = 1; i <= q; i ++ ) {
        int x, y, xx, yy;
        std::cin >> x >> xx >> y >> yy;
        event.push_back({yy, 2, xx, i});
        event.push_back({yy, 1, x - 1, i});
        event.push_back({y - 1, 1, xx, i});
        event.push_back({y - 1, 2, x - 1, i});
    }
    std::sort(all(vx));
    std::sort(all(event));
    vx.erase(std::unique(all(vx)), vx.end());
    int len = vx.size();
    std::vector<i64> ans(q + 1);
    Fenwick<i64> bit(len + 1);

    for (auto evt : event) {
        if (evt[1] == 0) {
            int y = std::lower_bound(all(vx), evt[2]) - vx.begin() + 1;
            bit.add(y, 1);
        } else {
            int y = std::upper_bound(all(vx), evt[2]) - vx.begin();
            int tmp = bit.sum(y);
            ans[evt[3]] += tmp * (evt[1] == 1 ? -1 : 1);
        }
    }
    for (int i = 1; i <= q; i ++ )
        std::cout << ans[i] << "\n";
}
```

##  SegmentTree
###  Lazy Tag
```cpp
// 区间最小值
struct Tag {
    int tag1;
};
struct Info {
    int minv;
};
Info operator + (const Info& a, const Info& b) {
    Info c;
    c.minv = std::min(a.minv, b.minv);
    return c;
}
struct Node {
    Info val; Tag tag;
} tr[N << 2];

void build(int u, int l, int r) {
    if (l == r) return void (tr[u].val.minv = a[l]);
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
    tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
}

void settag(int u, int v) {tr[u].val.minv = tr[u].tag.tag1 = v;}
void push(int u) {
    if (!tr[u].tag.tag1) return ;
    settag(u << 1, tr[u].tag.tag1), settag(u << 1 | 1, tr[u].tag.tag1);
    tr[u].tag.tag1 = 0;
}

void change(int u, int l, int r, int ln, int rn, int v) {
    if (l >= ln && r <= rn) return void(settag(u, v));
    push(u);
    int mid = l + r >> 1;
    if (mid >= ln) change(u << 1, l, mid, ln, rn, v);
    if (mid < rn) change(u << 1 | 1, mid + 1, r, ln, rn, v);
    tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
}

int query(int u, int l, int r, int ln, int rn) {
    if (l >= ln && r <= rn) return tr[u].val.minv;
    int mid = l + r >> 1; push(u);
    int ans = inf;
    if (mid >= ln) ans = min(ans, query(u << 1, l, mid, ln, rn)) ;
    if (mid < rn) ans = min(ans, query(u << 1 | 1, mid + 1, r, ln, rn));
    return ans;
}

//区间加
struct Seg {
    struct Node {
        int l, r;
        i64 val, tag;
    };
    const int n;
    std::vector<Node> tr;

    Seg(int n) : n(n), tr(4 << std::__lg(n)) {
        std::function<void(int, int, int)> build = [&] (int u, int l, int r) -> void {
            tr[u] = {l, r};
            if (l == r) return ;
            int mid = (l + r) >> 1;
            build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
            tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
        };
        build(1, 1, n);
    }

    void spread(int u, i64 x) {
        tr[u].val += x * (tr[u].r - tr[u].l + 1);
        tr[u].tag += x;
    }

    void push(int u) {
        if (!tr[u].tag) return ;
        spread(u << 1, tr[u].tag), spread(u << 1 | 1, tr[u].tag);
        tr[u].tag = 0;
    }

    void modify(int u, int pos, i64 x) {
        if (tr[u].l == tr[u].r && tr[u].l == pos)
            return void(tr[u].val = x);
        int mid = (tr[u].l + tr[u].r) >> 1;
        if (mid >= pos) modify(u << 1, pos, x);
        else modify(u << 1 | 1, pos, x);
        tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
    }

    void change(int u, int l, int r, i64 x) {
        if (tr[u].l >= l && tr[u].r <= r) return void(spread(u, x));
        push(u);
        int mid = (tr[u].l + tr[u].r) >> 1;
        if (mid >= l) change(u << 1, l, r, x);
        if (mid < r) change(u << 1 | 1, l, r, x);
        tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
    }

    i64 query(int u, int l, int r) {
        if (tr[u].l >= l && tr[u].r <= r) return tr[u].val;
        push(u);
        int mid = (tr[u].l + tr[u].r) >> 1;
        i64 ans = 0;
        if (mid >= l) ans += query(u << 1, l, r);
        if (mid < r) ans += query(u << 1 | 1, l, r);
        return ans;
    }
};
```

### 子段和
```cpp
struct Segment {
    const int n;
    struct Node {
        int l, r;
        i64 pre, suf, mx, val;

        Node operator + (const Node& lhs) {
            Node rhs;
            rhs.l = this -> l, rhs.r = lhs.r;
            rhs.val = this -> val + lhs.val;
            rhs.pre = std::max(this -> pre, this -> val + lhs.pre);
            rhs.suf = std::max(lhs.suf, this -> suf + lhs.val);
            rhs.mx = std::max({this -> mx, this -> suf + lhs.pre, lhs.mx});
            return rhs;
        }

    };

    std::vector<Node> tr;

    Segment(int n) : n(n), tr(4 << std::__lg(n)) {
        std::function<void(int, int, int)> build = [&] (int u, int l, int r) {
            tr[u].l = l, tr[u].r = r;
            if (l == r) return ;
            int mid = (l + r) >> 1;
            build(u << 1, l, mid);
            build(u << 1 | 1, mid + 1, r);
            tr[u] = tr[u << 1] + tr[u << 1 | 1];
        };
        build(1, 1, n);
    }

    void modify(int u, int pos, i64 x) {
        if (tr[u].l == tr[u].r && tr[u].l == pos) {
            tr[u].val = x;
            tr[u].pre = x;
            tr[u].suf = x;
            tr[u].mx = x;
            return ;
        }
        int mid = (tr[u].l + tr[u].r) >> 1;
        if (mid >= pos) modify(u << 1, pos, x);
        else modify(u << 1 | 1, pos, x);
        tr[u] = tr[u << 1] + tr[u << 1 | 1];
    }

    Node query(int u, int l, int r) {
        if (tr[u].l >= l && tr[u].r <= r) return tr[u];
        int mid = (tr[u].l + tr[u].r) >> 1;
        if (mid >= l && mid < r)
            return query(u << 1, l, r) + query(u << 1 | 1, l, r);
        if (mid >= l) return query(u << 1, l, r);
        if (mid < r) return query(u << 1 | 1, l, r);
    }

};
```

### 线段树二分
```cpp
int search(int u, int l, int r, i64 d) {
    if (tr[u].l >= l && tr[u].r <= r) {
        if (tr[u].val < d) return -1;
        if (tr[u].l == tr[u].r) return tr[u].l;
        int mid = (tr[u].l + tr[u].r) >> 1;
        if (tr[u << 1].val >= d) return search(u << 1, l, mid, d);
        else return search(u << 1 | 1, mid + 1, r, d);
    }
    int mid = (tr[u].l + tr[u].r) >> 1;
    if (r <= mid) return search(u << 1, l, r, d);
    else if (mid < l) return search(u << 1 | 1, l, r, d);
    else {
        int pos = search(u << 1, l, mid, d);
        if (pos == -1) return search(u << 1 | 1, mid + 1, r, d);
        else return pos;
    }
}
```
### SegmentTree with Matrix
```cpp
struct Matrix {
    mint a[3][3];
    // Matrix() { memset(a, 0, sizeof a); }
    void init() {
        a[1][1] = a[2][2] = 1,
        a[1][2] = a[2][1] = 0;
    }

    bool check() {
        return a[1][1].val() == 1 && a[2][2].val() == 1
            && a[1][2].val() == 0 && a[2][1].val() == 0;
    }

    Matrix operator+(const Matrix b) {
        Matrix res;
        for (int i = 1; i <= 2; ++i)
            for (int j = 1; j <= 2; ++j)
                res.a[i][j] = a[i][j] + b.a[i][j];
        return res;
    }

    Matrix operator*(const Matrix b) {
        Matrix res;
        for (int k = 1; k <= 2; ++k)
            for (int i = 1; i <= 2; ++i)
                for (int j = 1; j <= 2; ++j)
                    res.a[i][j] = res.a[i][j] + a[i][k] * b.a[k][j];
        return res;
    }

    Matrix operator^(int b) {
        Matrix res, Base;
        for (int i = 1; i <= 2; ++i)
            res.a[i][i] = 1;
        for (int i = 1; i <= 2; ++i)
            for (int j = 1; j <= 2; ++j)
                Base.a[i][j] = a[i][j];

        for (; b; b >>= 1, Base = Base * Base)
            if (b & 1) res = res * Base;
        return res;
    }
} base, res;

int a[N]; // 递推式的下标

struct Info {
    Matrix v, tag;
}tr[N << 2];

void pull(int u) { tr[u].v = tr[u << 1].v + tr[u << 1 | 1].v;}

void settag(int u, Matrix x) {
    tr[u].v = tr[u].v * x;
    tr[u].tag = tr[u].tag * x;
}

void push(int u) {
    if (tr[u].tag.check()) return ;
    settag(u << 1, tr[u].tag);
    settag(u << 1 | 1, tr[u].tag);
    tr[u].tag.init();
}

void build(int u, int l, int r) {
    tr[u].tag.init();
    if (l == r) {
        if (a[l] == 1) tr[u].v.a[1][1] = 1;
        else if (a[l] == 2) tr[u].v.a[1][1] = tr[u].v.a[1][2] = 1;
        else tr[u].v = res * (base ^ (a[l] - 2));
        return ;
    }
    int mid = (l + r) >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
    pull(u);
}

void modify(int u, int l, int r, int ln, int rn, Matrix x) {
    if (l >= ln && r <= rn) return void(settag(u, x));
    int mid = (l + r) >> 1;
    push(u);
    if (mid >= ln) modify(u << 1, l, mid, ln, rn, x);
    if (mid < rn) modify(u << 1 | 1, mid + 1, r, ln, rn, x);
    pull(u);
}

mint query(int u, int l, int r, int ln, int rn) {
    if (l >= ln && r <= rn) return tr[u].v.a[1][1];
    int mid = l + r >> 1;
    push(u);
    mint ans = 0;
    if (mid >= ln) ans = (ans + query(u << 1, l, mid, ln, rn));
    if (mid < rn) ans = (ans + query(u << 1 | 1, mid + 1, r, ln, rn));
    return ans;
}
```

### 扫描线

求面积并
```cpp
/*线段树*/
std::vector<int> vx;
struct Tag {
    int add;
};
struct Info {
    int minv, mincnt;
};
Info operator + (const Info& a, const Info& b) {
    Info c;
    c.minv = std::min(a.minv, b.minv);
    if (a.minv == b.minv) c.mincnt = a.mincnt + b.mincnt;
    else c.mincnt = (a.minv > b.minv ? b.mincnt : a.mincnt);
    return c;
}

struct Seg {
    struct Node {
        Tag tag;
        Info val;
    };
    const int n;
    std::vector<Node> tr;
    Seg(int n) : n(n), tr(8 << std::__lg(n)) {
        std::function<void(int, int, int)> build = [&] (int u, int l, int r){
            if (l == r) return void(tr[u].val = {0, vx[r] - vx[r - 1]});
            int mid = (l + r) >> 1;
            build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
            tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
        };
        build(1, 1, n);
    }

    void settag(int u, int v) {tr[u].val.minv += v, tr[u].tag.add += v;}

    void push(int u) {
        if (!tr[u].tag.add) return ;
        settag(u << 1, tr[u].tag.add);
        settag(u << 1 | 1, tr[u].tag.add);
        tr[u].tag.add = 0;
    }

    void change(int u, int l, int r, int ln, int rn, int v) {
        if (l >= ln && r <= rn) {
            settag(u, v);
            return ;
        }
        int mid = (l + r) >> 1;
        push(u);
        if (mid >= ln) change(u << 1, l, mid, ln, rn, v);
        if (mid < rn) change(u << 1 | 1, mid + 1, r, ln, rn, v);
        tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;
    }
};
/*主体部分*/
int n;
std::cin >> n;
std::vector<std::array<int, 4>> event;

for (int i = 0; i < n; i ++ ) {
    int x, xx, y, yy;
    std::cin >> x >> y >> xx >> yy;
    vx.push_back(x);
    vx.push_back(xx);
    event.push_back({y, 1, x, xx});
    event.push_back({yy, -1, x, xx});
}

std::sort(all(vx));
sort(all(event));
vx.erase(std::unique(all(vx)), vx.end());
int m = vx.size() - 1;

Seg SGT(m);
i64 ans = 0;
int prev = 0;
int tot = SGT.tr[1].val.mincnt;
for (auto evt : event) {
    i64 res = tot;
    if (SGT.tr[1].val.minv == 0) res = tot - SGT.tr[1].val.mincnt;
    ans += (evt[0] - prev) * 1ll * res;
    prev = evt[0];
    int x1 = std::lower_bound(all(vx), evt[2]) - vx.begin() + 1;
    int x2 = std::lower_bound(all(vx), evt[3]) - vx.begin();
    SGT.change(1, 1, m, x1, x2, evt[1]);
}

std::cout << ans << "\n";
```

```cpp
struct Node {  
    i64 y, x1, x2;  
    int k;  
    bool operator <(const Node& t) const {return y < t.y;}  
}seg[N << 1];  
std::vector<i64> xs;  
struct Segment {  
    struct Tree {  
        int l, r;  
        i64 val1, val2;  
        int tag;  
    };  
    const int n;  
    std::vector<Tree> tr;  

    Segment(int n) : n(n), tr(8 << std::__lg(n)) {}  
    void build(int u, int l, int r) {  
        tr[u] = {l, r};  
        if (l == r) return ;  
        int mid = (l + r) >> 1;  
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);  
    }  

    void pull(int u) {  
        if (tr[u].tag) tr[u].val1 = xs[tr[u].r] - xs[tr[u].l - 1];  
        else if (tr[u].l == tr[u].r) tr[u].val1 = 0;  
        else tr[u].val1 = tr[u << 1].val1 + tr[u << 1 | 1].val1;  

        /*至少覆盖两次及以上的*/
        if (tr[u].tag >= 2) tr[u].val2 = xs[tr[u].r] - xs[tr[u].l - 1];  
        else if (tr[u].l == tr[u].r) tr[u].val2 = 0;  
        else if (tr[u].tag) tr[u].val2 = tr[u << 1].val1 + tr[u << 1 | 1].val1;  
        else tr[u].val2 = tr[u << 1].val2 + tr[u << 1 | 1].val2;  
    }  

    void modify(int u, int l, int r, int v) {  
        if (tr[u].l >= l && tr[u].r <= r) {  
            tr[u].tag += v;  
            pull(u);  
            return ;  
        }  
        int mid = (tr[u].l + tr[u].r) >> 1;  
        if (mid >= l) modify(u << 1, l, r, v);  
        if (mid < r) modify(u << 1 | 1, l, r, v);  
        pull(u);  
    }  
};  

void solve() {  
    int n;
    std::cin >> n;  
    int idx = 0;  
    for (int i = 0; i < n; i ++ ) {  
        i64 x1, x2, y1, y2;  
        std::cin >> x1 >> y1 >> x2 >> y2;  
        seg[idx ++ ] = {y1, x1, x2, 1};  
        seg[idx ++ ] = {y2, x1, x2, -1};  
        xs.push_back(x1), xs.push_back(x2);  
    }  

    std::sort(seg, seg + idx);  
    std::sort(xs.begin(), xs.end());  
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    auto find = [&] (i64 x) -> int {  
        return std::lower_bound(xs.begin(), xs.end(), x) - xs.begin();  
    };  

    Segment SGT(n + 1);  
    SGT.build(1, 1, int(xs.size()) - 1);  
    i64 res = 0;  
    for (int i = 0; i < idx; i ++ ) {  
        if (i) res += SGT.tr[1].val2 * (seg[i].y - seg[i - 1].y);  
        SGT.modify(1, find(seg[i].x1) + 1, find(seg[i].x2), seg[i].k);  
    }  
    std::cout << res << "\n";  
    xs.clear();  
}
```

### 扫描线求周长

```cpp
struct pic {  
    int y, x1, x2;  
    int k;  
    bool operator <(const pic& t) const {  
        if (y == t.y) return k > t.k;  
        return y < t.y;  
    }  
}seg[2][N << 1];  
​  
vector<int> alls[2];  
​  
struct Seg {  
    struct Tree {  
        int l, r;  
        int cnt, val;  
    };  
​  
    vector<Tree> tr;  
    const int n;  
​  
    Seg(int n) : n(n), tr(n << 3) {}  
​  
    void build(int u, int l, int r) {  
        tr[u].l = l, tr[u].r = r;  
        if (l == r) return ;  
        int mid = (l + r) >> 1;  
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);  
    }  
​  
    void pushup(int u, int k) {  
        if (tr[u].cnt)
            tr[u].val = alls[k][tr[u].r] - alls[k][tr[u].l - 1];  
        else if (tr[u].l == tr[u].r) tr[u].val = 0;  
        else tr[u].val = tr[u << 1].val + tr[u << 1 | 1].val;  
    }  
​  
    void modify(int u, int l, int r, int v, int k) {  
        if (tr[u].l >= l && tr[u].r <= r) {  
            tr[u].cnt += v;  
            pushup(u, k);  
            return ;  
        }  
        int mid = (tr[u].l + tr[u].r) >> 1;  
        if (mid >= l) modify(u << 1, l, r, v, k);  
        if (mid < r) modify(u << 1 | 1, l, r, v, k);  
        pushup(u, k);  
    }  
};  
​  
int find(int x, int k) {  
    return std::lower_bound(alls[k].begin(), alls[k].end(), x) - alls[k].begin();  
}  
​  
int main() {  
    int n;  
    scanf("%d", &n);  
    for (int i = 0; i < n; i ++ ) {  
        int x1, y1, x2, y2;  
        scanf("%d%d%d%d", &x1, &y1, &x2, &y2);  
        seg[0][i << 1] = {x1, y1, y2, 1};  
        seg[0][i << 1 | 1] = {x2, y1, y2, -1};  
        seg[1][i << 1] = {y1, x1, x2, 1};  
        seg[1][i << 1 | 1] = {y2, x1, x2, -1};  
        alls[0].push_back(y1), alls[0].push_back(y2);  
        alls[1].push_back(x1), alls[1].push_back(x2);  
    }  
​  
    sort(alls[0].begin(), alls[0].end());  
    sort(alls[1].begin(), alls[1].end());  
    alls[1].erase(unique(alls[1].begin(), alls[1].end()), alls[1].end());  
    alls[0].erase(unique(alls[0].begin(), alls[0].end()), alls[0].end());  
​  
    long long ans = 0;  
    for (int i = 0; i < 2; i ++ ) {  
        sort(seg[i], seg[i] + n * 2);  
        Seg SGT(int(alls[i].size()) - 1);  
        int M = alls[i].size() - 1;  
        SGT.build(1, 1, M);  
        int last = 0;  
        for (int j = 0; j < n * 2; j ++ ) {  
            SGT.modify(1, find(seg[i][j].x1, i) + 1,
            find(seg[i][j].x2, i), seg[i][j].k, i);  
            ans += abs(SGT.tr[1].val - last);  
            last = SGT.tr[1].val;  
        }  
    }  
    printf("%lld\n", ans);  
}
```

### 扫描线求体积

```cpp
struct Node {  
    int y, x1, z1, x2, z2;  
    int k;  
    bool operator< (const Node& t) const {  
        return y < t.y;  
    }  
​  
}seg1[N << 1], seg2[N << 1];  
​  
std::vector<int> xs, zs;  
​  
struct Segment {  
    const int n;  
    struct Tree {  
        int l, r;  
        int tag;  
        int val1, val2, val3;  
    };  
​  
    std::vector<Tree> tr;  
​  
    Segment(int n) : n(n), tr(8 << std::__lg(n)) {  
        std::function<void(int, int, int)> build = [&] (int u, int l, int r) -> void {  
            tr[u] = {l, r};  
            if (l == r) return ;  
            int mid = (l + r) >> 1;  
            build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);  
        };  
        build(1, 1, n);  
    }  
​  
    void pushup(int u) {  
        int val = xs[tr[u].r] - xs[tr[u].l - 1];  
        if (tr[u].tag == 0) {  
            if (tr[u].l == tr[u].r)
                tr[u].val1 = tr[u].val2 = tr[u].val3 = 0;  
            else {  
                tr[u].val1 = tr[u << 1].val1 + tr[u << 1 | 1].val1;  
                tr[u].val2 = tr[u << 1].val2 + tr[u << 1 | 1].val2;  
                tr[u].val3 = tr[u << 1].val3 + tr[u << 1 | 1].val3;  
            }  
        } else if (tr[u].tag == 1) {  
            if (tr[u].l == tr[u].r)
                tr[u].val1 = val, tr[u].val2 = tr[u].val3 = 0;  
            else {  
                tr[u].val2 = tr[u << 1].val1 + tr[u << 1 | 1].val1;  
                tr[u].val3 = tr[u << 1].val3 + tr[u << 1 | 1].val3
                            + tr[u << 1].val2 + tr[u << 1 | 1].val2;  
                tr[u].val1 = val - tr[u].val2 - tr[u].val3;  
            }  
        } else if (tr[u].tag == 2) {  
            if (tr[u].l == tr[u].r)
                tr[u].val2 = val, tr[u].val1 = tr[u].val3 = 0;  
            else {  
                tr[u].val1 = 0;  
                tr[u].val3 = tr[u << 1].val1 + tr[u << 1 | 1].val1
                            + tr[u << 1].val2 + tr[u << 1 | 1].val2
                            + tr[u << 1 | 1].val3 + tr[u << 1].val3;  
                tr[u].val2 = val - tr[u].val3;  
            }  
        } else {  
            tr[u].val3 = val;  
            tr[u].val1 = tr[u].val2 = 0;  
        }  
    }  
​  
    void modify(int u, int l, int r, int v) {  
        if (tr[u].l >= l && tr[u].r <= r) {  
            tr[u].tag += v;  
            pushup(u);  
            return ;  
        }  
        int mid = (tr[u].l + tr[u].r) >> 1;  
        if (mid >= l) modify(u << 1, l, r, v);  
        if (mid < r) modify(u << 1 | 1, l, r, v);  
        pushup(u);  
    }  
};  
​  
void solve() {  
    int n; scanf("%d", &n);  
    xs.clear(), zs.clear();  
    int idx = 0;  
    for (int i = 0; i < n; i ++ ) {  
        int x1, y1, z1, x2, y2, z2;  
        scanf("%d%d%d%d%d%d", &x1, &y1, &z1, &x2, &y2, &z2);  
        seg1[idx ++ ] = {y1, x1, z1, x2, z2, 1};  
        seg1[idx ++ ] = {y2, x1, z1, x2, z2, -1};  
        xs.push_back(x1), xs.push_back(x2);  
        zs.push_back(z1), zs.push_back(z2);  
    }  
​  
    std::sort(seg1, seg1 + idx);  
    std::sort(all(xs));  
    std::sort(all(zs));  
    xs.erase(std::unique(all(xs)), xs.end());  
    zs.erase(std::unique(all(zs)), zs.end());  
​  
    auto find = [&] (int x) -> int {  
        return std::lower_bound(all(xs), x) - xs.begin();  
    };  
​  
    i64 ans = 0;  
    for (int i = 0; i < SZ(zs); i ++ ) {  
        int cnt = 0;  
        for (int j = 0; j < idx; j ++ ) {  
            if (seg1[j].z1 <= zs[i] && seg1[j].z2 > zs[i]) {  
                seg2[cnt ++ ] = seg1[j];  
            }  
        }  
        std::sort(seg2, seg2 + cnt);  
        Segment SGT(SZ(xs) - 1);  
        i64 res = 0;  
        for (int j = 0; j < cnt; j ++ ) {  
            if (j) res+=1ll*SGT.tr[1].val3*(seg2[j].y-seg2[j - 1].y);  
            SGT.modify(1,find(seg2[j].x1)+1,find(seg2[j].x2),seg2[j].k);  
        }  
        ans += res * (zs[i + 1] - zs[i]);  
    }  
    printf("%lld\n", ans);  
}

```

## 主席树
```cpp
int n, m;
int a[N], b[N];

struct Segmenttree { // 动态开点
    int ls[N << 5], rs[N << 5], val[N << 5], rt[N << 5];
    int idx = 0;

    void build(int& u, int l, int r) {
        u = ++ idx;
        if (l == r) return ;
        int mid = (l + r) >> 1;
        build(ls[u], l, mid), build(rs[u], mid + 1, r);
    }

    void modify(int& u, int pre, int l, int r, int pos) {
        u = ++ idx;
        ls[u] = ls[pre], rs[u] = rs[pre], val[u] = val[pre] + 1;
        if (l == r) return ;
        int mid = (l + r) >> 1;
        if (mid >= pos) modify(ls[u], ls[pre], l, mid, pos);
        else modify(rs[u], rs[pre], mid + 1, r, pos);
    }

    int query(int l, int r, int ln, int rn, int k) {
        int x = val[ls[rn]] - val[ls[ln]];
        if (l == r) return b[l];
        int mid = (l + r) >> 1;
        if (x >= k) return query(l, mid, ls[ln], ls[rn], k);
        else return query(mid + 1, r, rs[ln], rs[rn], k - x);
   }
}SGT;

int main() {
    cin >> n >> m;

    rep(i,1,n + 1) cin >> a[i], b[i] = a[i];

    std::sort(b + 1, b + 1 + n);
    int len = std::unique(b + 1, b + 1 + n) - b - 1;
    SGT.build(SGT.rt[0], 1, len);
    rep(i,1,n + 1) {
        int pos = std::lower_bound(b + 1, b + 1 + len, a[i]) - b;
        SGT.modify(SGT.rt[i], SGT.rt[i - 1], 1, len, pos);
    }

    //求区间第k大
    rep(i,0,m) {
        int l, r, k;
        cin >> l >> r >> k;
        printf("%d\n", SGT.query(1, len, SGT.rt[l - 1], SGT.rt[r], k));
    }

    return 0;
}
```
## 平衡树

###  Treap

```cpp
struct Treap {	//旋转  
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

    void zig(int& p) {   //左旋  
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
```

###  Splay
```cpp
// 实现了插入和删除的功能
int root;
struct Splay {
    struct Node {
        int ch[2];
        int val, siz, cnt;
        int fa;
    };

    const int n;
    int idx;
    std::vector<Node> tr;

    Splay(int n) : n(n), tr(n) {}

    inline void newnode(int& u, int fa, int val) { // 创建新节点
        tr[u = ++idx].val = val;
        tr[u].fa = fa;
        tr[u].siz = tr[u].cnt = 1;
    }
    // 判断是左儿子还是在右儿子上
    inline bool check(int u, int fa) {
        return tr[fa].ch[1] == u;
    }

    inline void connect(int u, int fa, int s) { // 建立父子关系
        tr[fa].ch[s] = u,tr[u].fa = fa;
    }

    void update(int u) {
        tr[u].siz = tr[tr[u].ch[0]].siz + tr[tr[u].ch[1]].siz + tr[u].cnt;
    }

    void rotate(int u) {
        int fa = tr[u].fa, pf = tr[fa].fa, k = check(u, fa);
        //左旋拎右左挂右， 右旋拎左右挂左
        connect(tr[u].ch[k ^ 1], fa, k);
        // 将旋转节点的左/右儿子和它的父亲建立父子关系
        connect(u, pf, check(fa, pf)); // 将旋转节点和祖父节点建立父子关系
        connect(fa, u, k ^ 1); // 将原先的父亲和旋转节点建立父子关系
        update(fa), update(u); // 更新父节点和当前节点
    }

    void splaying(int u, int top) { // 伸展
        if (!top) root = u;
        while(tr[u].fa != top) {
            int fa = tr[u].fa, pf = tr[fa].fa;
            if (pf != top) check(fa, pf)^check(u, fa)?rotate(u):rotate(fa);
            // 判断是链型的还是之字形的
            rotate(u); // 最后都要将u再左旋或右旋一次
        }
    }

    void insert(int val, int &u = root, int fa = 0) {
        if (!u) newnode(u, fa, val), splaying(u, 0);
        else if (val < tr[u].val) insert(val, tr[u].ch[0], u);
        else if (val > tr[u].val) insert(val, tr[u].ch[1], u);
        else tr[u].cnt++, splaying(u, 0);
    }

    void delnode(int u) { // 删除节点
        splaying(u, 0);
        // 如果这个数的个数大于1就直接在这个基础上-1就可以了
        if (tr[u].cnt > 1)
            tr[u].cnt --;
        else if(tr[u].ch[1]) {
// 如果只有一个,那么就查询他的后继,如果有的话,将让他后继的最左边的左儿子伸展到根节点的右儿子处
            int p = tr[u].ch[1];
            while(tr[p].ch[0])
                p = tr[p].ch[0];
            splaying(p, u);
            //将根节点的右儿子和左儿子建立父子关系
            connect(tr[u].ch[0], p, 0);
            root = p, tr[p].fa = 0;
            update(root);
        } else {  // 如果没有后继的话， 就直接让左儿子作为根节点
            root = tr[u].ch[0], tr[root].fa = 0;
        }
    }

    void del(int val, int u = root) {
        if (val == tr[u].val)
            delnode(u);
        else if (val < tr[u].val)
            del(val, tr[u].ch[0]);
        else
            del(val, tr[u].ch[1]);
    }
};
```

###  FHQ-Treap
##### 按权值分裂合并

```cpp
/*按照权值分裂合并*/
struct Treap {  
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
        if (tr[u].val <= a) {
            x = u;
            split(tr[u].r, a, tr[u].r, y);
        }  
        if (tr[u].val > a) {
            y = u;
            split(tr[u].l, a, x, tr[u].l);  
        }
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
        tr[++ idx].val = v;
        tr[idx].siz = 1;
        tr[idx].rnd = rand();  
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
            if (k <= tr[tr[now].l].siz)
                now = tr[now].l;  
            else if (k == tr[tr[now].l].siz + 1)
                return now;  
            else {
                k = k - tr[tr[now].l].siz - 1, now = tr[now].r;  
            }
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
```
#### 按大小分裂合并
```cpp
std::mt19937 rnd(233);
int root;

struct treap {
    struct Node {
        int son[2];
        int val, key;
        int siz, tag;
    };

    const int n;
    std::vector<Node> tr;
    int idx, x, y, z;

    treap(int n) : n(n), tr(n) {
        idx = x = y = z = 0;
    }

    inline int newnode(int val) {
        tr[ ++ idx].key = rnd();
        tr[idx].val = val;
        tr[idx].siz = 1;
        return idx;
    }

    inline void pull(int u) {
        tr[u].siz = tr[tr[u].son[0]].siz + tr[tr[u].son[1]].siz + 1;
    }

    inline void push(int u) { // 区间翻转打标记
        std::swap(tr[u].son[0], tr[u].son[1]);
        //标记下传
        tr[tr[u].son[0]].tag ^= 1;
        tr[tr[u].son[1]].tag ^= 1;
        tr[u].tag = 0;
    }

    void split(int u, int cnt, int& x, int& y) {
        if (!u) return x = y = 0, void();
        if (tr[u].tag) push(u);
        if (tr[tr[u].son[0]].siz < cnt) {
            x = u;
            split(tr[u].son[1],cnt-tr[tr[u].son[0]].siz-1,tr[u].son[1],y);
        } else {
            y = u;
            split(tr[u].son[0], cnt, x, tr[u].son[0]);
        }
        pull(u);
    }

    int merge(int u, int v) {
        if (!u || !v) return u | v;
        if (tr[u].key < tr[v].key) {
            if (tr[u].tag) push(u); // 如果有标记就要先传标记然后再合并
            tr[u].son[1] = merge(tr[u].son[1], v);
            pull(u);
            return u;
        } else {
            if (tr[v].tag) push(v);
            tr[v].son[0] = merge(u, tr[v].son[0]);
            pull(v);
            return v;
        }
    }

    void reverse(int l, int r) { // 区间翻转操作
        int x, y, z;
        // 将区间分成[1, l - 1], [l, r], [r + 1, n]三个区间
        split(root, l - 1, x, y);
        split(y, r - l + 1, y, z);
        tr[y].tag ^= 1;
        root = merge(merge(x, y), z);
    }

    void ldr(int u) { // 输出树的中序遍历
        if (!u) return ;
        if (tr[u].tag) push(u);
        ldr(tr[u].son[0]);
        std::cout << tr[u].val << " ";
        ldr(tr[u].son[1]);
    }
};

```

## 树链剖分
```cpp
// siz是子树大小，son是重儿子，dep是这个节点的深度，dfn是dfs序，top是链的端点
/*边权 ： 注释部分*/
std::vector<int> parent(n + 1), siz(n + 1), son(n + 1), dep(n + 1);
std::vector<int> dfn(n + 1), top(n + 1);
// std::vector<int> tmp(n + 1), va(n + 1);
int idx = 0;
Seg SGT(n + 1);

std::function<void(int,int,int)>dfs1=[&](int u,int fa,int depth) {  
//预处理出来轻重链
    parent[u] = fa;
    dep[u] = depth;
    siz[u] = 1;
    for (auto v : G[u]) {
    // for (auto& g : G[u]) {
        // int v = g.first, w = g.second;
        if (v == fa) continue;
        dfs1(v, u, depth + 1);
        siz[u] += siz[v];
        // tmp[v] = w;
        if (siz[v] > siz[son[u]])
            son[u] = v;
    }
};

std::function<void(int, int)> dfs2 = [&] (int u, int t) -> void {
    dfn[u] = ++ idx;
    top[u] = t;
    //va[idx] = tmp[u];
    if (!son[u]) return ;
    dfs2(son[u], t);

    for (auto v : G[u]) {
        if (v == parent[u] || v == son[u])
            continue;
        dfs2(v, v);
    }
};

auto ask = [&] (int u, int v) -> int {  
    //查询树上两个节点之间的最短距离中所有节点
    int ans = 0;
    while(top[u] != top[v]){
        if (dep[top[u]] < dep[top[v]])
            std::swap(u, v);
        ans += SGT.query(1, dfn[top[u]], dfn[u]);
        u = parent[top[u]];
    }
    if (dep[u] > dep[v]) std::swap(u, v);
    ans += SGT.query(1, dfn[u], dfn[v]);
    // ans += SGT.query(1, dfn[u] + 1, dfn[v]);
    return ans;
};

auto update = [&] (int u, int v, int x) -> void {   
    //修改两个节点最短路径上
    while(top[u] != top[v]) {
        if (dep[top[u]] < dep[top[v]])
            std::swap(u, v);
        SGT.change(1, dfn[top[u]], dfn[u], x);
        u = parent[top[u]];
    }
    if (dep[u] > dep[v]) std::swap(u, v);
    SGT.change(1, dfn[u], dfn[v], x);
    // SGT.change(1, dfn[u] + 1, dfn[v], x);
};

```

##  LCA
```cpp
struct Ancestor {
    const int n;
    std::vector<std::vector<int>> G;
    std::vector<int> siz, top, son, dep, parent;
    Ancestor(int n):n(n),G(n),siz(n),top(n),son(n),dep(n),parent(n){}

    void add(int a, int b) {
        G[a].push_back(b), G[b].push_back(a);
    }

    void build(int u) {
        function<void(int,int,int)>dfs1=[&](int u,int father,int depth){
            dep[u] = depth;
            parent[u] = father;
            siz[u] = 1;

            for (int &v: G[u]) {
                if (v == father)
                    continue;

                dfs1(v, u, depth + 1);
                siz[u] += siz[v];
                if (siz[son[u]] < siz[v])
                    son[u] = v;
            }
        };

        std::function<void(int, int)> dfs2 = [&](int u, int tp) -> void {
            top[u] = tp;
            for (int &v: G[u]) {
                if (v == parent[u])
                    continue;
                dfs2(v, v == son[u] ? tp : v);
            }
        };

        dfs1(u, 0, 1);
        dfs2(u, u);
    }

    int lca(int x, int y) {
        while (top[x] != top[y]) {
            if (dep[top[x]] < dep[top[y]])
                std::swap(x, y);
            x = parent[top[x]];
        }
        return dep[x] < dep[y] ? x : y;
    }
};
```

##  DSU On Tree

```cpp
    std::vector<int> siz(n + 1), son(n + 1);

    std::function<void(int, int)> dfs1 = [&] (int u, int fa) -> void {
        siz[u] = 1;
        for (auto v : G[u]) {
            if (v == fa) continue;
            dfs1(v, u);
            siz[u] += siz[v];
            if (siz[v] > siz[son[u]]) son[u] = v;
        }
    };

    dfs1(1, 0);

    std::vector<bool> vis(n + 1);
    std::vector<int> cnt(n + 1), ans(n + 1);
    int top = 0;
    std::function<void(int, int, int)> calc=[&](int u,int fa,int val) {
        if (val > 0) {
            if (!cnt[c[u]]) top++;
            cnt[c[u]] ++;
        } else {
            if (cnt[c[u]] <= 1) top--;
            cnt[c[u]] --;
        }
        for (auto v : G[u]) {
            if (v == fa || vis[v])
                continue;
            calc(v, u, val);
        }
    };

    std::function<void(int, int, bool)>dfs2=[&](int u,int fa,bool keep){
        for (auto v : G[u]) {
            if (v == fa || son[u] == v)
                continue;
            dfs2(v, u, false);
        }
        if (son[u])
            dfs2(son[u], u, true), vis[son[u]] = true;
        calc(u, fa, 1);
            vis[son[u]] = false, ans[u] = top;
        if (!keep) {
            calc(u, fa, -1);// top = 0;
        }
    };

    dfs2(1, 0, 0);
```

##  笛卡尔树
```cpp
int stk[N], l[N], r[N], n, a[N];
int ans[N], tot;

void dfs(int u) {
    ans[u] = ++ tot;
    if (l[u]) dfs(l[u]);
    if (r[u]) dfs(r[u]);
}

void build() {
    int top = 0;
    for (int i = 1; i <= n; i ++ )
        l[i] = r[i] = 0;
    for (int i = 1; i <= n; i ++ ) {
        int k = top;
        while(k > 0 && a[stk[k - 1]] > a[i]) -- k;
        if (k) r[stk[k - 1]] = i;
        if (k < top) l[i] = stk[k];
        stk[k ++ ] = i;
        top = k;
    }
    dfs(stk[0]); // 求出笛卡尔树的先序遍历
}
```

## 线性基
```cpp
struct LBase{
    vector<long long> _data;
    LBase(): _data(64, 0){}
    bool insert(long long x){
        for(int i = 63 - __builtin_clzll(x); i >= 0; --i){
            if((x >> i) & 1){
                if(_data[i]) x ^= _data[i];
                else{
                    _data[i] = x;
                    break;
                }
            }
        }
        return x > 0;
    }
    LBase& operator += (const LBase& arg){
        for(auto ptr = arg._data.rbegin(); ptr != arg._data.rend(); ++ptr){
            this->insert(*ptr);
        }
        return *this;
    }
    long long query(){
        long long ret = 0;
            for(auto ptr = _data.rbegin(); ptr != _data.rend(); ++ptr){
                if(*ptr){
                    if((ret ^ (*ptr)) > ret) ret ^= *ptr;
                }
            }
            return ret;
    }
    int count(){
        int ret = 0;
        for(auto& it: _data) if(it) ++ret;
        return ret;
    }
};
```

##  DSU

### 带权

```cpp
struct dsu {
  public:
    dsu() : _n(0) {}
    explicit dsu(int n) : _n(n), parent_or_size(n, -1) {}

    int merge(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        int x = leader(a), y = leader(b);
        if (x == y) return x;
        if (-parent_or_size[x] < -parent_or_size[y]) std::swap(x, y);
        parent_or_size[x] += parent_or_size[y];
        parent_or_size[y] = x;
        return x;
    }

    bool same(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        return leader(a) == leader(b);
    }

    int leader(int a) {
        assert(0 <= a && a < _n);
        if (parent_or_size[a] < 0) return a;
        return parent_or_size[a] = leader(parent_or_size[a]);
    }

    int size(int a) {
        assert(0 <= a && a < _n);
        return -parent_or_size[leader(a)];
    }

    std::vector<std::vector<int>> groups() {
        std::vector<int> leader_buf(_n), group_size(_n);
        for (int i = 0; i < _n; i++) {
            leader_buf[i] = leader(i);
            group_size[leader_buf[i]]++;
        }
        std::vector<std::vector<int>> result(_n);
        for (int i = 0; i < _n; i++) {
            result[i].reserve(group_size[i]);
        }
        for (int i = 0; i < _n; i++) {
            result[leader_buf[i]].push_back(i);
        }
        result.erase(
            std::remove_if(result.begin(), result.end(),
                           [&](const std::vector<int>& v) { return v.empty(); }),
            result.end());
        return result;
    }

  private:
    int _n;
    // root node: -1 * component size
    // otherwise: parent
    std::vector<int> parent_or_size;
};
```

## 分块
### 区间开方
```cpp
std::vector<int> a(n + 1), sum(n + 1);
std::vector<int> belong(n + 1), bln(n + 1), brn(n + 1);
std::vector<bool> sqrtfree(n + 1);
rep(i, 1, n + 1)
    std::cin >> a[i];
int lim = sqrt(n);

rep(i, 1, lim + 1) {
    bln[i] = (i - 1) * lim + 1;
    brn[i] = i * lim;
}

if (brn[lim] < n) {
    bln[++lim] = brn[lim - 1] + 1;
    brn[lim] = n;
}

rep(i, 1, lim + 1) {
    rep(j, bln[i], brn[i] + 1) {
        belong[j] = i, sum[i] += a[j];
    }
}

auto update = [&](int l, int r, int val) -> void {
    int L = belong[l], R = belong[r];

    if (L == R) {
        if (sqrtfree[belong[l]]) return ;

        for (int i = l, bel = belong[l]; i <= r; i ++) {
            sum[bel] -= a[i];
            a[i] = sqrt(a[i]);
            sum[bel] += a[i];
        }

        sqrtfree[L] = true;

        for (int i = bln[belong[l]], bel=belong[l];i<=brn[bel];i++)
            if (a[i] > 1) {
                sqrtfree[bel] = false;
                break;
            }

        return ;
    }

    for (int i = belong[l] + 1; i < belong[r]; i ++) {
        if (sqrtfree[i]) {
            continue;
        }

        sqrtfree[i] = true;

        for (int j = bln[i]; j <= brn[i]; j ++) {
            sum[i] -= a[j];
            a[j] = sqrt(a[j]);
            sum[i] += a[j];
        }

        for (int j = bln[i]; j <= brn[i]; j ++) {
            if (a[j] > 1) {
                sqrtfree[i] = false;
                break;
            }
        }
    }

    if (!sqrtfree[belong[l]]) {
        sqrtfree[belong[l]] = true;

        for (int i = l, bel = belong[l]; i <= brn[bel]; i ++) {
            sum[bel] -= a[i];
            a[i] = sqrt(a[i]);
            sum[bel] += a[i];
        }

        for (int i = bln[belong[l]],bel=belong[l];i<=brn[bel];i++) {
            if (a[i] > 1) {
                sqrtfree[bel] = false;
                break;
            }
        }
    }

    if (!sqrtfree[belong[r]]) {
        sqrtfree[belong[r]] = true;

        for (int i = bln[belong[r]], bel = belong[r]; i <= r; i ++) {
            sum[bel] -= a[i];
            a[i] = sqrt(a[i]);
            sum[bel] += a[i];
        }

        for (int i=bln[belong[r]],bel=belong[r];i<=brn[belong[r]];i++) {
            if (a[i] > 1) {
                sqrtfree[bel] = false;
                break;
            }
        }
    }
};

auto query = [&](int l, int r, int val) -> int {
    i64 ans = 0;

    if (belong[l] == belong[r]) {
        for (int i = l; i <= r; i ++)
            ans += a[i];

        return ans;
    }

    for (int i = belong[l] + 1; i < belong[r]; i ++ ) {
        ans += sum[i];
    }

    for (int i = l; i <= brn[belong[l]]; i ++ )
        ans += a[i];

    for (int i = bln[belong[r]]; i <= r; i ++ )
        ans += a[i];

    return ans;
};
```

##  ODT
###  Set
```cpp
struct ODT {
    struct Node {
        i64 l, r;
        mutable i64 v;

        Node (i64 l, i64 r = 0, i64 v = 0) : l(l), r(r), v(v) {}

        bool operator < (const Node& lhs) const {
            return l < lhs.l;
        }
    };
    std::set<Node> s;

    std::set<Node>::iterator split(int pos) { // 分裂区间
        std::set<Node>::iterator it = s.lower_bound(Node(pos));
        if (it -> l == pos && it != s.end())
            return it;
        -- it;
        if (it -> r < pos) {
            return s.end();
        }

        i64 l = it -> l, r = it -> r, v = it -> v;
        s.erase(it);
        s.insert(Node(l, pos - 1, v));
        return s.insert(Node(pos, r, v)).first;
    }

    void assign(int l, int r, i64 x) {    //区间推平
        std::set<Node>::iterator itr = split(r + 1), itl = split(l);
        s.erase(itl, itr);
        s.insert(Node(l, r, x));
    }

    void add(i64 l, i64 r, i64 x) {    //区间加法
        std::set<Node>::iterator itr = split(r + 1), itl = split(l);
        for (auto it = itl; it != itr; it ++ ) {
            it -> v += x;
        }
    }

    struct Rank {
        i64 val, cnt;
        bool operator < (const Rank& lhs) const {
            return val < lhs.val;
        }

        Rank(i64 val, i64 cnt) : val(val), cnt(cnt) {}
    };

    int rank(i64 l, i64 r, i64 x) {    //查询区间排名为x的数是多少
        std::set<Node>::iterator itr = split(r + 1), itl = split(l);

        std::vector<Rank> vec;
        for (auto it = itl; it != itr; ++ it ) {
            vec.push_back(Rank(it -> v, it -> r - it -> l + 1));
        }
        std::sort(all(vec));

        i64 ans = -1;
        for (auto p : vec) {
            if (p.cnt < x) {
                x -= p.cnt;
            } else {
                ans = p.val;
                break;
            }
        }
        return ans;
    }

    i64 query(i64 l, i64 r) { // 区间和
        auto itr = split(r + 1), itl = split(l);
        i64 ans = 0;
        for (auto it = itl; it != itr; ++ it) {
            ans += (it -> v) * (it -> r - it -> l + 1);
        }
        return ans;
    }
};
```

#  Math
##  线性筛
```cpp
const static int N = 100005;
int p[N], pr[N], pe[N];
int cnt;

void sieve(int n) {
    p[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!p[i]) p[i] = i, pr[++cnt] = i, pe[i] = i;
        for (int j = 1; j <= cnt && pr[j] * i <= N; j++) {
            p[i * pr[j]] = pr[j];
            if (p[i] == pr[j]) { // 如果 i 的最小素因子等于第 j 个素数
                pe[i * pr[j]] = pe[i] * pr[j];
                break;
            } else // 第 j 个素数小于 i 的最小素因子
                pe[i * pr[j]] = pr[j];
        }
    }
}
```

## Exgcd
```cpp
int exgcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }

    int xx, yy;
    int d = exgcd(b, a % b, xx, yy);
    x = yy;
    y = xx - (a / b) * yy;
    /*
        int d = exgcd(b, a % b, y, x);
        y -= (a / b) * x;
        return d;
    */
    return d;
}
/*
    int d = exgcd(a, b, x, y);
    assert(a * x + b * y == d);
    此时x y为 方程组的两个一般解
    调整x y大小则 x += b / d, y -= a / d;
    x0 = x + b * t, y0 = (d - a * x) / b;   这个是通解
*/

```

## 欧拉函数
```cpp
int Eular(int n) {
    i64 ans = n;

    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            ans = ans / i * (i - 1);
            while(n % i == 0) n /= i;
        }
    }
    if (n > 1) ans = ans / n * (n - 1);

    return ans;
}
```

## 整除分块
```cpp
for (i64 l = 1; l <= n; l++) {
    i64 v = n / l, r = n / v;
    // l ... r 之间的数都是 v
    l = r;
}
```

## CRT(中国剩余定理)
```cpp
i64 exgcd(i64 a, i64 b, i64& x, i64& y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }

    i64 d = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return d;
}

void merge(i64& a, i64& b, i64 c, i64 d) {
    // bt = c - a(mod d)
    if (a == -1 && b == -1) return ;
    i64 x, y;
    i64 g = exgcd(b, d, x, y);
    if ((c - a) % g) {
        a = -1, b = -1;
        return ;
    }
    // x = t;
    d /= g; // d'
    i64 t0 = ((c - a) / g) % d * x % d;
    if (t0 < 0) t0 += d;
    // t = t0(mod d)
    a = b * t0 + a;
    b = b * d; 	// [b, d]
}

void solve() {
    int n; scanf("%d", &n);

    i64 a = 0, b = 1;
    for (int i = 1; i <= n; i++) {
        int c, d;   // c是余数, d是模数
        scanf("%d%d", &c, &d);
        merge(a, b, c, d);
    }

    printf("%lld\n", a);
}
```

##  Binom
```cpp
namespace CNM {
    mint fac[N], fav[N];

    void init() {
        fac[0] = 1;
        for (int i = 1; i <= N - 1; i ++ ) fac[i] = fac[i - 1] * i;
        fav[N - 1] = fac[N - 1].inv();
        for (int i = N - 1; i; i -- ) fav[i - 1] = fav[i] * i;
    }

    mint binom(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac[n] * fav[m] * fav[n - m];
    }
}
```
##  Matrix
```cpp
// struct Matrix { // fibnacci的构造矩阵
//     int a[3][3];
//     Matrix() { memset(a, 0, sizeof a); }
//     void init() {
//         a[1][1] = a[2][2] = 1, a[1][2] = a[2][1] = 0;
//     }

//     bool check() {
//         return a[1][1] == 1 && a[2][2] == 1 && a[1][2] == 0 && a[2][1] == 0;
//     }

//     Matrix operator+(const Matrix b) {
//         Matrix res;
//         for (int i = 1; i <= 2; ++i)
//             for (int j = 1; j <= 2; ++j)
//                 res.a[i][j] = a[i][j] + b.a[i][j];
//         return res;
//     }

//     Matrix operator*(const Matrix b) {
//         Matrix res;
//         for (int k = 1; k <= 2; ++k)
//             for (int i = 1; i <= 2; ++i)
//                 for (int j = 1; j <= 2; ++j)
//                     res.a[i][j] = res.a[i][j] + a[i][k] * b.a[k][j];
//         return res;
//     }

//     Matrix operator^(int b) {
//         Matrix res, Base;
//         for (int i = 1; i <= 2; ++i)
//             res.a[i][i] = 1;
//         for (int i = 1; i <= 2; ++i)
//             for (int j = 1; j <= 2; ++j)
//                 Base.a[i][j] = a[i][j];

//         for (; b; b >>= 1, Base = Base * Base)
//             if (b & 1) res = res * Base;
//         return res;
//     }
// } base, res;
template<typename T>
struct Matrix{
    std::vector<T> data;
    int sz;
    // 构造全0矩阵，或者斜对角填上自定义数字
    Matrix(int sz, T v = 0): sz(sz), data(sz * sz, 0){
        int cur = 0;
        do{
            data[cur] = v;
            cur += sz + 1;
        }while(cur < sz * sz);
    }
    //从vector中构造矩阵
    Matrix(int sz, std::vector<T>& arg): sz(sz), data(sz * sz, 0){
        assert(arg.size() >= sz * sz);
        for(int i = 0; i < sz * sz; ++i) data[i] = arg[i];
    }
    //从vector中构造矩阵，右值
    Matrix(int sz, std::vector<T>&& arg): sz(sz), data(sz * sz, 0){
        assert(arg.size() >= sz * sz);
        for(int i = 0; i < sz * sz; ++i) data[i] = arg[i];
    }
    Matrix operator + (const Matrix& arg) const {
        assert(sz == arg.sz);
        Matrix ret(sz);
        for(int i = 0; i < sz * sz; ++i){
            ret.data[i] = data[i] + arg.data[i];
        }
        return ret;
    }
    Matrix operator * (const Matrix& arg) const {
        assert(sz == arg.sz);
        Matrix ret(sz);
        for(int i = 0; i < sz; ++i){
            for(int j = 0; j < sz; ++j){
                for(int k = 0; k < sz; ++k){
                    ret.data[i * sz + j] += data[i * sz + k] * arg.data[k * sz + j];
                }
            }
        }
        return ret;
    }
    Matrix operator - (const Matrix& arg) const {
        assert(sz == arg.sz);
        Matrix ret(sz);
        for(int i = 0; i < sz * sz; ++i) ret.data[i] = data[i] - arg.data[i];
        return ret;
    }
    friend std::ostream & operator << (std::ostream& ots, const Matrix& arg){
        for(int i = 0; i < arg.sz; ++i){
            for(int j = 0; j < arg.sz; ++j){
                if(j) ots << " ";
                ots << arg.data[i * arg.sz + j];
            }
            if(i + 1 != arg.sz) ots << "\n";
        }
        return ots;
    }
};
```

##  Gause
### 解线性方程组
```cpp
int n; std::cin >> n;
std::vector<std::vector<double>> arr(n, std::vector<double> (n + 1));

for (auto& x : arr) {
    for (auto& y : x)
        std::cin >> y;
}

auto gause = [&] () -> int {
    int r = 0, c = 0;
    for (r = 0, c = 0; c < n; c ++ ) {
        int t = r;
        for (int i = r; i < n; i ++ ) {
            if (fabs(arr[i][c]) > fabs(arr[t][c]))
                t = i;
        }
        if (fabs(arr[t][c]) < 1E-6) continue;
        std::swap(arr[t], arr[r]);
        for (int i = n; i >= c; i -- ) arr[r][i] /= arr[r][c];

        for (int i = r + 1; i < n; i ++ ) {
            if (fabs(arr[i][c]) > 1E-6) {
                for (int j = n; j >= c; j -- )
                    arr[i][j] -= arr[r][j] * arr[i][c];
            }
        }
        r ++;
    }
        // 返回2是有0 = 非零的一组对应无解
    // 返回1是有0 = 0的情况对应无穷多组解
    if (r < n) {
        for (int i = r; i < n; i ++ ) {
            if (fabs(arr[i][n]) > 1E-6) return 2;
        }
        return 1;
    }

    // 有唯一解
    for (int i = n - 1; i >= 0; i -- ) {
        for (int j = i + 1; j < n; j ++ )
            arr[i][n] -= arr[i][j] * arr[j][n];
    }
    return 0;
};
```
### 解异或线性方程组
```cpp
auto gause = [&] () -> int {
    int r = 0, c = 0;
    for (r = 0, c = 0; c < n; c ++ ) {
        int t = r;
        for (int i = r; i < n; i ++ ) {
            if (fabs(arr[i][c]) > fabs(arr[t][c]))
                t = i;
        }
        if (!arr[t][c]) continue;
        std::swap(arr[t], arr[r]);

        for (int i = r + 1; i < n; i ++ ) {
            if (arr[i][c]) {
                for (int j = n; j >= c; j -- )
                    arr[i][j] ^= arr[r][j];
            }
        }
        r ++;
    }
        // 返回2是有0 = 非零的一组对应无解
    // 返回1是有0 = 0的情况对应无穷多组解
    if (r < n) {
        for (int i = r; i < n; i ++ ) {
            if (arr[i][n]) return 2;
        }
        return 1;
    }

    // 有唯一解
    for (int i = n - 1; i >= 0; i -- ) {
        for (int j = i + 1; j < n; j ++ )
            arr[i][n] ^= arr[i][j] * arr[j][n];
    }
    return 0;
};

```

##  Lucas
```cpp
int qmi(int a, int k, int q) {
    int res = 1;
    while (k) {
        if (k & 1) res = res * (LL)a % q;
        a = a * (LL)a % q;
        k >>= 1;
    }
    return res;
}

int C(int a, int b, int q) {
    if (b > a) return 0;

    int res = 1;
    for (int i = 1, j = a; i <= b; i ++ , j -- ) {
        res = 1ll * res * j % q;
        res = 1ll * res * qmi(i, q - 2, q) % q;
    }
    return res;
}

int lucas(LL a, LL b, int q) {
    if (a < q && b < q) return C(a, b, q);
    return (LL)C(a % q, b % q, q) * lucas(a / q, b / q, q) % q;
}
```

##  FFT
```cpp
constexpr int N = 1300010;
constexpr double PI = acos(-1);

struct Complex {
    Complex(double a = 0, double b = 0) : x(a), y(b) {}
    double x, y;
    Complex operator+ (const Complex& a) const {
        return Complex(x + a.x, y + a.y);
    }
    Complex operator- (const Complex& a) const {
        return Complex(x - a.x, y - a.y);
    }
    Complex operator* (const Complex& a) const {
        return Complex(x * a.x - y * a.y, x * a.y + y * a.x);
    }
    Complex operator / (Complex const &a) const {
        double t = a.x * a.x + a.y * a.y;
        return Complex((x * a.x + y * a.y) / t, (y * a.x - x * a.y) / t);
        }
}f[N << 1];

int n, m;
namespace FFT {
    int tr[N << 1];

    void init() {
        int L = 0;
        for (m += n, n = 1; n <= m; n <<= 1) L++;
        for (int i = 0; i < n; i++) {
            tr[i] = (tr[i >> 1] >> 1) | ((i & 1) << (L - 1));
        }
    }

    void fft(Complex* f, bool flag) {
        for (int i = 0; i < n; i++) {
            if (i < tr[i])
                std::swap(f[i], f[tr[i]]);
        }

        for (int p = 2; p <= n; p <<= 1) {
            int len = p >> 1;
            Complex tG(cos(2 * PI / p), sin(2 * PI / p));
            if (!flag) tG.y *= -1;
            for (int k = 0; k < n; k += p) {
                Complex buf(1, 0);
                for (int l = k; l < k + len; l++) {
                    Complex tt = buf * f[len + l];
                    f[len + l] = f[l] - tt;
                    f[l] = f[l] + tt;
                    buf = buf * tG;
                }
            }
        }
    }
}
using namespace FFT;

signed main() {
    std::cin.tie(nullptr)->sync_with_stdio(false);
    std::cin >> n >> m;
    for (int i = 0; i <= n; i++) std::cin >> f[i].x;
    for (int i = 0; i <= m; i++) std::cin >> f[i].y;

    init();

    fft(f, 1);
    for (int i = 0; i < n; i++) f[i] = f[i] * f[i];
    fft(f, 0);
    for (int i = 0; i <= m; i++)
        std::cout << int(f[i].y / n / 2 + 0.49) << " \n"[i == m];

    return 0 ^ 0;
}
```
# 计算几何
```cpp
namespace std {
bool operator<(const complex<double> &a, const complex<double> &b) { return !(fabs(a.real() - b.real()) < 1e-8) ? a.real() < b.real() : a.imag() < b.imag(); }
} // namespace std
 
namespace Geometry {
using Real = double;
using Point = complex<Real>;
const Real EPS = 1e-8, PI = acos(-1);
 
inline bool eq(Real a, Real b) { return fabs(b - a) < EPS; }
 
Point operator*(const Point &p, const Real &d) { return Point(real(p) * d, imag(p) * d); }
 
istream &operator>>(istream &is, Point &p) {
    Real a, b;
    is >> a >> b;
    p = Point(a, b);
    return is;
}
 
ostream &operator<<(ostream &os, Point &p) { return os << p.real() << " " << p.imag(); }
 
// 点 p を反時計回りに theta 回転
Point rotate(Real theta, const Point &p) { return Point(cos(theta) * p.real() - sin(theta) * p.imag(), sin(theta) * p.real() + cos(theta) * p.imag()); }

Point rotate90(const Point& p) { return {-p.imag(), p.real()}; } 

Real radian_to_degree(Real r) { return (r * 180.0 / PI); }
 
Real degree_to_radian(Real d) { return (d * PI / 180.0); }
 
// a-b-c の角度のうち小さい方を返す
Real get_angle(const Point &a, const Point &b, const Point &c) {
    const Point v(b - a), w(c - b);
    Real alpha = atan2(v.imag(), v.real()), beta = atan2(w.imag(), w.real());
    if(alpha > beta) swap(alpha, beta);
    Real theta = (beta - alpha);
    return min(theta, 2 * acos(-1) - theta);
}
 
struct Line {
    Point a, b;
 
    Line() = default;
 
    Line(Point a, Point b) : a(a), b(b) {}
 
    Line(Real A, Real B, Real C) // Ax + By = C
    {
        if(eq(A, 0))
            a = Point(0, C / B), b = Point(1, C / B);
        else if(eq(B, 0))
            a = Point(C / A, 0), b = Point(C / A, 1);
        else
            a = Point(0, C / B), b = Point(C / A, 0);
    }
 
    friend ostream &operator<<(ostream &os, Line &p) { return os << p.a << " to " << p.b; }
 
    friend istream &operator>>(istream &is, Line &a) { return is >> a.a >> a.b; }
};
 
struct Segment : Line {
    Segment() = default;
 
    Segment(Point a, Point b) : Line(a, b) {}
};
 
struct Circle {
    Point p;
    Real r;
 
    Circle() = default;
 
    Circle(Point p, Real r) : p(p), r(r) {}
};
 
using Points = vector<Point>;
using Polygon = vector<Point>;
using Segments = vector<Segment>;
using Lines = vector<Line>;
using Circles = vector<Circle>;
 
Real cross(const Point &a, const Point &b) { return real(a) * imag(b) - imag(a) * real(b); }
 
Real dot(const Point &a, const Point &b) { return real(a) * real(b) + imag(a) * imag(b); }
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_1_C
// 点の回転方向
int ccw(const Point &a, Point b, Point c) {
    b = b - a, c = c - a;
    if(cross(b, c) > EPS) return +1;  // "COUNTER_CLOCKWISE"
    if(cross(b, c) < -EPS) return -1; // "CLOCKWISE"
    if(dot(b, c) < 0) return +2;      // "ONLINE_BACK"
    if(norm(b) < norm(c)) return -2;  // "ONLINE_FRONT"
    return 0;                         // "ON_SEGMENT"
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_2_A
// 平行判定
bool parallel(const Line &a, const Line &b) { return eq(cross(a.b - a.a, b.b - b.a), 0.0); }
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_2_A
// 垂直判定
bool orthogonal(const Line &a, const Line &b) { return eq(dot(a.a - a.b, b.a - b.b), 0.0); }
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_1_A
// 射影
// 直線 l に p から垂線を引いた交点を求める
Point projection(const Line &l, const Point &p) {
    Real t = dot(p - l.a, l.a - l.b) / norm(l.a - l.b);
    return l.a + (l.a - l.b) * t;
}
 
Point projection(const Segment &l, const Point &p) {
    Real t = dot(p - l.a, l.a - l.b) / norm(l.a - l.b);
    return l.a + (l.a - l.b) * t;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_1_B
// 反射
// 直線 l を対称軸として点 p  と線対称にある点を求める
Point reflection(const Line &l, const Point &p) { return p + (projection(l, p) - p) * 2.0; }
 
bool intersect(const Line &l, const Point &p) { return abs(ccw(l.a, l.b, p)) != 1; }
 
bool intersect(const Line &l, const Line &m) { return abs(cross(l.b - l.a, m.b - m.a)) > EPS || abs(cross(l.b - l.a, m.b - l.a)) < EPS; }
 
bool intersect(const Segment &s, const Point &p) { return ccw(s.a, s.b, p) == 0; }
 
bool intersect(const Line &l, const Segment &s) { return cross(l.b - l.a, s.a - l.a) * cross(l.b - l.a, s.b - l.a) < EPS; }
 
Real distance(const Line &l, const Point &p);
 
bool intersect(const Circle &c, const Line &l) { return distance(l, c.p) <= c.r + EPS; }
 
bool intersect(const Circle &c, const Point &p) { return abs(abs(p - c.p) - c.r) < EPS; }
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_2_B
bool intersect(const Segment &s, const Segment &t) { return ccw(s.a, s.b, t.a) * ccw(s.a, s.b, t.b) <= 0 && ccw(t.a, t.b, s.a) * ccw(t.a, t.b, s.b) <= 0; }
 
int intersect(const Circle &c, const Segment &l) {
    if(norm(projection(l, c.p) - c.p) - c.r * c.r > EPS) return 0;
    auto d1 = abs(c.p - l.a), d2 = abs(c.p - l.b);
    if(d1 < c.r + EPS && d2 < c.r + EPS) return 0;
    if(d1 < c.r - EPS && d2 > c.r + EPS || d1 > c.r + EPS && d2 < c.r - EPS) return 1;
    const Point h = projection(l, c.p);
    if(dot(l.a - h, l.b - h) < 0) return 2;
    return 0;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_7_A&lang=jp
int intersect(Circle c1, Circle c2) {   //return the number of tangent lines
    if(c1.r < c2.r) swap(c1, c2);
    Real d = abs(c1.p - c2.p);
    if(c1.r + c2.r < d) return 4;
    if(eq(c1.r + c2.r, d)) return 3;
    if(c1.r - c2.r < d) return 2;
    if(eq(c1.r - c2.r, d)) return 1;
    return 0;
}
 
Real distance(const Point &a, const Point &b) { return abs(a - b); }
 
Real distance(const Line &l, const Point &p) { return abs(p - projection(l, p)); }
 
Real distance(const Line &l, const Line &m) { return intersect(l, m) ? 0 : distance(l, m.a); }
 
Real distance(const Segment &s, const Point &p) {
    Point r = projection(s, p);
    if(intersect(s, r)) return abs(r - p);
    return min(abs(s.a - p), abs(s.b - p));
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_2_D
Real distance(const Segment &a, const Segment &b) {
    if(intersect(a, b)) return 0;
    return min({distance(a, b.a), distance(a, b.b), distance(b, a.a), distance(b, a.b)});
}
 
Real distance(const Line &l, const Segment &s) {
    if(intersect(l, s)) return 0;
    return min(distance(l, s.a), distance(l, s.b));
}
 
Point crosspoint(const Line &l, const Line &m) {
    Real A = cross(l.b - l.a, m.b - m.a);
    Real B = cross(l.b - l.a, l.b - m.a);
    if(eq(abs(A), 0.0) && eq(abs(B), 0.0)) return m.a;
    return m.a + (m.b - m.a) * B / A;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_2_C
Point crosspoint(const Segment &l, const Segment &m) { return crosspoint(Line(l), Line(m)); }
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_7_D
pair<Point, Point> crosspoint(const Circle &c, const Line l) {
    Point pr = projection(l, c.p);
    Point e = (l.b - l.a) / abs(l.b - l.a);
    if(eq(distance(l, c.p), c.r)) return {pr, pr};
    Real base = sqrt(c.r * c.r - norm(pr - c.p));
    return {pr - e * base, pr + e * base};
}
 
pair<Point, Point> crosspoint(const Circle &c, const Segment &l) {
    Line aa = Line(l.a, l.b);
    if(intersect(c, l) == 2) return crosspoint(c, aa);
    auto ret = crosspoint(c, aa);
    if(dot(l.a - ret.first, l.b - ret.first) < 0)
        ret.second = ret.first;
    else
        ret.first = ret.second;
    return ret;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_7_E
pair<Point, Point> crosspoint(const Circle &c1, const Circle &c2) {
    Real d = abs(c1.p - c2.p);
    Real a = acos((c1.r * c1.r + d * d - c2.r * c2.r) / (2 * c1.r * d));
    Real t = atan2(c2.p.imag() - c1.p.imag(), c2.p.real() - c1.p.real());
    Point p1 = c1.p + Point(cos(t + a) * c1.r, sin(t + a) * c1.r);
    Point p2 = c1.p + Point(cos(t - a) * c1.r, sin(t - a) * c1.r);
    return {p1, p2};
}

Circle inCircle(Polygon P) {
    assert((int)P.size() == 3);
    Real l0 = distance(P[1], P[2]), l1 = distance(P[0], P[2]), l2 = distance(P[0], P[1]);
    Circle c;
    c.p = crosspoint(Line(P[0], (P[1] * l1 + P[2] * l2) / (l1 + l2)), Line(P[1], (P[0] * l0 + P[2] * l2) / (l0 + l2)));
    c.r = distance(Line(P[0], P[1]), c.p);
    return c;
}

Circle circumcircle(Polygon P) {
    assert((int)P.size() == 3);
    Circle c;
    c.p = crosspoint(Line((P[0] + P[1]) / 2.0, (P[0] + P[1]) / 2.0 + rotate90(P[0] - P[1])),
        Line((P[0] + P[2]) / 2.0, (P[0] + P[2]) / 2.0 + rotate90(P[0] - P[2])));
    c.r = distance(c.p, P[0]);
    return c;
}

#define BOTTOM 0
#define LEFT 1
#define RIGHT 2
#define TOP 3

class EndPoint {
public :
    Point p;
    int seg, st; // 线段的ID，端点的种类
    EndPoint() {}
    EndPoint(Point p, int seg, int st) :p(p), seg(seg), st(st) {}

    // 按y坐标升序排序
    bool operator < (const EndPoint &ep) const {
        if (p.imag() == ep.p.imag()) return st < ep.st;
        return p.imag() < ep.p.imag();
    }

};
EndPoint EP[2 * 1000010];

// 线段相交问题，曼哈顿几何
int manhattanIntersection(vector<Segment> S) {
    int n = S.size();
    sort(EP, EP + (2 * n));    //按照端点的y坐标升序排序

    set<int> BT;            // 二叉搜索树
    BT.insert(2000000000); // 设置标记
    int cnt = 0;

    for (int i = 0; i < 2 * n; i++) {
        if (EP[i].st == TOP)
            BT.erase(EP[i].p.real()); //删除上端点
        else if (EP[i].st == BOTTOM)
            BT.insert(EP[i].p.real());
        else if (EP[i].st == LEFT) {
            set<int>::iterator b = BT.lower_bound(S[EP[i].seg].a.real());
            set<int>::iterator e = BT.upper_bound(S[EP[i].seg].b.real());

            // 加上b到e距离
            cnt += distance(b, e);
        }

    }
    return cnt;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_7_F
// 点 p を通る円 c の接線
pair<Point, Point> tangent(const Circle &c1, const Point &p2) { return crosspoint(c1, Circle(p2, sqrt(norm(c1.p - p2) - c1.r * c1.r))); }
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_7_G
// 円 c1, c2 の共通接線
Lines tangent(Circle c1, Circle c2) {
    Lines ret;
    if(c1.r < c2.r) swap(c1, c2);
    Real g = norm(c1.p - c2.p);
    if(eq(g, 0)) return ret;
    Point u = (c2.p - c1.p) / sqrt(g);
    Point v = rotate(PI * 0.5, u);
    for(int s : {-1, 1}) {
        Real h = (c1.r + s * c2.r) / sqrt(g);
        if(eq(1 - h * h, 0)) {
            ret.emplace_back(c1.p + u * c1.r, c1.p + (u + v) * c1.r);
        } else if(1 - h * h > 0) {
            Point uu = u * h, vv = v * sqrt(1 - h * h);
            ret.emplace_back(c1.p + (uu + vv) * c1.r, c2.p - (uu + vv) * c2.r * s);
            ret.emplace_back(c1.p + (uu - vv) * c1.r, c2.p - (uu - vv) * c2.r * s);
        }
    }
    return ret;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_3_B
// 凸性判定
bool is_convex(const Polygon &p) {
    int n = (int)p.size();
    for(int i = 0; i < n; i++) {
        if(ccw(p[(i + n - 1) % n], p[i], p[(i + 1) % n]) == -1) return false;
    }
    return true;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_4_A
// 凸包
Polygon convex_hull(Polygon &p, bool strict = true) {
    int n = (int)p.size(), k = 0;
    if(n <= 2) return p;
    sort(p.begin(), p.end());
    vector<Point> ch(2 * n);
    Real eps = strict ? EPS : -EPS;
    for(int i = 0; i < n; ch[k++] = p[i++]) {
        while(k >= 2 && cross(ch[k - 1] - ch[k - 2], p[i] - ch[k - 1]) < eps) --k;
    }
    for(int i = n - 2, t = k + 1; i >= 0; ch[k++] = p[i--]) {
        while(k >= t && cross(ch[k - 1] - ch[k - 2], p[i] - ch[k - 1]) < eps) --k;
    }
    ch.resize(k - 1);
    return ch;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_3_C
// 多角形と点の包含判定
enum { OUTTT, ONNN, INNN };
 
int contains(const Polygon &Q, const Point &p) {
    bool in = false;
    for(int i = 0; i < Q.size(); i++) {
        Point a = Q[i] - p, b = Q[(i + 1) % Q.size()] - p;
        if(a.imag() > b.imag()) swap(a, b);
        if(a.imag() <= 0 && 0 < b.imag() && cross(a, b) < 0) in = !in;
        if(cross(a, b) == 0 && dot(a, b) <= 0) return ONNN;
    }
    return in ? INNN : OUTTT;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1033
// 線分の重複除去
void merge_segments(vector<Segment> &segs) {
 
    auto merge_if_able = [](Segment &s1, const Segment &s2) {
        if(abs(cross(s1.b - s1.a, s2.b - s2.a)) > EPS) return false;
        if(ccw(s1.a, s2.a, s1.b) == 1 || ccw(s1.a, s2.a, s1.b) == -1) return false;
        if(ccw(s1.a, s1.b, s2.a) == -2 || ccw(s2.a, s2.b, s1.a) == -2) return false;
        s1 = Segment(min(s1.a, s2.a), max(s1.b, s2.b));
        return true;
    };
 
    for(int i = 0; i < segs.size(); i++) {
        if(segs[i].b < segs[i].a) swap(segs[i].a, segs[i].b);
    }
    for(int i = 0; i < segs.size(); i++) {
        for(int j = i + 1; j < segs.size(); j++) {
            if(merge_if_able(segs[i], segs[j])) { segs[j--] = segs.back(), segs.pop_back(); }
        }
    }
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1033
// 線分アレンジメント
// 任意の2線分の交点を頂点としたグラフを構築する
vector<vector<int>> segment_arrangement(vector<Segment> &segs, vector<Point> &ps) {
    vector<vector<int>> g;
    int N = (int)segs.size();
    for(int i = 0; i < N; i++) {
        ps.emplace_back(segs[i].a);
        ps.emplace_back(segs[i].b);
        for(int j = i + 1; j < N; j++) {
            const Point p1 = segs[i].b - segs[i].a;
            const Point p2 = segs[j].b - segs[j].a;
            if(cross(p1, p2) == 0) continue;
            if(intersect(segs[i], segs[j])) { ps.emplace_back(crosspoint(segs[i], segs[j])); }
        }
    }
    sort(begin(ps), end(ps));
    ps.erase(unique(begin(ps), end(ps)), end(ps));
 
    int M = (int)ps.size();
    g.resize(M);
    for(int i = 0; i < N; i++) {
        vector<int> vec;
        for(int j = 0; j < M; j++) {
            if(intersect(segs[i], ps[j])) { vec.emplace_back(j); }
        }
        for(int j = 1; j < vec.size(); j++) {
            g[vec[j - 1]].push_back(vec[j]);
            g[vec[j]].push_back(vec[j - 1]);
        }
    }
    return (g);
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_4_C
// 凸多角形の切断
// 直線 l.a-l.b で切断しその左側にできる凸多角形を返す
Polygon convex_cut(const Polygon &U, Line l) {
    Polygon ret;
    for(int i = 0; i < U.size(); i++) {
        Point now = U[i], nxt = U[(i + 1) % U.size()];
        if(ccw(l.a, l.b, now) != -1) ret.push_back(now);
        if(ccw(l.a, l.b, now) * ccw(l.a, l.b, nxt) < 0) { ret.push_back(crosspoint(Line(now, nxt), l)); }
    }
    return (ret);
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_3_A
// 多角形の面積
Real area(const Polygon &p) {
    Real A = 0;
    for(int i = 0; i < p.size(); ++i) { A += cross(p[i], p[(i + 1) % p.size()]); }
    return A * 0.5;
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_7_H
// 円と多角形の共通部分の面積
Real area(const Polygon &p, const Circle &c) {
    if(p.size() < 3) return 0.0;
    function<Real(Circle, Point, Point)> cross_area = [&](const Circle &c, const Point &a, const Point &b) {
        Point va = c.p - a, vb = c.p - b;
        Real f = cross(va, vb), ret = 0.0;
        if(eq(f, 0.0)) return ret;
        if(max(abs(va), abs(vb)) < c.r + EPS) return f;
        if(distance(Segment(a, b), c.p) > c.r - EPS) return c.r * c.r * arg(vb * conj(va));
        auto u = crosspoint(c, Segment(a, b));
        vector<Point> tot{a, u.first, u.second, b};
        for(int i = 0; i + 1 < tot.size(); i++) { ret += cross_area(c, tot[i], tot[i + 1]); }
        return ret;
    };
    Real A = 0;
    for(int i = 0; i < p.size(); i++) { A += cross_area(c, p[i], p[(i + 1) % p.size()]); }
    return A;
}
 
Real area(const Circle& a,const Circle& b){
   Real d = abs(a.p - b.p);
   if(d >= a.r + b.r - EPS) return .0;
   if(d <= abs(a.r - b.r) + EPS){
      Real r = min(a.r,b.r);
      return PI * r * r;
   }
   Real ath = acos(( a.r * a.r + d * d - b.r * b.r) / d / a.r / 2.);
   Real res = a.r * a.r * (ath - sin(ath * 2) / 2.);
   Real bth = acos((b.r * b.r + d * d - a.r * a.r) / d / b.r / 2.);
   res += b.r * b.r * (bth - sin(bth * 2) / 2.);
   return res;
}

// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_4_B
// 凸多角形の直径(最遠頂点対間距離)
Real convex_diameter(const Polygon &p) {
    int N = (int)p.size();
    int is = 0, js = 0;
    for(int i = 1; i < N; i++) {
        if(p[i].imag() > p[is].imag()) is = i;
        if(p[i].imag() < p[js].imag()) js = i;
    }
    Real maxdis = norm(p[is] - p[js]);
 
    int maxi, maxj, i, j;
    i = maxi = is;
    j = maxj = js;
    do {
        if(cross(p[(i + 1) % N] - p[i], p[(j + 1) % N] - p[j]) >= 0) {
            j = (j + 1) % N;
        } else {
            i = (i + 1) % N;
        }
        if(norm(p[i] - p[j]) > maxdis) {
            maxdis = norm(p[i] - p[j]);
            maxi = i;
            maxj = j;
        }
    } while(i != is || j != js);
    return sqrt(maxdis);
}
 
// http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=CGL_5_A
// 最近点対
Real closest_pair(Points ps) {
    if(ps.size() <= 1) throw(0);
    sort(begin(ps), end(ps));
 
    auto compare_y = [&](const Point &a, const Point &b) { return imag(a) < imag(b); };
    vector<Point> beet(ps.size());
    const Real INF = 1e18;
 
    function<Real(int, int)> rec = [&](int left, int right) {
        if(right - left <= 1) return INF;
        int mid = (left + right) >> 1;
        auto x = real(ps[mid]);
        auto ret = min(rec(left, mid), rec(mid, right));
        inplace_merge(begin(ps) + left, begin(ps) + mid, begin(ps) + right, compare_y);
        int ptr = 0;
        for(int i = left; i < right; i++) {
            if(abs(real(ps[i]) - x) >= ret) continue;
            for(int j = 0; j < ptr; j++) {
                auto luz = ps[i] - beet[ptr - j - 1];
                if(imag(luz) >= ret) break;
                ret = min(ret, abs(luz));
            }
            beet[ptr++] = ps[i];
        }
        return ret;
    };
    return rec(0, (int)ps.size());
}
} // namespace Geometry
using namespace Geometry;

```
