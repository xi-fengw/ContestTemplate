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
