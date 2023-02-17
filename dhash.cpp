using hashv = std::pair<int, int>;
const long long mod1 = 1000000007;
const long long mod2 = 1000000009;
 
hashv operator + (hashv a,hashv b) {
    int c1 = a.first + b.first, c2 = a.second + b.second;
    if (c1 >= mod1) c1 -= mod1;
    if (c2 >= mod2) c2 -= mod2;
	return std::make_pair(c1, c2);
}
 
hashv operator - (hashv a,hashv b) {
    int c1 = a.first - b.first, c2 = a.second - b.second;
    if (c1 < 0) c1 += mod1;
    if (c2 < 0) c2 += mod2;
	return std::make_pair(c1, c2);
}
 
hashv operator * (hashv a,hashv b) {
    return std::make_pair(1LL * a.first * b.first % mod1, 1LL * a.second * b.second % mod2);
}
