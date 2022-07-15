using i64 = long long;

constexpr int mod = 998244353;

// assume -P <= x < 2P
int norm(int x) {
    if (x < 0) x += mod;
    if (x >= mod) x -= mod;
    return x;
}

template <class T> 
T power(T a, int b) {
    T res = 1;
    while (b) {
        if (b & 1) res *= a;
        b >>= 1, a *= a;
    }
    return res;
}

struct Z {
    int x;  
    Z(int x = 0) : x(norm(x)) {}
    int val() const {
        return x;
    }
    Z operator-() const {
        return Z(norm(mod - x));
    }
    Z inv() const {
        assert(x != 0);
        return power(*this, mod - 2);
    }
    Z &operator*=(const Z &rhs) {
        x = i64(x) * rhs.x % mod;
        return *this;
    }
    Z &operator+=(const Z &rhs) {
        x = norm(x + rhs.x);
        return *this;
    }
    Z &operator-=(const Z &rhs) {
        x = norm(x - rhs.x);
        return *this;
    }
    Z &operator/=(const Z &rhs) {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator+(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res += rhs;
        return res;
    }
    friend Z operator-(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res /= rhs;
        return res;
    }
};

//组合数
namespace CNM {
	Z fac[N], fav[N];
	 
	void init() {
		fac[0] = 1;
		for (int i = 1; i <= N - 1; i ++ ) fac[i] = fac[i - 1] * i;
		fav[N - 1] = fac[N - 1].inv();
		for (int i = N - 1; i; i -- ) fav[i - 1] = fav[i] * i;
	}
	 
	Z binom(int n, int m) {
		if (n < m || m < 0) return 0;
		return fac[n] * fav[m] * fav[n - m];
	}
}
