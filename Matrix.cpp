template<typename T>
struct Matrix {
    const int n;
    std::vector<std::vector<T>> M;

    Matrix(int n) : n(n), M(n, std::vector<T> (n)) {}

    Matrix operator + (const Matrix& A) const {
        Matrix res;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                res.M[i][j] = M[i][j] + A.M[i][j];
        return res;
    }

    Matrix operator - (const Matrix& A) const {
        Matrix res;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                res.M[i][j] = M[i][j] - A.M[i][j];
        return res;
    }

    Matrix operator * (const Matrix& A) const {
        Matrix res;
        for (int i = 0; i < n; i ++ )
            for (int k = 0; k < n; k ++ )
                for (int j = 0; j < n; j ++ )
                    res.M[i][j] += M[i][k] * A.M[k][j];
        return res;
    }

    Matrix operator ^ (i64 x) const {   //矩阵次方
        Matrix res, base;
        for (int i = 0; i < n; i ++ )
            res.M[i][i] = 1;
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                base.M[i][j] = M[i][j];
        while(x) {
            if (x & 1) res = res * base;
            base = base * base;
            x >>= 1;
        }
        return res;
    }
};
