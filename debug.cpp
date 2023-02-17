namespace {
 
template <typename Tuple, size_t... Index>
std::ostream &serialize_tuple(std::ostream &out, const Tuple &t,
                                std::index_sequence<Index...>) {
    out << "(";
    (..., (out << (Index == 0 ? "" : ", ") << std::get<Index>(t)));
    return out << ")";
}
 
} // namespace
 
template <typename A, typename B>
std::ostream &operator<<(std::ostream &out, const std::pair<A, B> &v) {
    return out << "(" << v.first << ", " << v.second << ")";
}
 
template <typename... T>
std::ostream &operator<<(std::ostream &out, const std::tuple<T...> &t) {
    return serialize_tuple(out, t, std::make_index_sequence<sizeof...(T)>());
}
 
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    out << "[";
    bool first = true;
    for (auto &&e : v) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << e;
    }
    return out << "]";
}
 
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &v) {
    return out << std::vector<T>(v.begin(), v.end());
}
 
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::list<T> &v) {
    out << "[";
    bool first = true;
    for (auto &&e : v) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << e;
    }
    return out << "]";
}
 
template <typename K>
std::ostream &operator<<(std::ostream &out, const std::set<K> &s) {
    out << "{";
    bool first = true;
    for (auto &&k : s) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << k;
    }
    return out << "}";
}
 
template <typename K, typename V>
std::ostream &operator<<(std::ostream &out, const std::map<K, V> &m) {
    out << "{";
    bool first = true;
    for (auto &&[k, v] : m) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << k << ": " << v;
    }
    return out << "}";
}
 
#define debug(x...)                                                            \
  do {                                                                         \
    std::cerr << "\033[32;1m" << #x << " -> ";                                      \
    rd_debug(x);                                                               \
  } while (0)
void rd_debug() { std::cerr << "\033[39;0m" << "\n"; }
template <class T, class... Ts> void rd_debug(const T &arg, const Ts &...args) {
    std::cerr << arg << " ";
    rd_debug(args...);
}

#define KV(x) #x << "=" << (x) << ";"
#define KV1(x) #x << "=" << (x) + 1 << ";"
