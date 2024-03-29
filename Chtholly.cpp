struct Chtholly {
  Chtholly(int n) : total(n) {
    intervals[n] = 1;
    intervals[n + 1] = n + 1;
  }
 
    void dump() const {
        std::cerr << "[";
        for (auto&& iterator : intervals) {
            std::cerr << "[" << iterator.second << "," << iterator.first << "] ";
            // printf("[%d,%d] ", iterator.second, iterator.first);
        }
        std::cerr << "]";
        // puts("");
    }
 
  void split(int x) {
    auto iterator = intervals.lower_bound(x);
    if (iterator->second < x) {
      int r = iterator->first;
      int l = iterator->second;
      intervals.erase(iterator);
      intervals[r] = x;
      intervals[x - 1] = l;
    }
  }
 
  void clear(int l, int r) {
    split(r + 1);
    split(l);
    while (true) {
      auto iterator = intervals.lower_bound(l);
      if (iterator->first > r) {
        break;
      }
      total -= iterator->first - iterator->second + 1;
      intervals.erase(iterator);
    }
  }
 
  void add(int l, int r) {
    total += r - l + 1;
    intervals[r] = l;
  }
 
  int total;
  std::map<int, int> intervals;
};