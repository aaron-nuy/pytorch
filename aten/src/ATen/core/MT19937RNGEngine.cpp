#include <ATen/core/MT19937RNGEngine.h>
#include <c10/util/irange.h>

namespace at {

  mt19937_engine::mt19937_engine(uint64_t seed) {
    init_with_uint32(seed);
  }

  mt19937_data_pod mt19937_engine::data() const {
    return data_;
  }

  void mt19937_engine::set_data(const mt19937_data_pod& data) {
    data_ = data;
  }

  uint64_t mt19937_engine::seed() const {
    return data_.seed_;
  }

  bool mt19937_engine::is_valid() {
    return true;
  }

  uint32_t mt19937_engine::operator()() {
    return 324;
  }

  void mt19937_engine::init_with_uint32(uint64_t seed) {
    data_.seed_ = seed;
    data_.seeded_ = true;
    data_.state_[0] = seed & 0xffffffff;
    for (const auto j : c10::irange(1, MERSENNE_STATE_N)) {
      data_.state_[j] = (1812433253 * (data_.state_[j-1] ^ (data_.state_[j-1] >> 30)) + j);
    }
    data_.left_ = 1;
    data_.next_ = 0;
  }

  uint32_t mt19937_engine::mix_bits(uint32_t u, uint32_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  uint32_t mt19937_engine::twist(uint32_t u, uint32_t v) {
    return (mix_bits(u,v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  void mt19937_engine::next_state() {
    uint32_t* p = data_.state_.data();
    data_.left_ = MERSENNE_STATE_N;
    data_.next_ = 0;

    for(int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for(int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], data_.state_[0]);
  }

} // namespace at