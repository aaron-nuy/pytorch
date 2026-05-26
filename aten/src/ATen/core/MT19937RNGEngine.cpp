#include <ATen/core/MT19937RNGEngine.h>
#include <c10/util/irange.h>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace at {

static const std::string PYTORCH_RNG_TYPE_ENV_VAR = "PYTORCH_RNG_TYPE";

GeneratorType get_rng_type();

GeneratorType get_rng_type() {
  const char* env_var = std::getenv(PYTORCH_RNG_TYPE_ENV_VAR.c_str());
  std::string env_var_string = env_var ? env_var : "";

  if (env_var_string == "PCG")
    return GeneratorType::PCG;

  if (env_var_string == "PHILOX")
    return GeneratorType::PHILOX;

  if (env_var_string == "LCG")
    return GeneratorType::LCG;

  return GeneratorType::MT19937;
}


namespace mt19937_rng {

  void init_seed(uint64_t seed, mt19937_data_pod& d);

  void init_seed(uint64_t seed, mt19937_data_pod& d) {
    d.seed_ = seed;

    d.seeded_ = true;

    d.state_[0] = static_cast<uint32_t>(seed & 0xffffffff);

    for (const auto j : c10::irange(1, MERSENNE_STATE_N)) {
      d.state_[j] = 1812433253u * (d.state_[j-1] ^ (d.state_[j-1] >> 30)) + j;
    }

    d.left_ = 1;
    d.next_ = 0;
  }

}

namespace lcg_rng {

  void init_seed(uint64_t seed, mt19937_data_pod& d);

  uint32_t next(mt19937_data_pod& d);

  void init_seed(uint64_t seed, mt19937_data_pod& d) {
    d.seed_ = seed;
    d.seeded_ = true;
    d.state_[0] = seed ? static_cast<uint32_t>(seed % (1ULL << 31)) : 1u;
    d.left_ = 1;
    d.next_ = 0;
  }

  uint32_t next(mt19937_data_pod& d) {
    d.state_[0] = static_cast<uint32_t>((65539ULL * d.state_[0]) % (1ULL << 31));
    return d.state_[0];
  }

}

namespace philox_rng {
  static constexpr uint32_t PHILOX_M2x32_0 = 0xD256D193u;
  static constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9u;

  void init_seed(uint64_t seed, mt19937_data_pod& d);
  uint32_t next(mt19937_data_pod& d);

  static void philox2x32_10(uint32_t c0, uint32_t c1, uint32_t k, uint32_t& r0, uint32_t& r1) {

    for (int i = 0; i < 10; ++i) {
      uint64_t prod = static_cast<uint64_t>(PHILOX_M2x32_0) * c0;
      uint32_t hi = static_cast<uint32_t>(prod >> 32);
      uint32_t lo = static_cast<uint32_t>(prod & 0xffffffff);
      c0 = hi ^ k ^ c1;
      c1 = lo;
      k += PHILOX_W32_0;
    }

    r0 = c0;
    r1 = c1;

  }

  void init_seed(uint64_t seed, mt19937_data_pod& d) {
    d.seed_ = seed;
    d.seeded_ = true;
    d.state_[0] = 0u;
    d.state_[1] = 0u;
    d.state_[2] = static_cast<uint32_t>(seed & 0xffffffff);
    d.state_[3] = 0u;
    d.left_ = 0;
    d.next_ = 0;
  }

  uint32_t next(mt19937_data_pod& d) {
    if (d.left_ == 1) {
      d.left_ = 0;
      return d.state_[3];
    }

    uint32_t r0, r1;
    philox2x32_10(d.state_[0], d.state_[1], d.state_[2], r0, r1);

    if (++d.state_[0] == 0u)
      ++d.state_[1];

    d.state_[3] = r1;
    d.left_     = 1;

    return r0;
  }
}

namespace pcg_rng {
  static constexpr uint64_t PCG_MULT = 6364136223846793005ULL;
  static constexpr uint64_t PCG_INC = 1442695040888963407ULL;

  void init_seed(uint64_t seed, mt19937_data_pod& d);
  uint32_t next(mt19937_data_pod& d);

  static inline uint32_t rotr32(uint32_t x, uint32_t r) {
    return (x >> r) | (x << ((-r) & 31u));
  }

  void init_seed(uint64_t seed, mt19937_data_pod& d) {
    d.seed_ = seed;
    d.seeded_ = true;

    uint64_t s = (seed + PCG_INC) * PCG_MULT + PCG_INC;
    d.state_[0] = static_cast<uint32_t>(s & 0xffffffffu);
    d.state_[1] = static_cast<uint32_t>(s >> 32);
    d.left_ = 1;
    d.next_ = 0;
  }

  uint32_t next(mt19937_data_pod& d) {
    uint64_t state = (static_cast<uint64_t>(d.state_[1]) << 32) |  static_cast<uint64_t>(d.state_[0]);

    uint32_t xorshifted = static_cast<uint32_t>(((state >> 18u) ^ state) >> 27u);
    uint32_t rot = static_cast<uint32_t>(state >> 59u);
    uint32_t out = rotr32(xorshifted, rot);

    state = state * PCG_MULT + PCG_INC;
    d.state_[0] = static_cast<uint32_t>(state & 0xffffffffu);
    d.state_[1] = static_cast<uint32_t>(state >> 32);

    return out;
  }
}

mt19937_engine::mt19937_engine(uint64_t seed) {
  data_.rng_type_ = get_rng_type();
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
  if (!data_.seeded_)
    return false;

  switch (data_.rng_type_) {
    case GeneratorType::LCG:
    case GeneratorType::PCG:
    case GeneratorType::PHILOX:
      return true;
    default:
      return (data_.left_ > 0 && data_.left_ <= MERSENNE_STATE_N)
          && (data_.next_ <= MERSENNE_STATE_N);
  }
}

uint32_t mt19937_engine::operator()() {
  switch (data_.rng_type_) {
    case GeneratorType::LCG:
      return lcg_rng::next(data_);
    case GeneratorType::PHILOX:
      return philox_rng::next(data_);
    case GeneratorType::PCG:
      return pcg_rng::next(data_);
    default: {
      if (--(data_.left_) == 0) next_state();
      uint32_t y = *(data_.state_.data() + data_.next_++);
      y ^= (y >> 11);
      y ^= (y << 7)  & 0x9d2c5680u;
      y ^= (y << 15) & 0xefc60000u;
      y ^= (y >> 18);
      return y;
    }
  }
}

void mt19937_engine::init_with_uint32(uint64_t seed) {
  switch (data_.rng_type_) {
    case GeneratorType::LCG:
      lcg_rng::init_seed(seed, data_);
      return;
    case GeneratorType::PHILOX:
      philox_rng::init_seed(seed, data_);
      return;
    case GeneratorType::PCG:
      pcg_rng::init_seed(seed, data_);
      return;
    default:
      mt19937_rng::init_seed(seed, data_);
      return;
  }
}

uint32_t mt19937_engine::mix_bits(uint32_t u, uint32_t v) {
  return (u & UMASK) | (v & LMASK);
}

uint32_t mt19937_engine::twist(uint32_t u, uint32_t v) {
  return (mix_bits(u, v) >> 1) ^ (v & 1 ? MATRIX_A : 0u);
}

void mt19937_engine::next_state() {
  uint32_t* p = data_.state_.data();
  data_.left_ = MERSENNE_STATE_N;
  data_.next_ = 0;
  for (int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++)
    *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
  for (int j = MERSENNE_STATE_M; --j; p++)
    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
  *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], data_.state_[0]);
}

}