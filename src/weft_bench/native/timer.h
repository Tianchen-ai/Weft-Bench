#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace wb {
using Clock = std::chrono::steady_clock;
inline double option(const char *key, double fallback) {
  const char *value = std::getenv(key);
  if (!value) return fallback;
  char *end = nullptr;
  double parsed = std::strtod(value, &end);
  if (end == value || *end || !std::isfinite(parsed)) throw std::runtime_error(key);
  return parsed;
}
inline void barrier() { asm volatile("" ::: "memory"); }
inline double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
inline void evict(std::vector<std::uint8_t> &data) {
  for (std::size_t i = 0; i < data.size(); i += 64) data[i] += 1;
  barrier();
}

// reset() restores mutable input between invocations, outside the measured region.
// Algorithm-required output initialization belongs inside run().
template<class Run, class Reset>
void measure(Run run, Reset reset, const char *correctness = "within-tolerance", bool stateful = false, const char *implementation = "") {
  const int warmup = static_cast<int>(option("WB_WARMUP", 3));
  const int samples = static_cast<int>(option("WB_SAMPLES", 5));
  const double budget = option("WB_BATCH_MS", 10);
  const char *mode = std::getenv("WB_CACHE");
  const bool cold = mode && std::strcmp(mode, "cold") == 0;
  if (warmup < 0 || samples < 1 || budget <= 0) throw std::runtime_error("invalid timing settings");
  std::vector<std::uint8_t> eviction(cold ? static_cast<std::size_t>(option("WB_EVICTION_BYTES", 67108864)) : 0, 1);
  for (int i = 0; i < warmup; ++i) { reset(); run(); barrier(); }
  reset();
  auto start = Clock::now();
  run(); barrier();
  double estimate = std::max(milliseconds(start), 0.000001);
  std::size_t iterations = cold ? 1 : std::min<std::size_t>(1000000, std::max<std::size_t>(1, std::ceil(budget / estimate)));
  std::vector<double> timings;
  for (int sample = 0; sample < samples; ++sample) {
    double elapsed = 0;
    if (stateful || cold) {
      for (std::size_t i = 0; i < iterations; ++i) {
        reset();
        if (cold) evict(eviction);
        start = Clock::now();
        run(); barrier();
        elapsed += milliseconds(start);
      }
    } else {
      reset();
      start = Clock::now();
      for (std::size_t i = 0; i < iterations; ++i) { run(); barrier(); }
      elapsed = milliseconds(start);
    }
    timings.push_back(elapsed / iterations);
  }
  std::printf("WB_RESULT {\"implementation\":\"%s\",\"samples_ms\":[", implementation);
  for (std::size_t i = 0; i < timings.size(); ++i) std::printf("%s%.12g", i ? "," : "", timings[i]);
  std::printf("],\"iterations\":%zu,\"correctness\":\"%s\",\"cache\":\"%s\"}\n", iterations, correctness, cold ? "cold" : "reuse");
}
template<class Run> void measure(Run run, const char *correctness = "within-tolerance") {
  measure(run, [] {}, correctness);
}
}
