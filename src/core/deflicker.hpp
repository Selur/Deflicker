// SPDX-License-Identifier: GPL-2.0-or-later
// Host-independent Deflicker engine: per-frame luma statistics, temporal mean intensity stabilization (simplified
// AURORA method by A.G. Balakhnin aka Fizick), luma correction and the info overlay.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kernels.hpp"

namespace deflicker {

enum class Layout { Planar, YUY2 };

struct Format {
  int width = 0;   // luma samples
  int height = 0;
  int bits = 8;    // 8-16 for integer, 32 for float
  bool is_float = false;
  Layout layout = Layout::Planar;
  bool chroma = false;  // planar U and V present (YUV); false for Gray / Y
  int ssw = 0, ssh = 0; // chroma subsampling (log2)
};

struct Params {
  double percent = 85.0;
  int lag = 25;
  double noise = 10.0;
  int scene = 40;
  int lmin = 0;
  int lmax = 255;
  int border = 0;
  bool info = false;
  bool debug = false;
  int opt = -1;
};

struct ConstPlane {
  const uint8_t* data = nullptr;
  ptrdiff_t stride = 0;
};

// Planar: Y, U, V. YUY2: the packed plane in [0].
struct Frame {
  uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};
};

// Luma plane of a source frame plus whatever keeps it alive.
struct LumaRef {
  ConstPlane luma;
  std::shared_ptr<const void> holder;
};
using Fetch = std::function<LumaRef(int)>;

class Engine {
 public:
  Engine(const Params& p, const Format& f, int num_frames);  // throws std::runtime_error

  // lag = 0: the filter returns the source unchanged.
  bool passthrough() const { return range_ == 0; }
  // Source frames process(n) may need.
  void window(int n, int& first, int& last) const;
  // Writes the corrected luma of frame n to dst. Chroma (planar) is the host's job; with info=true it must already
  // hold a writable copy of the source chroma, because the overlay draws on it.
  void process(int n, const Fetch& fetch, ConstPlane src, Frame& dst, bool field_based);

  Isa isa() const { return isa_; }

 private:
  struct Stats {
    double mean;  // integer formats: truncated to an integer as in Deflicker 0.6
    double var;
  };
  Stats stats(int k, const Fetch& fetch, int n, ConstPlane src);
  Stats measure(ConstPlane p) const;
  void overlay(int n, int base, const Stats& cur, float mult, float add, float smoothed, Frame& dst,
               bool field_based) const;

  Format fmt_;
  int num_frames_;
  int lag_, range_, border_;
  bool info_, debug_;
  float percent_;
  float varnoise_;     // (noise in sample units)^2
  double scene_;       // mean luma test, sample units
  float scene_f_;      // standard deviation test, sample units
  int lmin8_, lmax8_;  // 8 bit scale as given
  float lo_, hi_;      // output clamp in sample units (high bit depth and float)
  Isa isa_;
  const Kernels* k_;

  std::mutex mutex_;
  std::unordered_map<int, Stats> cache_;
  std::deque<int> order_;
  size_t capacity_;
};

}  // namespace deflicker
