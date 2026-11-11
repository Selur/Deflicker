// SPDX-License-Identifier: GPL-2.0-or-later
// AviSynth+ interface: Deflicker() with the arguments of 0.6; YUY2, Y and planar YUV(A), 8-16 bit and float.
#include <memory>
#include <stdexcept>
#include <string>

#include "avisynth.h"
#include "core/deflicker.hpp"
#include "version.hpp"

using namespace deflicker;

namespace {

class DeflickerAvs : public GenericVideoFilter {
 public:
  DeflickerAvs(PClip child, const Params& p, const Format& fmt, IScriptEnvironment* env)
      : GenericVideoFilter(child), fmt_(fmt) {
    try {
      env->CheckVersion(8);
      has_v8_ = true;
    } catch (const AvisynthError&) {
      has_v8_ = false;
    }
    try {
      engine_ = std::make_unique<Engine>(p, fmt_, vi.num_frames);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
  }

  bool passthrough() const { return engine_->passthrough(); }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override {
    n = n < 0 ? 0 : (n >= vi.num_frames ? vi.num_frames - 1 : n);
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = has_v8_ ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);
    if (fmt_.layout == Layout::Planar) {
      static const int planes[] = {PLANAR_U, PLANAR_V, PLANAR_A};
      const int count = vi.NumComponents() - 1;
      for (int i = 0; i < count; i++)
        env->BitBlt(dst->GetWritePtr(planes[i]), dst->GetPitch(planes[i]), src->GetReadPtr(planes[i]),
                    src->GetPitch(planes[i]), src->GetRowSize(planes[i]), src->GetHeight(planes[i]));
    }
    const Fetch fetch = [this, env](int k) {
      auto frame = std::make_shared<PVideoFrame>(child->GetFrame(k, env));
      LumaRef r;
      r.luma.data = (*frame)->GetReadPtr();
      r.luma.stride = (*frame)->GetPitch();
      r.holder = frame;
      return r;
    };
    Frame out;
    out.data[0] = dst->GetWritePtr();
    out.stride[0] = dst->GetPitch();
    if (fmt_.chroma) {
      out.data[1] = dst->GetWritePtr(PLANAR_U);
      out.stride[1] = dst->GetPitch(PLANAR_U);
      out.data[2] = dst->GetWritePtr(PLANAR_V);
      out.stride[2] = dst->GetPitch(PLANAR_V);
    }
    ConstPlane luma;
    luma.data = src->GetReadPtr();
    luma.stride = src->GetPitch();
    try {
      engine_->process(n, fetch, luma, out, vi.IsFieldBased());
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    (void)frame_range;
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

 private:
  Format fmt_;
  bool has_v8_ = false;
  std::unique_ptr<Engine> engine_;
};

Format formatOf(const VideoInfo& vi, IScriptEnvironment* env) {
  Format f;
  f.width = vi.width;
  f.height = vi.height;
  f.bits = vi.BitsPerComponent();
  f.is_float = f.bits == 32;
  if (vi.IsYUY2()) {
    f.layout = Layout::YUY2;
  } else if (vi.IsY()) {
    f.layout = Layout::Planar;
  } else if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    f.layout = Layout::Planar;
    f.chroma = true;
    f.ssw = vi.GetPlaneWidthSubsampling(PLANAR_U);
    f.ssh = vi.GetPlaneHeightSubsampling(PLANAR_U);
  } else {
    env->ThrowError("Deflicker: input to filter must be Y, YUV or YUY2!");
  }
  return f;
}

AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env) {
  PClip child = args[0].AsClip();
  Params p;
  p.percent = args[1].AsFloat(85.0f);
  p.lag = args[2].AsInt(25);
  p.noise = args[3].AsFloat(10.0f);
  p.scene = args[4].AsInt(40);
  p.lmin = args[5].AsInt(0);
  p.lmax = args[6].AsInt(255);
  p.border = args[7].AsInt(0);
  p.info = args[8].AsBool(false);
  p.debug = args[9].AsBool(false);
  p.opt = args[10].AsInt(-1);
  const Format fmt = formatOf(child->GetVideoInfo(), env);
  auto* filter = new DeflickerAvs(child, p, fmt, env);
  if (filter->passthrough()) {
    delete filter;
    return child;
  }
  return filter;
}

}  // namespace

#ifdef _WIN32
#define DEFLICKER_AVS_EXPORT extern "C" __declspec(dllexport)
#else
#define DEFLICKER_AVS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

const AVS_Linkage* AVS_linkage = nullptr;

DEFLICKER_AVS_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env,
                                                               const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  env->AddFunction("Deflicker", "c[percent]f[lag]i[noise]f[scene]i[lmin]i[lmax]i[border]i[info]b[debug]b[opt]i",
                   Create, nullptr);
  return "Deflicker " DEFLICKER_VERSION;
}
