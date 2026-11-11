// SPDX-License-Identifier: GPL-2.0-or-later
// VapourSynth (API 4) interface: deflicker.Deflicker, Gray and YUV, 8-16 bit integer and 32 bit float.
#include <memory>
#include <stdexcept>
#include <string>

#include "VSHelper4.h"
#include "VapourSynth4.h"
#include "core/deflicker.hpp"
#include "version.hpp"

using namespace deflicker;

namespace {

struct FilterData {
  VSNode* node = nullptr;
  int planes = 1;
  bool info = false;
  std::unique_ptr<Engine> engine;
};

const VSFrame* VS_CC getFrame(int n, int reason, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                              const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  if (reason == arInitial) {
    int first, last;
    d->engine->window(n, first, last);
    for (int k = first; k <= last; k++) vsapi->requestFrameFilter(k, d->node, ctx);
    return nullptr;
  }
  if (reason != arAllFramesReady) return nullptr;

  const VSFrame* src = vsapi->getFrameFilter(n, d->node, ctx);
  const Fetch fetch = [d, ctx, vsapi](int k) {
    std::shared_ptr<const VSFrame> frame(vsapi->getFrameFilter(k, d->node, ctx),
                                         [vsapi](const VSFrame* f) { vsapi->freeFrame(f); });
    LumaRef r;
    r.luma.data = vsapi->getReadPtr(frame.get(), 0);
    r.luma.stride = vsapi->getStride(frame.get(), 0);
    r.holder = frame;
    return r;
  };

  // chroma planes are shared with the source; writing them (info overlay) makes a private copy
  const VSFrame* plane_src[3] = {nullptr, src, src};
  const int plane_idx[3] = {0, 1, 2};
  VSFrame* dst = vsapi->newVideoFrame2(vsapi->getVideoFrameFormat(src), vsapi->getFrameWidth(src, 0),
                                       vsapi->getFrameHeight(src, 0), plane_src, plane_idx, src, core);
  Frame out;
  for (int i = 0; i < (d->info ? d->planes : 1); i++) {
    out.data[i] = vsapi->getWritePtr(dst, i);
    out.stride[i] = vsapi->getStride(dst, i);
  }
  ConstPlane luma;
  luma.data = vsapi->getReadPtr(src, 0);
  luma.stride = vsapi->getStride(src, 0);
  int err = 0;
  vsapi->mapGetInt(vsapi->getFramePropertiesRO(src), "_Field", 0, &err);
  const bool field_based = !err;
  try {
    d->engine->process(n, fetch, luma, out, field_based);
  } catch (const std::exception& e) {
    vsapi->setFilterError(e.what(), ctx);
    vsapi->freeFrame(src);
    vsapi->freeFrame(dst);
    return nullptr;
  }
  vsapi->freeFrame(src);
  return dst;
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  vsapi->freeNode(d->node);
  delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
  auto d = std::make_unique<FilterData>();
  d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
  auto fail = [&](const std::string& msg) {
    vsapi->mapSetError(out, msg.c_str());
    vsapi->freeNode(d->node);
  };

  int e;
  Params p;
  auto i = [&](const char* key, int& v) {
    const int64_t x = vsapi->mapGetInt(in, key, 0, &e);
    if (!e) v = int(x);
  };
  auto b = [&](const char* key, bool& v) {
    const int64_t x = vsapi->mapGetInt(in, key, 0, &e);
    if (!e) v = x != 0;
  };
  auto f = [&](const char* key, double& v) {
    const double x = vsapi->mapGetFloat(in, key, 0, &e);
    if (!e) v = x;
  };
  f("percent", p.percent);
  i("lag", p.lag);
  f("noise", p.noise);
  i("scene", p.scene);
  i("lmin", p.lmin);
  i("lmax", p.lmax);
  i("border", p.border);
  b("info", p.info);
  b("debug", p.debug);
  i("opt", p.opt);

  const VSVideoInfo* vi = vsapi->getVideoInfo(d->node);
  if (!vsh::isConstantVideoFormat(vi)) return fail("Deflicker: only constant format input is supported");
  const VSVideoFormat& vf = vi->format;
  Format fmt;
  fmt.width = vi->width;
  fmt.height = vi->height;
  fmt.bits = vf.bitsPerSample;
  fmt.is_float = vf.sampleType == stFloat;
  if (fmt.is_float ? vf.bitsPerSample != 32 : vf.bitsPerSample > 16)
    return fail("Deflicker: only 8-16 bit integer and 32 bit float clips are supported");
  switch (vf.colorFamily) {
    case cfGray:
      d->planes = 1;
      break;
    case cfYUV:
      d->planes = 3;
      fmt.chroma = true;
      fmt.ssw = vf.subSamplingW;
      fmt.ssh = vf.subSamplingH;
      break;
    default:
      return fail("Deflicker: input must be Gray or YUV");
  }
  d->info = p.info;
  try {
    d->engine = std::make_unique<Engine>(p, fmt, vi->numFrames);
  } catch (const std::exception& ex) {
    return fail(ex.what());
  }
  if (d->engine->passthrough()) {
    vsapi->mapConsumeNode(out, "clip", d->node, maReplace);
    d->node = nullptr;
    return;
  }

  VSFilterDependency deps[] = {{d->node, rpGeneral}};
  vsapi->createVideoFilter(out, "Deflicker", vi, getFrame, freeFilter, fmParallel, deps, 1, d.release(), core);
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
  vspapi->configPlugin("io.github.selur.deflicker", "deflicker", "Deflicker " DEFLICKER_VERSION,
                       VS_MAKE_VERSION(DEFLICKER_VERSION_MAJOR, DEFLICKER_VERSION_MINOR), VAPOURSYNTH_API_VERSION, 0,
                       plugin);
  vspapi->registerFunction("Deflicker",
                           "clip:vnode;percent:float:opt;lag:int:opt;noise:float:opt;scene:int:opt;lmin:int:opt;"
                           "lmax:int:opt;border:int:opt;info:int:opt;debug:int:opt;opt:int:opt;",
                           "clip:vnode;", create, nullptr, plugin);
}
