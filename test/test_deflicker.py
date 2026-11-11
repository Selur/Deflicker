#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Functional tests for the Deflicker VapourSynth plugin (API 4); run in CI on every platform.

Usage: python test/test_deflicker.py [path/to/Deflicker plugin]

Without an argument the plugin is expected to be autoloaded (e.g. from the installed wheel).
Bit-exactness against the original AviSynth plugin is checked by compare_original.py (Windows only).
"""
import hashlib
import platform
import sys

import vapoursynth as vs

core = vs.core
FAILURES = []
W, H = 333, 201          # odd sizes exercise every SIMD tail
SCENE_LEN, SCENES = 20, 3


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("FAIL: " + msg)
        return
    print("ok  ", msg)


def expect_error(fn, msg, contains):
    try:
        fn()
    except vs.Error as e:
        check(contains in str(e), "%s (%s)" % (msg, str(e).strip().splitlines()[-1]))
    else:
        check(False, "no error raised: " + msg)


def lcg(seed):
    state = seed & 0x7FFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield (state >> 8) / float(1 << 23)


def texture(seed, w, h, lo, hi, cells=(12, 9)):
    """Smooth deterministic GRAYS texture: a grid of random grey cells, bicubic upscaled."""
    rnd = lcg(seed)
    rows = []
    for _ in range(cells[1]):
        row = [core.std.BlankClip(width=4, height=4, format=vs.GRAYS, length=1, color=lo + (hi - lo) * next(rnd))
               for _ in range(cells[0])]
        rows.append(core.std.StackHorizontal(row))
    grid = core.std.StackVertical(rows)
    return core.resize.Bicubic(grid, w, h)


def source(fmt, w=W, h=H):
    """Three scenes with strong frame to frame intensity flicker (gain and offset), in format fmt."""
    f = core.get_video_format(fmt)
    w, h = w >> f.subsampling_w << f.subsampling_w, h >> f.subsampling_h << f.subsampling_h
    scenes = []
    for s in range(SCENES):
        luma = texture(101 + s, w, h, *((0.1, 0.4), (0.5, 0.8), (0.2, 0.5))[s])
        flick = []
        rnd = lcg(7 + s)
        for _ in range(SCENE_LEN):
            gain, off = 0.92 + 0.16 * next(rnd), -0.03 + 0.06 * next(rnd)
            flick.append(core.std.Expr(luma, "x %f * %f +" % (gain, off)))
        scenes.append(core.std.Splice(flick))
    y = core.std.Splice(scenes)
    if f.color_family == vs.YUV:
        cw, ch = w >> f.subsampling_w, h >> f.subsampling_h
        u = texture(201, cw, ch, -0.2, 0.2) * y.num_frames
        v = texture(301, cw, ch, -0.15, 0.25) * y.num_frames
        # chroma of float YUV is centered at 0; the integer conversion adds the offset
        clip = core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)
        if f.sample_type == vs.INTEGER:
            peak = (1 << f.bits_per_sample) - 1
            half = 1 << (f.bits_per_sample - 1)
            clip = core.std.Expr(clip, ["x %d *" % peak, "x %d * %d +" % (peak, half)],
                                 format=f.replace(bits_per_sample=32, sample_type=vs.FLOAT).id)
            clip = core.std.Expr(clip, ["x", "x"], format=fmt)
        else:
            clip = core.resize.Point(clip, format=fmt)
        return clip
    if f.sample_type == vs.INTEGER:
        peak = (1 << f.bits_per_sample) - 1
        return core.std.Expr(y, "x %d *" % peak, format=fmt)
    return y


def digest(clip, frames=None):
    h = hashlib.md5()
    for n in (range(clip.num_frames) if frames is None else frames):
        f = clip.get_frame(n)
        for p in range(f.format.num_planes):
            h.update(bytes(f[p]))
    return h.hexdigest()


def means(clip):
    st = core.std.PlaneStats(clip)
    return [st.get_frame(n).props["PlaneStatsAverage"] for n in range(clip.num_frames)]


def spread(values):
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


FORMATS = [vs.GRAY8, vs.YUV420P8, vs.YUV422P8, vs.YUV444P8, vs.YUV420P10, vs.YUV444P12, vs.GRAY16, vs.YUV420P16,
           vs.GRAYS, vs.YUV444PS]
PARAMS = [{}, {"border": 7, "info": True}, {"lag": -10}, {"percent": 50.0, "noise": 4.0, "scene": 20},
          {"lmin": 16, "lmax": 235}, {"lag": 3, "percent": 100.0}]


def main():
    if len(sys.argv) > 1:
        core.std.LoadPlugin(sys.argv[1])
    check(hasattr(core, "deflicker") and hasattr(core.deflicker, "Deflicker"), "plugin loaded (namespace deflicker)")
    df = core.deflicker.Deflicker
    arm = platform.machine().lower() in ("arm64", "aarch64")
    opts = [0, 1] if arm else [0, 1, 2, 3]

    # 1. every instruction set gives the same output as C, every format and parameter set
    for fmt in FORMATS:
        src = source(fmt)
        name = core.get_video_format(fmt).name
        for prm in PARAMS:
            ref = digest(df(src, opt=0, **prm))
            same = all(digest(df(src, opt=o, **prm)) == ref for o in opts[1:])
            check(same, "%s %s: opt %s identical to C" % (name, prm, opts))

    # 2. flicker is reduced inside each scene, scene changes are kept
    for fmt in (vs.YUV420P8, vs.YUV420P16, vs.YUV444PS):
        src = source(fmt)
        out = df(src)
        name = core.get_video_format(fmt).name
        mi, mo = means(src), means(out)
        for s in range(SCENES):
            a, b = s * SCENE_LEN + 5, (s + 1) * SCENE_LEN
            si, so = spread(mi[a:b]), spread(mo[a:b])
            check(so < 0.5 * si, "%s scene %d: mean luma spread %.4f -> %.4f" % (name, s, si, so))
        check(abs(mo[SCENE_LEN] - mi[SCENE_LEN]) < 1e-9 and abs(mo[2 * SCENE_LEN] - mi[2 * SCENE_LEN]) < 1e-9,
              "%s: first frame of each scene unchanged" % name)

    # 3. 8 bit, 16 bit and float agree (thresholds scale with the bit depth)
    s8 = source(vs.GRAY8)
    m8 = means(df(s8))
    m16 = means(df(core.std.Expr(s8, "x 256 *", format=vs.GRAY16)))
    mf = means(df(core.std.Expr(s8, "x 255 /", format=vs.GRAYS)))
    d16 = max(abs(a - b) for a, b in zip(m8, m16))
    dfl = max(abs(a - b) for a, b in zip(m8, mf))
    check(d16 < 2.5 / 255 and dfl < 2.5 / 255, "8 bit vs 16 bit / float mean luma within 2.5 levels (%.5f, %.5f)"
          % (d16 * 255, dfl * 255))

    # 4. lag=0 returns the source; chroma is never changed without info; info leaves the source intact
    src = source(vs.YUV420P8)
    check(digest(df(src, lag=0)) == digest(src), "lag=0 is the identity")
    out = df(src)
    same_chroma = all(bytes(out.get_frame(n)[p]) == bytes(src.get_frame(n)[p]) for n in (0, 9, 33) for p in (1, 2))
    check(same_chroma, "chroma planes copied")
    before = digest(src)
    info = df(src, info=True, border=10)
    digest(info)
    check(digest(src) == before and digest(info) != digest(out), "info overlay drawn, source frames unchanged")

    # 5. results do not depend on the request order (statistics cache, threads)
    for fmt in (vs.YUV420P8, vs.YUV420P10, vs.GRAYS):
        src = source(fmt)
        seq = [digest(df(src, lag=8), [n]) for n in range(src.num_frames)]
        order = list(range(src.num_frames))[::-1]
        order = order[::3] + order[1::3] + order[2::3]
        fresh = df(src, lag=8)
        rnd = {n: digest(fresh, [n]) for n in order}
        check(all(rnd[n] == seq[n] for n in range(src.num_frames)), "%s: order independent" % src.format.name)

    # 6. tiny frames with the overlay
    for fmt in (vs.GRAY8, vs.YUV420P8, vs.YUV420P10, vs.YUV444PS):
        tiny = source(fmt, 16, 12)
        try:
            digest(df(tiny, info=True, border=2))
            check(True, "%s 16x12 with info" % core.get_video_format(fmt).name)
        except vs.Error as e:
            check(False, "%s 16x12 with info: %s" % (core.get_video_format(fmt).name, e))

    # 7. parameter and format errors
    src = source(vs.YUV420P8)
    expect_error(lambda: df(src, percent=101.0), "percent > 100", "percent")
    expect_error(lambda: df(src, lmin=100, lmax=50), "lmin > lmax", "lmin")
    expect_error(lambda: df(src, lmax=300), "lmax > 255", "lmax")
    expect_error(lambda: df(src, border=101), "border too wide", "border")
    expect_error(lambda: df(src, noise=-1.0), "negative noise", "noise")
    expect_error(lambda: df(core.std.BlankClip(format=vs.RGB24)), "RGB input", "Gray or YUV")
    expect_error(lambda: df(core.std.BlankClip(format=vs.YUV420PH)), "half float input", "32 bit float")

    print("%d failure(s)" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
