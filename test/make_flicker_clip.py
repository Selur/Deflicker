#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Writes an 8 bit YUV420 y4m test clip with synthetic intensity flicker (per-frame gain and offset on luma).

Usage: python make_flicker_clip.py SOURCE OUT.y4m [--bestsource path/to/libbestsource]
"""
import argparse
import random

import vapoursynth as vs

core = vs.core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--bestsource")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.bestsource:
        core.std.LoadPlugin(a.bestsource)
    src = core.resize.Bicubic(core.bs.VideoSource(a.source), format=vs.YUV420P8)
    rnd = random.Random(a.seed)
    frames = []
    for n in range(src.num_frames):
        gain, off = rnd.uniform(0.85, 1.15), rnd.uniform(-12, 12)
        frames.append(core.std.Expr(src[n], ["x %f * %f +" % (gain, off), ""]))
    clip = core.std.Splice(frames)
    with open(a.out, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420jpeg\n" % (clip.width, clip.height, clip.fps.numerator,
                                                                  clip.fps.denominator))
        for fr in clip.frames():
            f.write(b"FRAME\n")
            for p in range(3):
                f.write(bytes(fr[p]))


if __name__ == "__main__":
    main()
