#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compares the VapourSynth and the AviSynth+ interface of the same Deflicker build (Windows).

Both hosts read the same 8 bit y4m clip with BestSource; higher bit depths are made by a bit shift in both hosts.
Every output byte must match.

Usage: python compare_hosts.py --plugin build/Deflicker.dll --work DIR --avs2yuv avs2yuv64.exe
       --avs-bestsource BestSource.dll --vs-bestsource libbestsource.dll CLIP.y4m...   (VapourSynth Python)
"""
import argparse
import hashlib
import os
import subprocess
import sys

import vapoursynth as vs


PARAMS = ["", "lag=-10", "percent=50, noise=4, scene=20", "lmin=16, lmax=235", "border=16, info=true",
          "lag=5, percent=100, info=true"]
# (AviSynth conversion, VapourSynth format, bit shift)
FORMATS = {
    "YUV420P8": ("", vs.YUV420P8, 0),
    "Y8": ("ConvertToY8()", vs.GRAY8, 0),
    "YUV420P10": ("ConvertBits(10)", vs.YUV420P10, 2),
    "YUV420P16": ("ConvertBits(16)", vs.YUV420P16, 8),
    "Y16": ("ConvertToY8().ConvertBits(16)", vs.GRAY16, 8),
}


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def avs_frames(avs2yuv, path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    p = subprocess.run([avs2yuv, "-nstdr", "-raw", path, "-o", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("%s: %s" % (path, p.stderr.decode(errors="replace")[-400:]))
    return p.stdout


def vs_bytes(clip):
    out = bytearray()
    for f in clip.frames():
        for p in range(f.format.num_planes):
            out += bytes(f[p])  # plane rows without stride padding
    return bytes(out)


def args_vs(prm):
    kw = {}
    for part in filter(None, prm.split(", ")):
        k, v = part.split("=", 1)
        if v in ("true", "false"):
            kw[k] = v == "true"
        else:
            kw[k] = float(v) if "." in v or k in ("percent", "noise") else int(v)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--avs2yuv", required=True, help="avs2yuv64 executable")
    ap.add_argument("--avs-bestsource", required=True, help="BestSource plugin for AviSynth+")
    ap.add_argument("--vs-bestsource", required=True, help="BestSource plugin for VapourSynth")
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    core = vs.core
    core.std.LoadPlugin(a.vs_bestsource)
    core.std.LoadPlugin(os.path.abspath(a.plugin))
    failures = total = 0
    for ci, clip in enumerate(a.clips):
        src = core.bs.VideoSource(clip)
        for name, (conv, vfmt, shift) in FORMATS.items():
            vsrc = core.std.ShufflePlanes(src, 0, vs.GRAY) if vfmt in (vs.GRAY8, vs.GRAY16) else src
            if shift:
                vsrc = core.std.Expr(vsrc, "x %d *" % (1 << shift), format=vfmt)
            for prm in PARAMS:
                total += 1
                avs = avs_frames(a.avs2yuv, os.path.join(a.work, "h%04d.avs" % total), 'LoadPlugin("%s")\nLoadPlugin("%s")\n'
                                 'BSVideoSource("%s")\n%s\nDeflicker(%s)\n' % (fwd(a.avs_bestsource), fwd(a.plugin), fwd(clip), conv,
                                                                               prm))
                vsb = vs_bytes(core.deflicker.Deflicker(vsrc, **args_vs(prm)))
                ok = hashlib.md5(avs).digest() == hashlib.md5(vsb).digest()
                failures += not ok
                print("%s clip%d %-10s Deflicker(%s)%s" % ("ok  " if ok else "FAIL", ci, name, prm,
                                                           "" if ok else " (%d vs %d bytes)" % (len(avs), len(vsb))))
    print("%d of %d cases differ" % (failures, total))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
