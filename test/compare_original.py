#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compares the port with the original Deflicker 0.6 DLL (pinterf, 2019) inside AviSynth+ (Windows).

Every clip is an 8 bit y4m file, read with BestSource in both scripts, so both filters see identical frames. For each
colorspace and parameter set the MD5 of every output frame must match. info=true is left out: the port places the
overlay inside the frame (see README, differences).

Usage: python compare_original.py --original Deflicker.dll --plugin build/Deflicker.dll --work DIR --avs2yuv avs2yuv64.exe
       --avs-bestsource BestSource.dll CLIP.y4m...
"""
import argparse
import concurrent.futures
import hashlib
import os
import subprocess
import sys


# colorspace -> (conversion, output conversion for avs2yuv, bytes per output frame)
FORMATS = {
    "YV12": ("", "", lambda w, h: w * h * 3 // 2),
    "YV16": ("ConvertToYV16()", "", lambda w, h: w * h * 2),
    "YV24": ("ConvertToYV24()", "", lambda w, h: w * h * 3),
    "Y8": ("ConvertToY8()", "", lambda w, h: w * h),
    "YV411": ("ConvertToYV411()", '.ConvertToYV24(chromaresample="point")', lambda w, h: w * h * 3),
    "YUY2": ("ConvertToYUY2()", ".ConvertToYV16()", lambda w, h: w * h * 2),
}

PARAMS = [
    "",
    "debug=true",
    "lag=-10",
    "lag=-25, scene=20",
    "lag=5, percent=100",
    "lag=1",
    "percent=0",
    "percent=50, noise=4, scene=20",
    "noise=0",
    "noise=40",
    "scene=5",
    "scene=255",
    "lmin=16, lmax=235",
    "lmin=30, lmax=200, percent=95",
    "border=16",
    "border=3, lag=-7",
    "lag=60",
]
ALL_FORMAT_PARAMS = {"", "lag=-10", "percent=50, noise=4, scene=20", "lmin=16, lmax=235", "border=16"}
# the port's instruction sets against the original's default (AVX2 on current CPUs)
PORT_OPTS = ["", "opt=0", "opt=2", "opt=3"]


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def run(avs2yuv, script_path, text, frame_size):
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(text)
    p = subprocess.run([avs2yuv, "-nstdr", "-raw", script_path, "-o", "-"], capture_output=True)
    data = p.stdout
    if p.returncode != 0 or not data or len(data) % frame_size:
        raise RuntimeError("%s: rc=%d len=%d %s" % (script_path, p.returncode, len(data),
                                                     p.stderr.decode(errors="replace").strip()[-600:]))
    return [hashlib.md5(data[i:i + frame_size]).hexdigest() for i in range(0, len(data), frame_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--avs2yuv", required=True, help="avs2yuv64 executable")
    ap.add_argument("--avs-bestsource", required=True, help="BestSource plugin for AviSynth+")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    cases = []
    for ci, clip in enumerate(a.clips):
        for fmt, (conv, post, size) in FORMATS.items():
            for prm in PARAMS:
                if fmt != "YV12" and prm not in ALL_FORMAT_PARAMS:
                    continue
                for opt in (PORT_OPTS if prm in ALL_FORMAT_PARAMS else [""]):
                    cases.append((ci, clip, fmt, conv, post, size(a.width, a.height), prm, opt))

    def one(idx_case):
        idx, (ci, clip, fmt, conv, post, fsize, prm, opt) = idx_case
        args = ", ".join(x for x in (prm, opt) if x)
        head = 'LoadPlugin("%s")\nLoadPlugin("%%s")\nBSVideoSource("%s")\n%s\n' % (fwd(a.avs_bestsource), fwd(clip), conv)
        orig = run(a.avs2yuv, os.path.join(a.work, "o%04d.avs" % idx), head % fwd(a.original) + "Deflicker(%s)%s\n" % (prm, post),
                   fsize)
        port = run(a.avs2yuv, os.path.join(a.work, "p%04d.avs" % idx), head % fwd(a.plugin) + "Deflicker(%s)%s\n" % (args, post),
                   fsize)
        diff = [i for i, (x, y) in enumerate(zip(orig, port)) if x != y]
        return ci, fmt, args, len(orig), len(port), diff

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for ci, fmt, args, no, np_, diff in ex.map(one, enumerate(cases)):
            ok = not diff and no == np_
            failures += not ok
            print("%s clip%d %-6s Deflicker(%s): %d frames%s" % ("ok  " if ok else "FAIL", ci, fmt, args, no,
                                                                 "" if ok else " differ at %s" % diff[:10]))
    print("%d of %d cases differ" % (failures, len(cases)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
