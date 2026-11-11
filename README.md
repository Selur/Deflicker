# Deflicker

Mean intensity stabilizer: removes the intensity flicker of old film by temporal smoothing of the mean luma, and
corrects the brightness pumping of automatic gain control after flashes. One plugin binary for **AviSynth+** and
**VapourSynth** (API 4), built for Windows x64, Linux x86_64 and macOS arm64.

- Versions 0.1–0.4 (2004) by Alexander G. Balakhnin aka Fizick
- Versions 0.5–0.6 (2019) by pinterf: AviSynth 2.6/AviSynth+ interface, x64, SSE2/AVX2
- Version 0.7.0: AviSynth+ and VapourSynth from one source, 10–16 bit and 32 bit float, NEON, multithreading,
  Linux and macOS builds

With 8 bit input the output is bit-identical to Deflicker 0.6 (see [Differences](#differences-to-06)).

## Description

When amateur film (16, 18 or 24 fps) is transferred to video by simply filming the projection, the brightness of the
frames varies strongly: the shutter and the video camera are not synchronized. Even transfers made frame by frame
show some flicker, caused by the uneven optical density of aged film stock. Stabilizing the brightness improves the
viewing impression and also helps the filters that follow, such as motion compensated noise reduction, which work
better when consecutive frames have the same brightness.

Deflicker uses a simplified version of the intensity flicker correction of the AURORA project (*AUtomated
Restoration of ORiginal film and video Archives*, an EU ACTS project started in 1995), described in

> P.M.B. van Roosmalen, *Restoration of Archived Film and Video*, PhD thesis, Delft University of Technology, 1999,
> ISBN 90-901-2792-5

The method uses a statistical degradation model with a multiplicative and an additive flicker component:

    z(n) = alfa(n) * y(n) + beta(n) + noise        z: measured luma, y: true luma

and estimates the correction

    y(n) = a(n) * z(n) + b(n)

from the evolution of the mean luma and its variance over a series of frames. Deflicker measures mean and variance of
every frame and propagates the estimate from a base frame up to `lag` frames back to the current frame, frame by
frame. The compensation is deliberately partial (`percent`), which keeps the correction stable. A scene change, found
as a large jump of the mean luma or of the standard deviation, or a nearly flat frame (variance below the noise),
starts a new base, so the averaging never crosses a cut.

Simplifications compared to the thesis: the correction is global (one gain and offset for the whole frame, no
regions) and motion is not taken into account.

The result is applied to the luma only (`Y' = mult * Y + add`, limited to `lmin`..`lmax`); chroma is copied
unchanged.

Similar filters work differently: Donald Graft's Antiflicker averages over a fixed window and suits strictly periodic
flicker; the VirtualDub Deflick filter needs two passes. Deflicker works in one pass with controlled smoothing.

## Requirements

| | AviSynth+ | VapourSynth |
|---|---|---|
| Interface | AviSynth+ (frame properties from interface version 8) | API 4 (R55 or newer) |
| Formats | Y, planar YUV and YUVA in 8, 10, 12, 14, 16 bit and 32 bit float; YUY2 | Gray and YUV in 8–16 bit integer and 32 bit float |
| Not supported | RGB | RGB, half float |
| File | `Deflicker.dll` | `Deflicker.dll`, `libDeflicker.so`, `libDeflicker.dylib` |

The clip should be progressive. For interlaced material separate the fields first (`SeparateFields()` /
`std.SeparateFields`) and weave them back afterwards.

## Usage

AviSynth+:

```
LoadPlugin("Deflicker.dll")   # or put it in the plugins folder for autoloading
Deflicker()
```

VapourSynth:

```python
clip = core.deflicker.Deflicker(clip)
```

The VapourSynth plugin is also available as a wheel (`pip install vapoursynth_deflicker-…whl`), which installs it
into `site-packages/vapoursynth/plugins` for autoloading.

## Syntax and parameters

```
Deflicker(clip, float "percent" = 85.0, int "lag" = 25, float "noise" = 10.0, int "scene" = 40,
          int "lmin" = 0, int "lmax" = 255, int "border" = 0, bool "info" = false, bool "debug" = false,
          int "opt" = -1)
```

All parameters are optional and named alike in both hosts (VapourSynth takes `True`/`False` or 1/0 for the
booleans). Luma values (`noise`, `scene`, `lmin`, `lmax`) are given on the 8 bit scale 0–255 for every format; see
[High bit depth](#high-bit-depth-and-float).

* **percent** — how strongly the luma of the previous frames is followed, in percent, 0.0–100.0, default 85.
  100 compensates fully, 0 does nothing. Values below 100 keep the correction stable.

* **lag** — the largest distance in frames from the base frame from which the mean luma is estimated and
  compensated, default 25. Negative values work backward in time (estimation from the following frames, see the
  examples). 0 returns the clip unchanged.

* **noise** — standard deviation of the luma noise caused by motion, grain etc., 0–255, default 10. A frame whose
  luma standard deviation is not above `noise` is treated as flat (bad) and ends the propagation.

* **scene** — scene change threshold, 0–255, default 40. A new scene starts where the mean luma of consecutive
  frames differs by more than `scene`, or twice the difference of their standard deviations does. Lower values detect
  more scene changes.

* **lmin**, **lmax** — limits of the output luma, 0 <= lmin <= lmax <= 255, defaults 0 and 255.

* **border** — width in pixels of a band at all four edges that is left out of the measurement (for black borders,
  sprocket holes, burnt-in text), default 0. The whole frame is still corrected.

* **info** — draws the measurement (frame, base frame, mean, standard deviation, mult, add, corrected mean) and the
  measured area onto the frame. "DeFlicker:NEW" marks the first frame of a scene. Default false.

* **debug** — writes the same numbers for every frame to the debug output: `OutputDebugString` on Windows (view
  with DebugView), stderr elsewhere. Default false.

* **opt** — instruction set, for testing: -1 automatic (default), 0 C, 1 or 2 SSE2, 3 AVX2 on x86-64; on ARM64 0 C,
  anything else NEON. A set the CPU lacks falls back to the next lower one. All give identical results.

## Examples

### Film flicker

The default parameters fit most transfers. To tune them, turn on `info`, look at the mean and deviation values across
frames and adjust `scene` (or the others) so that real cuts, and only those, are marked NEW.

```
AviSource("film.avi")
Deflicker(scene=30)
```

```python
clip = core.deflicker.Deflicker(clip, scene=30)
```

### Gain control pumping after flashes

Camcorders regulate their gain with some rise and fall time. After a flash or a sudden change of brightness the gain
drops and recovers slowly, so the following frames are too dark. Correcting this needs the estimation from the
*following* frames, i.e. working backward in time. Instead of `Reverse()` around the filter, use a negative `lag`:
while the dark frames after the flash are processed, the flash itself is not used for smoothing, because it is
detected as a scene change (as is the frame before it).

```
Deflicker(lag=-25)
```

### Interlaced source

```
AssumeTFF()
SeparateFields()
Deflicker()
Weave()
```

### Place in a film restoration chain

Deflicker is sensitive to global motion (and somewhat to local motion), while motion search, spot removal and
temporal denoising are sensitive to brightness changes. A sensible order:

1. color and brightness correction
2. global motion compensation (e.g. DePan)
3. **Deflicker**
4. local motion compensation (MVTools)
5. temporal spot removal (DeSpot)
6. temporal noise reduction
7. spatial smoothing
8. global motion stabilization
9. encoding

## High bit depth and float

- Every format is processed at its own precision; `noise`, `scene`, `lmin` and `lmax` stay on the 8 bit scale and
  are scaled with the format: multiplied by 2^(bits-8) for 10–16 bit, divided by 255 for float. `lmax=255` always
  means the largest value of the format (1023, 65535, 1.0).
- 10–16 bit: the correction is computed in single precision float and rounded to the nearest value (8 bit keeps the
  fixed point arithmetic of 0.6, the gain in steps of 1/256).
- Float: luma is limited to `lmin/255`..`lmax/255`, i.e. 0.0–1.0 by default. Statistics are accumulated in double
  precision.

## Differences to 0.6

- 8 bit output is bit-identical to Deflicker 0.6 for Y8, YV12, YV16, YV24, YV411 and YUY2 and all parameters except
  `info` (checked frame by frame against the original DLL).
- `info`: the text is placed inside the frame, at about a quarter of the width in the row 0.6 used, and clipped at
  the edges. 0.6 computed the position from the row size in bytes and wrote past the frame width, which
  could crash on small frames.
- Invalid `lmin`/`lmax`, `border` (must leave part of the frame), negative `noise` or `scene` are errors.
- Results do not depend on the order in which frames are requested; the filter is thread-safe (`MT_NICE_FILTER`,
  `fmParallel`). 0.6 kept a per-instance cache that was not.
- Frames of any width: 0.6 summed the squares of a row in 32 bit and overflowed beyond about 33000 pixels.
- `opt=1` means SSE2 (the MMX/integer SSE code of the 32 bit build is gone).
- In the extreme case of a gain of 128 or more, all instruction sets use the SSE2 arithmetic; 0.6 handled the last
  pixels of a row (C code) differently.
- `debug` writes to stderr outside Windows.
- New: 10–16 bit, float, VapourSynth, NEON, Linux and macOS.

## Building

```
meson setup build
ninja -C build
```

Windows with MSVC: `meson setup build --vsenv -Db_vscrt=mt` (or `build.bat`). The AVX2 kernels are compiled
separately and only used when the CPU supports AVX2. A VapourSynth wheel: `pip install .` (meson-python).

The GitHub Actions workflow builds and tests wheels and plugin archives for Windows x64, Linux x86_64 (manylinux_2_28)
and macOS arm64, and attaches them to the release for `v*` tags.

## Tests

- `test/test_deflicker.py` (VapourSynth, runs in CI on all platforms): every instruction set against C for all
  formats and several parameter sets, flicker reduction per scene, 8 bit/16 bit/float agreement, chroma kept,
  request order independence, overlay on tiny frames, parameter errors.
- `test/compare_original.py` (Windows, AviSynth+): MD5 of every frame against the original Deflicker 0.6 DLL for six
  colorspaces, 17 parameter sets and every instruction set of the port.
- `test/compare_hosts.py` (Windows): AviSynth+ against VapourSynth output of the same build, 8/10/16 bit and Gray.
- `test/make_flicker_clip.py`: makes a y4m test clip with synthetic flicker from any source.

## Changelog

| Version | Date | Changes |
|---|---|---|
| 0.7.0 | 2026-11-11 | AviSynth+ and VapourSynth API 4 from one source; 10–16 bit and 32 bit float; NEON (ARM64); thread-safe statistics cache; Linux and macOS builds with Meson; overlay kept inside the frame; parameter checks |
| 0.6 | 2019-05-23 | pinterf: SSE2 and AVX2 (was ISSE in 32 bit, C in x64); no overflow above 8 Mpixel; all pixels used for non-YUY2 formats (was every second one) |
| 0.5 | 2019-05-22 | pinterf: Visual Studio 2019, AviSynth 2.6 interface with AviSynth+ headers, all 8 bit Y and YUV formats, x64 |
| 0.4 | 2004-08-16 | Fizick: YUY2 support, ISSE optimizations |
| 0.3 | 2004-06-06 | Fizick: border shown in info mode |
| 0.2 | 2004-06-05 | Fizick: reverse mode (negative lag), info |
| 0.1 | 2004-06-03 | Fizick: first (not public) version |

## License

GNU General Public License, version 2 or (at your option) any later version; see [LICENSE](LICENSE). The info
overlay font comes from `info.cpp` by Donald Graft, extended for high bit depth by pinterf. The vendored AviSynth+ and
VapourSynth headers keep their own licenses.

## Links

- Original documentation: http://avisynth.nl/index.php/DeFlicker
- pinterf's 0.6 sources: https://github.com/pinterf/Deflicker
- Fizick's original page: http://avisynth.org.ru/deflicker/deflicker.html
