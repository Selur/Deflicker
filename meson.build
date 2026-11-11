project('Deflicker', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version : '>=0.64.0',
  version : '0.7.0'
)

sources = [
  'src/core/deflicker.cpp',
  'src/core/text.cpp',
  'src/core/cpu.cpp',
  'src/core/kernels_c.cpp',
  'src/avs_plugin.cpp',
  'src/vs_plugin.cpp',
]

# Vendored AviSynth+ (3.7.5) and VapourSynth (API 4) headers.
incs = include_directories('include/avisynth', 'include/vapoursynth', 'src')

cxx = meson.get_compiler('cpp')
msvc = cxx.get_argument_syntax() == 'msvc'
cpu = host_machine.cpu_family()

if msvc
  # Same float results as the MSVC build of Deflicker 0.6; no contraction into FMA.
  add_project_arguments('/fp:precise', '/D_CRT_SECURE_NO_WARNINGS', language : 'cpp')
else
  # Separate float multiply and add everywhere, so C, SSE2, AVX2 and NEON give identical results.
  add_project_arguments('-Wall', '-Wextra', '-ffp-contract=off', language : 'cpp')
  if cpu in ['x86', 'x86_64']
    add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
  endif
endif

simd_libs = []
if cpu in ['x86', 'x86_64']
  sources += 'src/core/kernels_sse2.cpp'
  simd_libs += static_library('kernels_avx2', 'src/core/kernels_avx2.cpp',
    include_directories : incs,
    cpp_args : msvc ? ['/arch:AVX2'] : ['-mavx2'],
    gnu_symbol_visibility : 'hidden',
    pic : true
  )
elif cpu == 'aarch64'
  sources += 'src/core/kernels_neon.cpp'
endif

if host_machine.system() == 'windows'
  windows = import('windows')
  sources += windows.compile_resources('src/deflicker.rc', include_directories : incs)
endif

threads = dependency('threads')

# VapourSynth only autoloads *.dylib on macOS, while Meson defaults to *.so for modules.
module_kwargs = {}
if host_machine.system() == 'darwin'
  module_kwargs += {'name_suffix' : 'dylib'}
endif

if get_option('python_package')
  # Same layout as the VapourSynth PyPI package autoloads from: site-packages/vapoursynth/plugins
  py = import('python').find_installation(pure : false)
  install_dir = join_paths(py.get_install_dir(), 'vapoursynth', 'plugins')
else
  install_dir = join_paths(get_option('prefix'), get_option('libdir'), 'vapoursynth')
endif

shared_module('Deflicker', sources,
  include_directories : incs,
  dependencies : threads,
  link_with : simd_libs,
  install : true,
  install_dir : install_dir,
  gnu_symbol_visibility : 'hidden',
  kwargs : module_kwargs
)
