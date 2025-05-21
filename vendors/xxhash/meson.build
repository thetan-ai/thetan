# meson.build for xxhash library

xxhash_sources = [
  'xxhash.c',
]

xxhash_headers = [
  'xxhash.h',
]

xxhash_inc = include_directories('.')

xxhash_lib = static_library(
  'libxxhash',
  xxhash_sources,
  include_directories: xxhash_inc,
  install: false,
)

# This dependency can be used by other parts of the project
xxhash_dep = declare_dependency(
  link_with: xxhash_lib,
  include_directories: xxhash_inc,
)