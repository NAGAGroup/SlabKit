# The build directory is named for the pixi ENVIRONMENT, not the preset: every
# dev-* environment uses the `dev` preset but a different toolchain, and one
# shared directory would hand the next environment a stale CMake cache.
def main [preset: string = "default"] {
  let dir = (["build" ($env.PIXI_ENVIRONMENT_NAME? | default $preset)] | path join)
  cmake -S . -B $dir --preset $preset
}
