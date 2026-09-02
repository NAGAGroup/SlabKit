def main [preset: string = "default"] {
  let dir = (["build" ($env.PIXI_ENVIRONMENT_NAME? | default $preset)] | path join)
  cmake --build $dir
}
