def main [preset: string = "default"] {
  let dir = (["build" ($env.PIXI_ENVIRONMENT_NAME? | default $preset)] | path join)
  ctest --test-dir $dir --output-on-failure
}
