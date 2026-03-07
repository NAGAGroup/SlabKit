def main [preset: string = "default"] {
  ctest $"(path join build $preset)"
}
