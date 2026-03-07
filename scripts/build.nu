def main [preset: string = "default"] {
  cmake --build $"(path join build $preset)"
}
