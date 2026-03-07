def main [preset: string = "default"] {
  cmake -S . -B $"(path join build $preset)" --preset $preset
}
