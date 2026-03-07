def main [preset: string = "default"] {
  cmake -S . -B $"(echo build | path join $preset)" --preset $preset
}
