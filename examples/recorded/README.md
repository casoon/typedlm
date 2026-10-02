Output of `cargo run --example <name>` from `crates/typedlm/examples/`, recorded on
2026-10-02 against a local Ollama 0.35 with `qwen3:32b`. Model answers differ between
runs and models; these files are shown as recorded, not regenerated at build time.

- `<name>.txt` — the recorded output; the project site's showcase reads it.
- `<name>.terminal.yaml` — a [castwright](https://github.com/casoon/castwright) demo
  that types the command and plays the recorded output.
- `<name>.svg` — the demo as animated SVG (no script, honours
  `prefers-reduced-motion`), used in the README and on the site's start page.

After changing a `.txt` or `.terminal.yaml`, rebuild the SVGs:

```sh
cd examples/recorded
for f in *.terminal.yaml; do
  npx -y -p @casoon/castwright@0.4.0 -p shiki castwright build "$f" --format svg -o .
done
```
