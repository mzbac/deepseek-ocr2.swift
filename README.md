# DeepSeek OCR 2 (Swift)

Swift/MLX port of `deepseek-ai/DeepSeek-OCR-2`.

## Install

Download the latest release from GitHub Releases:

- [Latest release](https://github.com/mzbac/deepseek-ocr2.swift/releases/latest)

Extract the archive and put `DeepSeekOCR2CLI` somewhere on your `PATH`.
If there isn’t a release for your platform yet, build from source below.

## CLI usage

Build the CLI:

```bash
xcodebuild -scheme DeepSeekOCR2CLI -destination "platform=macOS" -configuration Release -derivedDataPath build/DerivedData build
```

By default, the CLI uses `--model deepseek-ai/DeepSeek-OCR-2` (downloaded and cached automatically).

Make the built binary available on your `PATH` (for this shell):

```bash
export PATH="$PWD/build/DerivedData/Build/Products/Release:$PATH"
```

Run OCR:

```bash
DeepSeekOCR2CLI --image /path/to/image.png
```

Override the GPU cache limit (default: 2048MB):

```bash
DeepSeekOCR2CLI --cache-limit 1024 --image /path/to/image.png
```

Run OCR on multiple images:

```bash
DeepSeekOCR2CLI --image a.png --image b.png
```

Use a local model directory (instead of downloading from Hugging Face):

```bash
DeepSeekOCR2CLI --model /path/to/DeepSeek-OCR-2 --image /path/to/image.png
```

Customize the prompt (`<image>` is the placeholder for image tokens):

```bash
DeepSeekOCR2CLI --image /path/to/image.png --prompt $'<image>\\nFree OCR.'
```

## Quantization

Quantize into a local directory (4-bit or 8-bit):

```bash
DeepSeekOCR2CLI quantize --output-dir temp/DeepSeek-OCR-2-q4 --bits 4 --group-size 64 --mode affine
```

Run OCR with quantized weights:

```bash
DeepSeekOCR2CLI --model temp/DeepSeek-OCR-2-q4 --image /path/to/image.png
```

## Help

```bash
DeepSeekOCR2CLI --help
```

## License

Apache-2.0. See `LICENSE`.
