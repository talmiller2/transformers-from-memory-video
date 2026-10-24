#!/bin/zsh
# Download the files the video needs but this repo doesn't redistribute:
#   - Kokoro TTS model + voices (Apache-2.0, from the kokoro-onnx release)
#   - the kea painting from alexwang.ai, shown on the title card (Alex Wang's artwork)
set -e
cd "${0:A:h}"
mkdir -p models assets
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [[ -f models/$f ]] || curl -fsSL -o models/$f https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f
done
if [[ ! -f assets/kea.png ]]; then
  curl -fsSL -o assets/kea.webp https://alexwang.ai/images/kea/painting.webp
  ffmpeg -y -loglevel error -i assets/kea.webp assets/kea.png
fi
echo "assets ready"
