#!/bin/zsh
# Render every scene and join them into one video with narration, subtitles and a narration script in out/.
#   ./build.sh          -> 480p15 draft
#   ./build.sh final    -> 1080p60 for upload (slow)
set -e
cd "${0:A:h}"
if [[ "$1" == final ]]; then
  FLAGS=(--resolution 1920,1080 --fps 60 --disable_caching); RES=1080p60; NAME=transformers_from_memory
else
  FLAGS=(-ql --disable_caching); RES=480p15; NAME=transformers_from_memory_draft
fi
# use a project-local env if there is one (../.env), else whatever is on PATH
[[ -d ../.env/bin ]] && export PATH="$PWD/../.env/bin:$PATH"
mkdir -p out media
SCENES=(s0_intro:Intro s1_outer:OuterProduct s2_interference:Interference s3_optimal:OptimalLinear
        s4_mlp:MLPStore s5_attention:Attention s6_longctx:LongContext s7_outro:Outro)
TITLES=("Intro" "1. The outer-product memory store" "2. Recall picks up interference"
        "3. The optimal linear memory store" "4. The MLP as a nonlinear memory store"
        "5. Self-attention as a non-parametric memory store" "6. Long context" "Outro and credits")
: > media/concat.txt
for pair in $SCENES; do
  f=${pair%%:*}; c=${pair##*:}
  echo "== rendering $f"
  manim $FLAGS $f.py $c > media/log_$f.txt 2>&1 || { tail -20 media/log_$f.txt; exit 1; }
  grep -a -E "overrun|offscreen" media/log_$f.txt || true
  d=media/videos/$f/$RES
  # manim leaves the narration in a sibling .wav; mux it in (pad audio so every segment has full-length sound)
  ffmpeg -y -loglevel error -i $d/$c.mp4 -i $d/$c.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -ar 48000 \
    -af apad -shortest $d/${c}_av.mp4
  echo "file '$PWD/$d/${c}_av.mp4'" >> media/concat.txt
done
OUT=out/$NAME.mp4
ffmpeg -y -loglevel error -f concat -safe 0 -i media/concat.txt \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart $OUT
python3 make_srt.py media/concat.txt out/$NAME.srt
{
  echo "# You could have invented transformers from memory: narration script"
  echo
  echo "Adapted from *You could have invented transformers from memory* by Alex Wang"
  echo "(https://alexwang.ai/posts/associative-memory/). Independent adaptation:"
  echo "not made by, endorsed by, or affiliated with Alex Wang."
  i=1
  for pair in $SCENES; do
    c=${pair##*:}
    echo; echo "## ${TITLES[$i]}"; echo
    cat media/transcript/$c.txt
    i=$((i+1))
  done
} > out/script.md
ffprobe -v error -show_entries format=duration -of csv=p=0 $OUT
echo "silent stretches over 1.2s:"
ffmpeg -hide_banner -i $OUT -map 0:a -af silencedetect=noise=-40dB:d=1.2 -f null - 2>&1 | grep -a silence_end | awk '{print "  ends " $5 "s, lasted " $8 "s"}'
