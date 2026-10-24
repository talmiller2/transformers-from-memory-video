#!/bin/zsh
# usage: frames.sh video.mp4 outdir N  -> N evenly spaced frames, tiled into one contact sheet
v=$1; out=$2; n=${3:-12}
[[ -f $v ]] || { echo missing $v; exit 1; }
mkdir -p $out
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $v)
fps=$(python3 -c "print($n/$dur)")
ffmpeg -y -loglevel error -i $v -vf "fps=$fps,scale=640:-1,tile=3x$(( (n+2)/3 ))" -frames:v 1 $out/sheet.png
echo "$dur s -> $out/sheet.png"
