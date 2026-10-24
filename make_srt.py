"""Build YouTube subtitles (SRT) from the per-sentence timings each scene logs while rendering.

usage: make_srt.py media/concat.txt out.srt
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent
MAX_LINE, MAX_LINES = 42, 2
# narration is written for the ear; subtitles show the written form
WRITTEN = {"alex wang dot A I": "alexwang.ai", "Qwen two point five one M": "Qwen2.5-1M", "twenty twenty one": "2021"}


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def chunks(sentence):
    """Split a sentence into pieces that fit in two subtitle lines, preferring breaks after commas."""
    limit = MAX_LINE * MAX_LINES
    out, cur = [], ""
    for word in sentence.split():
        nxt = f"{cur} {word}".strip()
        if len(nxt) > limit or (len(cur) > limit * 0.55 and cur.endswith((",", ":", ";"))):
            out.append(cur)
            nxt = word
        cur = nxt
    out.append(cur)
    return out


def stamp(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def main(concat, out):
    entries, offset = [], 0.0
    for line in Path(concat).read_text().splitlines():
        movie = line.split("'")[1]
        scene = Path(movie).stem.removesuffix("_av")
        cues = json.loads((ROOT / "media" / "transcript" / f"{scene}.cues.json").read_text())
        for start, end, sentence in cues:
            for spoken, written in WRITTEN.items():
                sentence = sentence.replace(spoken, written)
            pieces = chunks(sentence)
            total = sum(len(p) for p in pieces)
            t = start
            for p in pieces:
                d = (end - start) * len(p) / total
                entries.append((offset + t, offset + t + d, "\n".join(textwrap.wrap(p, MAX_LINE))))
                t += d
        offset += duration(movie)
    with open(out, "w") as f:
        for i, (a, b, text) in enumerate(entries, 1):
            f.write(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n")
    print(f"{len(entries)} subtitles -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
