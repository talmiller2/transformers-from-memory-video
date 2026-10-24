# You could have invented transformers from memory: the video

Source code for an animated explainer video adapted from Alex Wang's post
[*You could have invented transformers from memory*](https://alexwang.ai/posts/associative-memory/).

The ideas, derivations and capacity results all come from that post. This video and repository are an
independent adaptation: they are not made by, endorsed by, or affiliated with Alex Wang.

The video walks through the post's argument that linear maps, MLPs and self-attention are all associative
memory stores. It shows how each one's capacity climbs, at 64-dimensional keys, from about 64 memories (outer
product) to 128 (least squares), about 2,700 (squared-ReLU MLP) and over a million (softmax attention).

## What's here

| File | What it does |
|---|---|
| `sim.py` | Monte Carlo check of every interference-to-signal ratio at d_k = 64. Writes `data/curves.json`, which the plots read. |
| `common.py` | Palette (taken from alexwang.ai), fonts, shared visuals, and the narration helper. |
| `s0_intro.py` … `s7_outro.py` | One Manim scene per section of the video. |
| `make_srt.py` | Builds YouTube subtitles from the per-sentence timings each scene logs while rendering. |
| `build.sh` | Renders every scene, joins them with narration, and writes the video, subtitles and narration script to `out/`. |
| `fetch_assets.sh` | Downloads the voice model and the title-card painting (not redistributed here). |
| `frames.sh` | Contact sheet of frames from a rendered scene, for checking layout. |
| `thumbnail.py` | The YouTube thumbnail: `manim -s -r 1280,720 --disable_caching thumbnail.py Thumbnail`. |

## How the narration works

Each scene narrates through `with self.voice("..."):` blocks. The text is synthesized locally with
[Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (voice `af_heart`), one sentence at a time with a pause
between sentences. Each block's animations stretch to fit its audio. The sentence timings are logged as the scene
renders, which is where the subtitles come from.

To change the voice, set `VO_VOICE` (any Kokoro voice) and optionally `VO_SPEED`. `VO_ENGINE=say` uses the
macOS system voice instead.

## Building it

Requirements: Python 3.11, ffmpeg, and a LaTeX install with the `ebgaramond` and `CormorantGaramond` packages
(both are in TeX Live). Developed on macOS.

```bash
pip install -r requirements.txt
./fetch_assets.sh
python sim.py          # optional: data/curves.json is already included
./build.sh             # 480p15 draft
./build.sh final       # 1080p60 for upload
```

Rendering uses `--disable_caching`, because Manim's cache can keep a stale audio track when only narration
changes.

## Credits

- Original post and all of its ideas: Alex Wang, [alexwang.ai](https://alexwang.ai/posts/associative-memory/).
  The kea painting on the title card is from the same site and is downloaded at build time.
- Animation: [Manim Community Edition](https://www.manim.community/).
- Narration: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via kokoro-onnx.
