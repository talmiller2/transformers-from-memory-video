"""Shared style, data, and narration helpers for every scene."""
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import *

ROOT = Path(__file__).parent
DATA = json.loads((ROOT / "data" / "curves.json").read_text())

# ---------- palette (from alexwang.ai: paper, ink, crimson accent, warm brown) ----------
BG = "#F5F3EE"        # --color-bg-paper
INK = "#2C2D30"       # --color-text-ink
SOFT = "#595A5D"      # --color-text-light
FAINT = "#8F8D88"
DIM = "#A8A49B"       # grid lines and quiet labels
LINE = "#D9D5CC"
SIG = "#B3202A"       # --color-accent: the memory we want back
NOISE = "#8A8172"     # warm brown: interference from everything else
KEY = "#2E5E8C"       # keys / key space
VAL = "#B07A12"       # values / value space
QUERY = "#3F7D4E"     # the query vector
ACCENT = "#6A4C93"
SWI = "#C76A1F"

config.background_color = BG
for cls in (VMobject, Mobject):
    cls.set_default(color=INK)

D = 64

# body text in EB Garamond, headings in Cormorant Garamond (the blog's heading face)
BODY_TEX = TexTemplate()
BODY_TEX.add_to_preamble(r"\usepackage[T1]{fontenc}\usepackage[lining]{ebgaramond}")
HEAD_TEX = TexTemplate()
HEAD_TEX.add_to_preamble(r"\usepackage[T1]{fontenc}\usepackage[lining]{CormorantGaramond}")

TEXT_SCALE = 1.2              # every label and caption is drawn this much larger than its nominal size
MIN_TEXT, MIN_MATH = 40, 40   # and never smaller than this


def T(*s, size=40, color=INK, **kw):
    """Body text set in LaTeX so inline math matches."""
    kw.setdefault("tex_template", BODY_TEX)
    return Tex(*s, font_size=max(size * TEXT_SCALE, MIN_TEXT), color=color, **kw)


def H(s, size=46, color=INK):
    """Heading in Cormorant Garamond bold."""
    return Tex(rf"\textbf{{{s}}}", font_size=size, color=color, tex_template=HEAD_TEX)


def M(*s, size=48, color=INK, **kw):
    return MathTex(*s, font_size=max(size * TEXT_SCALE, MIN_MATH), color=color, **kw)


# ---------- narration ----------
# Kokoro (local neural TTS) by default; VO_ENGINE=say falls back to the macOS voice.
ENGINE = os.environ.get("VO_ENGINE", "kokoro")
VOICE = os.environ.get("VO_VOICE", "af_heart" if ENGINE == "kokoro" else "Daniel")
SPEED = float(os.environ.get("VO_SPEED", "0.92"))
VO_DIR = ROOT / "media" / "vo"
GAP = 0.12          # pause after each narrated line
SILENT_MAX = 0.45   # cap on any animation or wait that plays with no narration

# spellings the TTS reads better than the written form (the transcript keeps the written form)
SAY_AS = {
    "SwiGLU": "swig-loo", "ReLU": "ray-loo", "YaRN": "yarn", "Qwen": "Kwen", "Allen-Zhu": "Allen Joo",
    "Su in": "Soo in", "Hebb's": "Hebb's", "arXiv": "archive",
}

_kokoro = None


SENTENCE_PAUSE = 0.85  # added silence between sentences; with the voice's own tail this is ~1s


def sentences(text):
    return [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]


def _synth(text, out):
    global _kokoro
    for a, b in SAY_AS.items():
        text = text.replace(a, b)
    if ENGINE == "kokoro":
        import soundfile as sf
        from kokoro_onnx import Kokoro
        if _kokoro is None:
            _kokoro = Kokoro(str(ROOT / "models" / "kokoro-v1.0.onnx"), str(ROOT / "models" / "voices-v1.0.bin"))
        audio, sr = _kokoro.create(text, voice=VOICE, speed=SPEED, lang="en-gb" if VOICE[0] == "b" else "en-us")
        sf.write(str(out), audio, sr)
    else:
        aiff = out.with_suffix(".aiff")
        subprocess.run(["say", "-v", VOICE, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), str(out)], check=True)
        aiff.unlink()


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def _speak(text, wav):
    """One clip per sentence, joined with SENTENCE_PAUSE of silence. Writes sentence timings beside the wav."""
    parts, cues, t = [], [], 0.0
    for i, sent in enumerate(sentences(text)):
        part = wav.with_name(f"{wav.stem}.{i}.wav")
        _synth(sent, part)
        d = _duration(part)
        if parts:
            t += SENTENCE_PAUSE
        cues.append([round(t, 3), round(t + d, 3), sent])
        t += d
        parts.append(part)
    inputs, filters = [], []
    for i, part in enumerate(parts):
        inputs += ["-i", str(part)]
        pad = f",apad=pad_dur={SENTENCE_PAUSE}" if i < len(parts) - 1 else ""
        filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo{pad}[a{i}]")
    graph = ";".join(filters) + ";" + "".join(f"[a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]", str(wav)],
                   check=True)
    for part in parts:
        part.unlink()
    wav.with_suffix(".json").write_text(json.dumps(cues))


def tts(text):
    """Synthesize `text`; return (wav path, seconds, sentence cues). Cached by content and voice."""
    VO_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{ENGINE}|{VOICE}|{SPEED}|{SENTENCE_PAUSE}|{text}".encode()).hexdigest()[:16]
    wav = VO_DIR / f"{h}.wav"
    if not wav.exists():
        _speak(text, wav)
    return wav, _duration(wav), json.loads(wav.with_suffix(".json").read_text())


class Tracker:
    def __init__(self, scene, duration):
        self.scene, self.duration, self.t0 = scene, duration, scene.renderer.time

    def left(self, floor=0.3):
        return max(floor, self.duration - (self.scene.renderer.time - self.t0))

    def frac(self, f):
        return max(0.3, f * self.duration)


class VScene(Scene):
    """Scene with a `voice` block: narration plays while the body animates, then waits out the rest.

    Outside a voice block, animations and waits are capped at SILENT_MAX so the video never sits in silence.
    """

    def setup(self):
        self._in_voice = False
        self._script = ROOT / "media" / "transcript" / f"{type(self).__name__}.txt"
        self._cues = self._script.with_suffix(".cues.json")
        self._script.parent.mkdir(parents=True, exist_ok=True)
        self._script.write_text("")
        self._cue_list = []

    def play(self, *anims, **kw):
        if not self._in_voice:
            kw["run_time"] = min(kw.get("run_time", 1.0), SILENT_MAX)
        super().play(*anims, **kw)
        self._check_frame()

    def _check_frame(self, margin=-0.1):
        """Log any text that sits partly outside the frame, so layout overflow shows up in the build log."""
        w, h = config.frame_width / 2 + margin, config.frame_height / 2 + margin
        seen = getattr(self, "_offscreen", set())
        for m in self.mobjects:
            for t in m.get_family():
                if not isinstance(t, SingleStringMathTex) or id(t) in seen:
                    continue
                glyphs = t.family_members_with_points()
                if glyphs and max(g.get_fill_opacity() for g in glyphs) > 0.05:
                    if t.get_left()[0] < -w or t.get_right()[0] > w or t.get_bottom()[1] < -h or t.get_top()[1] > h:
                        seen.add(id(t))
                        print(f"[offscreen] {type(self).__name__} t={self.renderer.time:.1f}: {t.tex_string[:50]}",
                              file=sys.stderr)
        self._offscreen = seen

    def wait(self, duration=1.0, *a, **kw):
        if not self._in_voice:
            duration = min(duration, SILENT_MAX)
        super().wait(duration, *a, **kw)

    @contextmanager
    def voice(self, text, gap=GAP):
        wav, dur, cues = tts(text)
        self.add_sound(str(wav))
        tr = Tracker(self, dur)
        with self._script.open("a") as f:
            f.write(text + "\n\n")
        self._cue_list += [[tr.t0 + a, tr.t0 + b, sent] for a, b, sent in cues]
        self._cues.write_text(json.dumps(self._cue_list))
        self._in_voice = True
        yield tr
        rest = dur - (self.renderer.time - tr.t0)
        if rest > 0.02:
            self.wait(rest)
        elif rest < -0.4:
            print(f"[overrun {-rest:.1f}s] {type(self).__name__}: {text[:60]}", file=sys.stderr)
        if gap:
            self.wait(gap)
        self._in_voice = False

    def clear_all(self, run_time=0.4):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)


# ---------- shared visuals ----------
def keep_in_frame(m, margin=0.3):
    """Shift `m` just enough that it sits inside the frame with `margin` to spare."""
    w, h = config.frame_width / 2 - margin, config.frame_height / 2 - margin
    dx = max(0, -w - m.get_left()[0]) - max(0, m.get_right()[0] - w)
    dy = max(0, -h - m.get_bottom()[1]) - max(0, m.get_top()[1] - h)
    return m.shift(RIGHT * dx + UP * dy)


rng_global = np.random.default_rng(7)


def unit_vectors(n, d=D, seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def similarity_sample(n, seed=0):
    """Similarities between one target key and n-1 other random keys in d=64."""
    k = unit_vectors(n, seed=seed)
    s = k[1:] @ k[0]
    return np.concatenate([[1.0], s])


SCORE_FNS = {
    "linear": lambda s: s,
    "relu": lambda s: np.maximum(s, 0),
    "relu2": lambda s: np.maximum(s, 0) ** 2,
    "relu3": lambda s: np.maximum(s, 0) ** 3,
}


def softmax_w(s, beta):
    e = np.exp(beta * (s - s.max()))
    return e / e.max()


def bar_chart(weights, width=11, height=3.0, target_index=0, bar_color=NOISE):
    """Bars for each memory's recall weight, relative to the target (target = height)."""
    n = len(weights)
    bw = width / n
    g = VGroup()
    for i, w in enumerate(weights):
        h = float(w) * height
        bar = Rectangle(
            width=bw * 0.78, height=max(abs(h), 0.002), stroke_width=0,
            fill_color=SIG if i == target_index else bar_color, fill_opacity=1,
        )
        bar.move_to([(-width / 2 + bw * (i + 0.5)), h / 2, 0])
        g.add(bar)
    return g


def ladder(width=3.2, height=6.0):
    """Vertical log axis from 10 to 10^7 used to track capacity across the video."""
    ax = NumberLine(
        x_range=[1, 7, 1], length=height, rotation=PI / 2, include_numbers=False,
        stroke_width=2, color=DIM, include_tip=False,
    )
    labels = VGroup(*[
        M(rf"10^{k}", size=30, color=FAINT).next_to(ax.n2p(k), LEFT, buff=0.15) for k in range(1, 8)
    ])
    title = T("capacity at $d_k=64$", size=32, color=SOFT).next_to(ax, UP, buff=0.3)
    return VGroup(ax, labels, title)


def ladder_rung(lad, value, label, color):
    ax = lad[0]
    p = ax.n2p(np.log10(value))
    dot = Dot(p, color=color, radius=0.07)
    txt = T(label, size=34, color=color).next_to(dot, RIGHT, buff=0.2)
    return VGroup(dot, txt)


def loglog_axes(x_max_exp=6.5, y_min_exp=-3, y_max_exp=1.5, w=9, h=5.2):
    ax = Axes(
        x_range=[0.5, x_max_exp, 1], y_range=[y_min_exp, y_max_exp, 1],
        x_length=w, y_length=h, tips=False,
        axis_config={"color": DIM, "stroke_width": 2, "include_ticks": True},
    )
    xl = VGroup(*[
        M(rf"10^{k}", size=24, color=FAINT).next_to(ax.c2p(k, y_min_exp), DOWN, buff=0.15)
        for k in range(1, int(x_max_exp) + 1)
    ])
    yl = VGroup(*[
        M(rf"10^{{{k}}}", size=24, color=FAINT).next_to(ax.c2p(0.5, k), LEFT, buff=0.15)
        for k in range(int(y_min_exp), int(y_max_exp) + 1)
    ])
    one = DashedLine(ax.c2p(0.5, 0), ax.c2p(x_max_exp, 0), color=INK, stroke_width=1.5, dash_length=0.08)
    xt = T("number of memories $M$", size=28, color=SOFT).next_to(xl, DOWN, buff=0.2)
    yt = T(r"$\|\text{interference}\| / \|\text{signal}\|$", size=28, color=SOFT).rotate(PI / 2)
    yt.next_to(yl, LEFT, buff=0.2)
    return VGroup(ax, xl, yl, one, xt, yt)


def loglog_curve(ax, xs, ys, color, y_min_exp=-3, stroke_width=4):
    x_lo, x_hi = ax.x_range[0], ax.x_range[1]
    y_hi = ax.y_range[1]
    pts = [ax.c2p(np.log10(m), np.log10(y)) for m, y in zip(xs, ys)
           if y > 10 ** y_min_exp and x_lo <= np.log10(m) <= x_hi and np.log10(y) <= y_hi]
    c = VMobject(color=color, stroke_width=stroke_width)
    return c.set_points_smoothly(pts) if len(pts) > 3 else c.set_points_as_corners(pts)


def curve_on(ax, key, color, source="analytic", y_min_exp=-3, stroke_width=4):
    grid = np.array(DATA["grid"], dtype=float)
    ys = DATA["long_context"][key] if source == "long_context" else DATA[source][key]
    xs = np.logspace(0.5, 6.4, 300)
    ys = np.exp(np.interp(np.log(xs), np.log(grid), np.log(np.maximum(ys, 1e-30))))
    return loglog_curve(ax, xs, ys, color, y_min_exp, stroke_width)


def mc_dots(ax, key, color, y_min_exp=-3):
    g = VGroup()
    for m, y in DATA["mc"][key].items():
        if y > 10 ** y_min_exp and np.log10(float(m)) <= ax.x_range[1]:
            g.add(Dot(ax.c2p(np.log10(float(m)), np.log10(y)), radius=0.05, color=color))
    return g
