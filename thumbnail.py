"""YouTube thumbnail (1280x720): render with
    manim -s -r 1280,720 --disable_caching thumbnail.py Thumbnail
"""
from common import *


class Thumbnail(Scene):
    def construct(self):
        title = VGroup(
            H(r"You could have invented", size=78),
            H(r"transformers", size=150, color=SIG),
            H(r"from memory", size=112),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).to_corner(UL, buff=0.55).shift(DOWN * 0.15)

        # the video's signature picture: the memory you asked for, towering over the interference
        sims = similarity_sample(26, seed=11)
        w = np.exp(8 * (sims - 1))
        w[1:] = sims[1:] * 1.6
        bars = bar_chart(w, width=6.2, height=1.9)
        base = Line(bars.get_left() + LEFT * 0.1, bars.get_right() + RIGHT * 0.1, color=DIM, stroke_width=2)
        base.set_y(0)
        chart = VGroup(base, bars).next_to(title, DOWN, buff=0.55, aligned_edge=LEFT).shift(RIGHT * 0.05)
        cap = T(r"from 64 to 1{,}000{,}000+ memories", size=46, color=SOFT)
        cap.next_to(chart, DOWN, buff=0.35, aligned_edge=LEFT)

        kea = ImageMobject(str(ROOT / "assets" / "kea.png")).set_height(8.2).to_edge(RIGHT, buff=0).shift(RIGHT * 1.9 + UP * 0.1)
        self.add(kea, title, chart, cap)
