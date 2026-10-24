from common import *


class Outro(VScene):
    def construct(self):
        lad = ladder(height=6.0).shift(LEFT * 2.2 + DOWN * 0.1)
        rungs = [
            (64, r"outer product: 64", KEY),
            (128, r"optimal linear \;/\; ReLU MLP: 128", QUERY),
            (2800, r"ReLU$^2$ \;/\; SwiGLU MLP: $\approx$2,700--2,900", VAL),
            (34953, r"ReLU$^3$ MLP: $\approx$35,000", ACCENT),
            (1.2e6, r"softmax attention: $\approx$1.2 million", SIG),
        ]
        rs = [ladder_rung(lad, v, l, c) for v, l, c in rungs]
        rs[0][1].shift(DOWN * 0.12)
        rs[1][1].shift(UP * 0.12)
        extra = T(r"($>$2 million with $\beta \propto \log M$)", size=24, color=SIG).next_to(rs[-1][1], DOWN, buff=0.1,
                                                                                           aligned_edge=LEFT)
        with self.voice(
            "So here's the whole ladder. A linear map, writing by outer products, holds about d memories. "
            "Writing carefully, with least squares, doubles that."
        ) as tr:
            self.play(FadeIn(lad))
            for r in rs[:2]:
                self.play(FadeIn(r[0], scale=2), Write(r[1]), run_time=tr.frac(0.18))
        with self.voice(
            "Being picky at read time, by passing similarities through a nonlinearity, turns d into d squared, or d "
            "cubed. That's the MLP."
        ) as tr:
            for r in rs[2:4]:
                self.play(FadeIn(r[0], scale=2), Write(r[1]), run_time=tr.frac(0.3))
        with self.voice(
            "And taking every power at once, through the exponential, gives softmax attention. Its capacity grows "
            "exponentially in the sharpness, and its memories are the context itself."
        ) as tr:
            self.play(FadeIn(rs[4][0], scale=2), Write(rs[4][1]), run_time=tr.frac(0.3))
            self.play(FadeIn(extra))
        self.clear_all()

        cav = VGroup(
            T(r"Caveat: these are counts for \emph{random} keys and values.", size=34),
            T(r"Trained models learn structured knowledge, measured in bits per parameter:", size=28, color=SOFT),
            T(r"$\approx 2$ bits/param (Allen-Zhu \& Li) \qquad $\approx 3.6$ bits/param (Morris et al.)", size=28, color=SOFT),
        ).arrange(DOWN, buff=0.35)
        with self.voice(
            "One caveat. All of these numbers count random keys and values. Real models learn structured knowledge "
            "from data, and measured capacity is better quoted in bits per parameter: about two in Allen-Zhu and Li's "
            "experiments, and about three point six in Morris and colleagues'."
        ):
            self.play(FadeIn(cav[0]))
            self.play(FadeIn(cav[1]))
            self.play(FadeIn(cav[2]))
        with self.voice(
            "Still, the picture is a useful one. Linear layers, MLPs and attention are all the same kind of thing: "
            "a store of keys and values, read out by similarity. What changes is how carefully they write, "
            "and how selectively they recall.",
            gap=0.15,
        ):
            self.play(FadeOut(cav))
            one = VGroup(
                M(r"f(x) = \sum_{i} v_i\; \sigma(k_i^\top x)", size=60),
                T(r"write carefully \;$\cdot$\; recall selectively", size=34, color=SOFT),
            ).arrange(DOWN, buff=0.5)
            self.play(Write(one[0]), run_time=2)
            self.play(FadeIn(one[1]))
        self.clear_all()

        # ---- credits ----
        credits = VGroup(
            T(r"Based on", size=28, color=SOFT),
            T(r"\emph{You could have invented transformers from memory}", size=40),
            T(r"by Alex Wang \;$\cdot$\; October 2026", size=30),
            T(r"\texttt{alexwang.ai/posts/associative-memory}", size=30, color=KEY),
            T(r"The ideas, derivations and capacity results are from the original post.", size=26, color=SOFT),
            T(r"This is an independent adaptation:", size=26, color=SOFT),
            T(r"not made by, endorsed by, or affiliated with Alex Wang.", size=26, color=SOFT),
            T(r"Any errors are the adapter's own.", size=26, color=SOFT),
            T(r"Animation: Manim Community Edition \;$\cdot$\; Narration: synthesized voice", size=22, color=FAINT),
            T(r"Simulations: NumPy Monte Carlo at $d_k=64$", size=22, color=FAINT),
        ).arrange(DOWN, buff=0.22)
        credits[4:8].shift(DOWN * 0.3)
        credits[8:].shift(DOWN * 0.6)
        with self.voice(
            "This video is adapted from Alex Wang's post, You could have invented transformers from memory. "
            "It's an independent adaptation, not made by or affiliated with Alex Wang. The original goes deeper, "
            "with full derivations and a reading list on Hopfield networks, fast weights, retrieval and knowledge "
            "editing. It's well worth reading at alex wang dot A I.",
            gap=2.0,
        ):
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in credits], lag_ratio=0.15), run_time=3)
        self.play(FadeOut(credits), run_time=1.5)
