import math

from common import *

BETA = 8.0


def trunc(n):
    return lambda s: sum((BETA * s) ** p / math.factorial(p) for p in range(n))


class Attention(VScene):
    def construct(self):
        header = H(r"5.\; Self-attention as a non-parametric memory store", size=48).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        series = M(r"e^{\beta s}", r"=", r"1", r"+ \beta s", r"+ \frac{(\beta s)^2}{2!}", r"+ \frac{(\beta s)^3}{3!}",
                   r"+ \cdots", size=50).next_to(header, DOWN, buff=0.4)
        with self.voice(
            "Every extra power helped. So why not use all of them at once? The exponential function is exactly that: "
            "a weighted sum of every power of s."
        ) as tr:
            self.play(Write(series), run_time=2.5)

        ax = Axes(x_range=[-0.4, 1.0, 0.2], y_range=[-0.3, 1.0, 0.5], x_length=6.2, y_length=4.2, tips=False,
                  axis_config={"color": DIM, "font_size": 22},
                  x_axis_config={"numbers_to_include": [0, 0.5, 1]},
                  y_axis_config={"numbers_to_include": [0.5, 1]}).shift(LEFT * 2.6 + DOWN * 1.3)
        yl = T(r"weight $\div$ target's weight", size=24, color=SOFT).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)
        sl = M("s", size=30, color=SOFT).next_to(ax.x_axis, RIGHT, buff=0.1)
        specs = [(2, KEY, "40", 2), (3, QUERY, "450", 3), (4, VAL, r"3{,}100", 4)]
        legend = VGroup()
        curves = VGroup()
        for n, col, cap, terms in specs:
            f = trunc(n)
            c = ax.plot(lambda s, f=f: f(s) / f(1.0), x_range=[-0.4, 1.0, 0.005], color=col, stroke_width=4)
            curves.add(c)
            legend.add(T(rf"{terms} terms: capacity $\approx {cap}$", size=30, color=col))
        cexp = ax.plot(lambda s: math.exp(BETA * (s - 1)), x_range=[-0.4, 1.0, 0.005], color=SIG, stroke_width=5)
        legend.add(T(r"all terms: capacity $\approx 1.2$ million", size=30, color=SIG))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(RIGHT * 3.6 + DOWN * 1.0)
        with self.voice(
            "Here's the weight each truncated series gives to a memory with similarity s, relative to the target. "
            "Keep two terms, and the capacity is only about forty. Three terms, about four hundred fifty. "
            "Four terms, about thirty one hundred. And the full exponential presses stray similarities so flat that "
            "its capacity runs past a million."
        ) as tr:
            self.play(Create(ax), FadeIn(yl), FadeIn(sl), run_time=1)
            for i in range(3):
                self.play(Create(curves[i]), FadeIn(legend[i], shift=LEFT * 0.2), run_time=tr.frac(0.14))
                self.wait(tr.frac(0.04))
            self.play(Create(cexp), FadeIn(legend[3], shift=LEFT * 0.2), run_time=tr.frac(0.14))
        self.play(FadeOut(VGroup(ax, yl, sl, curves, cexp, legend, series)))

        # ---- normalize: softmax ----
        raw = M(r"f(x) = \sum_{i=1}^{M} v_i\, e^{\beta\,k_i^\top x}", size=50).shift(UP * 1.4)
        soft = M(r"f(x)", r"=", r"\sum_{i=1}^{M} v_i\,\frac{e^{\beta\,k_i^\top x}}{\sum_j e^{\beta\,k_j^\top x}}",
                 r"=", r"V^\top \mathrm{softmax}(\beta K x)", size=50).shift(UP * 1.4)
        soft[4].set_color(SIG)
        why = VGroup(
            T(r"dividing every weight by the same total keeps their ratios", size=30, color=SOFT),
            T(r"$\Rightarrow$ same interference-to-signal ratio, but numerically stable", size=30, color=SOFT),
        ).arrange(DOWN, buff=0.2).shift(DOWN * 0.6)
        with self.voice(
            "Use the exponential as our sigma. Then divide every weight by their total."
        ):
            self.play(Write(raw), run_time=2)
        with self.voice(
            "Dividing by a shared constant doesn't change any ratios, so the interference ratio is untouched. But it "
            "keeps the output on a sane scale, and the arithmetic stable. And what we've written down is softmax attention."
        ) as tr:
            self.play(TransformMatchingShapes(raw, soft[:3]), run_time=1.5)
            self.play(FadeIn(why[0]))
            self.play(FadeIn(why[1]))
            self.play(Write(soft[3:]), run_time=1.5)
            self.play(Circumscribe(soft[4], color=SIG))
        self.play(FadeOut(why), soft.animate.scale(0.75).next_to(header, DOWN, buff=0.3))

        # ---- the memories are the context ----
        words = ["The", "rules", "say", "each", "player", "draws", "four", "tiles"]
        toks = VGroup(*[
            VGroup(RoundedRectangle(width=1.25, height=0.6, corner_radius=0.1, color=SOFT, stroke_width=2),
                   T(w, size=28)) for w in words
        ]).arrange(RIGHT, buff=0.18).shift(DOWN * 0.2)
        for t in toks:
            t[1].move_to(t[0])
        ks = VGroup(*[Square(0.32, color=KEY, fill_opacity=0.7).next_to(t, DOWN, buff=0.35).shift(LEFT * 0.22) for t in toks])
        vs = VGroup(*[Square(0.32, color=VAL, fill_opacity=0.7).next_to(t, DOWN, buff=0.35).shift(RIGHT * 0.22) for t in toks])
        kvl = VGroup(M("k_i", size=30, color=KEY), M("v_i", size=30, color=VAL)).arrange(RIGHT, buff=0.4)
        kvl.next_to(VGroup(ks, vs), DOWN, buff=0.3)
        qd = Square(0.36, color=QUERY, fill_opacity=0.8).next_to(toks[-1], RIGHT, buff=0.55)
        qa = Arrow(toks[-1].get_right(), qd.get_left(), buff=0.08, color=QUERY, stroke_width=3)
        ql = M("x", size=32, color=QUERY).next_to(qd, UP, buff=0.12)
        r = np.random.default_rng(5)
        att = np.array([0.04, 0.55, 0.06, 0.04, 0.08, 0.1, 0.05, 0.08])
        bars = VGroup(*[Rectangle(width=0.6, height=0.05 + 2.0 * a, stroke_width=0,
                                  fill_color=SIG if a == att.max() else NOISE, fill_opacity=1)
                        .next_to(t, UP, buff=0.12) for t, a in zip(toks, att)])
        with self.voice(
            "Notice where the memories come from now. In an MLP, keys and values are learned weights, frozen after "
            "training. In attention, every token in the context writes its own key and value, on the fly."
        ) as tr:
            self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in toks], lag_ratio=0.1), run_time=1.5)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(k, shift=DOWN * 0.2), FadeIn(v, shift=DOWN * 0.2))
                                    for k, v in zip(ks, vs)], lag_ratio=0.12), FadeIn(kvl), run_time=2)
        with self.voice(
            "The newest token makes a query, and recall is a similarity weighted mix of every value in the context. "
            "The memory store is the context itself. It's non parametric: the more you read, the more it holds."
        ) as tr:
            self.play(GrowArrow(qa), FadeIn(qd), Write(ql))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08), run_time=1.5)
        self.play(FadeOut(VGroup(toks, ks, vs, kvl, qd, qa, ql, bars)))

        # ---- capacity ----
        capeq = M(r"\frac{\|\text{interference}\|}{\|\text{signal}\|}", r"\approx", r"\sqrt{M}", r"\; e^{-\beta}",
                  r"\; e^{\beta^2/d_k}", size=66).shift(DOWN * 0.1)
        capeq[2].set_color(NOISE)
        capeq[3].set_color(SIG)
        capeq[4].set_color(KEY)
        l_m = keep_in_frame(T("random walk, as before", size=32, color=NOISE).next_to(capeq[2], UP, buff=0.9).shift(LEFT * 2.2))
        l_w = VGroup(T(r"the win: target scores $e^{\beta}$,", size=32, color=SIG),
                     T(r"a typical stray about $e^{0} = 1$", size=32, color=SIG)).arrange(DOWN, buff=0.1)
        keep_in_frame(l_w.next_to(capeq[3], DOWN, buff=0.9))
        l_p = VGroup(T("the penalty: lucky strays that", size=32, color=KEY),
                     T("happen to match well get amplified", size=32, color=KEY)).arrange(DOWN, buff=0.1)
        keep_in_frame(l_p.next_to(capeq[4], UP, buff=0.9).shift(RIGHT * 1.8))
        arr = VGroup(
            Arrow(l_m.get_bottom(), capeq[2].get_top(), buff=0.1, color=NOISE, stroke_width=3),
            Arrow(l_w.get_top(), capeq[3].get_bottom(), buff=0.1, color=SIG, stroke_width=3),
            Arrow(l_p.get_bottom(), capeq[4].get_top(), buff=0.1, color=KEY, stroke_width=3),
        )
        plug = M(r"\beta = \sqrt{d_k} = 8:\quad \sqrt{M}\, e^{-8}\, e^{1} = 1 \;\Rightarrow\; M = e^{14} \approx 1.2\text{ million}",
                 size=38, color=SIG).to_edge(DOWN, buff=0.35)
        with self.voice(
            "Same question as before: how big is the noise, compared to the signal? For softmax, the answer has "
            "three pieces."
        ):
            self.play(FadeOut(soft), run_time=0.5)
            self.play(Write(capeq), run_time=2.5)
        with self.voice("Root M is the familiar random walk. More memories, more noise."):
            self.play(FadeIn(l_m), GrowArrow(arr[0]))
        with self.voice(
            "E to the minus beta is the win. The target scores e to the beta, while a typical stray, with "
            "similarity near zero, scores about e to the zero, which is one. So the target beats each stray by a "
            "factor of e to the beta."
        ) as tr:
            self.play(FadeIn(l_w), GrowArrow(arr[1]))
        with self.voice(
            "And e to the beta squared over d is a penalty. Now and then, a stray key happens to line up fairly "
            "well, and the exponential amplifies those lucky outliers."
        ):
            self.play(FadeIn(l_p), GrowArrow(arr[2]))
        with self.voice(
            "With the standard choice, beta equals root d, which is eight. The win is e to the minus eight, the "
            "penalty is only e to the one, so the ratio is root M times e to the minus seven. That reaches one when "
            "M is e to the fourteenth: about one point two million memories, from the same sixty four dimensions."
        ) as tr:
            self.wait(tr.frac(0.1))
            self.play(Write(plug), run_time=3)
        self.play(FadeOut(VGroup(capeq, l_m, l_w, l_p, arr, plug)), FadeIn(soft))

        # ---- beta demo ----
        Mn = 150
        sims = similarity_sample(Mn, seed=31)
        beta = ValueTracker(8.0)
        base_y = -2.3
        base = Line(LEFT * 6, RIGHT * 6, color=DIM, stroke_width=1.5).shift(UP * base_y)

        def chart():
            w = np.exp(beta.get_value() * (sims - 1))
            return bar_chart(w, width=11.5, height=3.6).shift(UP * base_y)

        def readout():
            w = np.exp(beta.get_value() * (sims - 1))
            ratio = np.sqrt(np.sum(w[1:] ** 2))
            col = SIG if ratio > 1 else QUERY
            return VGroup(M(rf"\beta = {beta.get_value():.1f}", size=40),
                          M(rf"M = {Mn}", size=34, color=SOFT),
                          M(rf"\text{{interference}}/\text{{signal}} \approx {ratio:.2f}", size=34, color=col)) \
                .arrange(DOWN, aligned_edge=RIGHT).to_corner(UR, buff=0.4).shift(DOWN * 1.6)

        ch, ro = always_redraw(chart), always_redraw(readout)
        with self.voice(
            "The sharpness beta matters, though. Here are a hundred and fifty memories under softmax, at beta equals eight. "
            "The target towers over everything else."
        ):
            self.play(Create(base), FadeIn(ch), FadeIn(ro))
        with self.voice(
            "Turn beta down to two, and the grey bars rise to meet it. With so soft a softmax, recall fails after "
            "just a few dozen memories."
        ) as tr:
            self.play(beta.animate.set_value(2.0), run_time=tr.left(1) - 0.5, rate_func=smooth)
        with self.voice("Sharpness and dimension, together, set the capacity.", gap=0.15):
            self.play(beta.animate.set_value(8.0), run_time=2)
        ch.clear_updaters(); ro.clear_updaters()
        self.clear_all()
