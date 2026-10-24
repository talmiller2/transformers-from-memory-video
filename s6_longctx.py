from common import *


class LongContext(VScene):
    def construct(self):
        header = H(r"6.\; Long context: how sharp should attention be?", size=48).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        # ---- too-large beta collapses to argmax ----
        sims = np.array([0.31, 0.12, 0.9, 0.05, 0.84, 0.2, 0.43, 0.15, 0.77, 0.02, 0.36, 0.1])
        beta = ValueTracker(4.0)
        base_y = -1.8

        def bars():
            w = np.exp(beta.get_value() * (sims - sims.max()))
            w = w / w.sum()
            g = VGroup()
            for i, x in enumerate(w):
                g.add(Rectangle(width=0.55, height=max(0.01, 4.0 * x), stroke_width=0,
                                fill_color=SIG if i == int(np.argmax(sims)) else NOISE, fill_opacity=1)
                      .move_to([-3.85 + 0.7 * i, base_y, 0], aligned_edge=DOWN))
            return g

        def label():
            return M(rf"\beta = {beta.get_value():.0f}", size=40).move_to([4.6, 1.4, 0])

        b, bl = always_redraw(bars), always_redraw(label)
        base = Line(LEFT * 4.4, RIGHT * 4.4, color=DIM).shift(UP * base_y + LEFT * 0.0)
        cap = T(r"softmax weights over 12 tokens", size=28, color=SOFT).next_to(base, DOWN, buff=0.25)
        with self.voice(
            "In long contexts, M keeps growing. With beta fixed at eight, the interference reaches the signal at "
            "around a million tokens. The formula says a sharper beta helps, since the win, e to the minus beta, "
            "gets stronger. So should we just keep turning beta up?"
        ):
            self.play(Create(base), FadeIn(b), FadeIn(bl), FadeIn(cap))
        warn = VGroup(
            T(r"$\beta \to \infty$: softmax $\to$ argmax", size=34, color=SIG),
            T(r"gradients to every other token vanish", size=30, color=SOFT),
        ).arrange(DOWN, buff=0.15).move_to([4.2, 0.0, 0])
        keep_in_frame(warn)
        with self.voice(
            "Only up to a point. As beta grows, softmax becomes a hard argmax. All the weight lands on a single "
            "token, the output stops responding smoothly to the others, and their gradients vanish. "
            "Training needs some softness."
        ) as tr:
            self.play(beta.animate.set_value(60), run_time=tr.frac(0.55), rate_func=smooth)
            self.play(FadeIn(warn, shift=UP * 0.2))
        b.clear_updaters(); bl.clear_updaters()
        self.play(FadeOut(VGroup(b, bl, base, cap, warn)))

        # ---- scaling rules ----
        rules = VGroup(
            M(r"\text{fixed: }\; \beta = 8", size=34, color=SOFT),
            M(r"\text{log-}n\text{ scaling: }\; \beta \cdot \frac{\log M}{\log L}", size=34, color=QUERY),
            M(r"\text{YaRN (Qwen2.5-1M): }\; \beta\cdot\big(0.1\,\ln\tfrac{M}{L} + 1\big)^2", size=34, color=VAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(header, DOWN, buff=0.4)
        cite = VGroup(T(r"log-$n$: Su (2021); Scalable-Softmax; Qwen LogN-Scaling", size=24, color=FAINT),
                      T(r"$L$ = training length (1024 here)", size=24, color=FAINT)).arrange(DOWN, buff=0.1)
        cite.to_edge(DOWN, buff=0.4)
        with self.voice(
            "So in practice, beta grows slowly, with the logarithm of the context length. Log n scaling, from Su "
            "in twenty twenty one, Scalable Softmax, and Qwen, multiplies the logits by log M over log L, where L is "
            "the training length. Qwen two point five one M, through YaRN, uses a gentler squared log factor."
        ) as tr:
            for r in rules:
                self.play(Write(r), run_time=tr.frac(0.22))
            self.play(FadeIn(cite))
        self.play(FadeOut(cite), FadeOut(rules))

        plot = loglog_axes(x_max_exp=6.4, y_min_exp=-5, y_max_exp=1, w=10.5, h=5.4).move_to(DOWN * 0.55 + RIGHT * 0.3)
        ax = plot[0]
        cols = {"fixed": SOFT, "logn": QUERY, "yarn": VAL}
        curves = VGroup(*[curve_on(ax, k, c, source="long_context", y_min_exp=-5) for k, c in cols.items()])
        mill = Dot(ax.c2p(np.log10(1.2e6), 0), color=SIG, radius=0.08)

        def at(key, x):  # log10 of a curve's value at log10(M) = x
            grid = np.log10(np.array(DATA["grid"], dtype=float))
            return np.interp(x, grid, np.log10(DATA["long_context"][key]))

        names = {"fixed": r"fixed $\beta = 8$", "logn": r"log-$n$ scaling", "yarn": "YaRN"}
        where = {"fixed": (4.2, 0.5), "logn": (5.6, 0.45), "yarn": (5.0, -0.55)}  # (log10 M, offset in decades)
        labels = VGroup(*[T(names[k], size=34, color=c).move_to(ax.c2p(where[k][0], at(k, where[k][0]) + where[k][1]))
                          for k, c in cols.items()])
        with self.voice(
            "Plot all three out to two million memories. The fixed beta store crosses one near a million. "
            "The scaled versions bend away, keeping the interference orders of magnitude below the signal, "
            "far past two million tokens.",
            gap=0.15,
        ) as tr:
            self.play(Create(plot), run_time=1.2)
            self.play(Create(curves[0]), FadeIn(labels[0]), run_time=tr.frac(0.2))
            self.play(FadeIn(mill, scale=2))
            self.play(Create(curves[1]), Create(curves[2]), FadeIn(labels[1:]), run_time=tr.frac(0.3))
        self.clear_all()
