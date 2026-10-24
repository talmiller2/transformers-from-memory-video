from common import *

ACTS = [
    ("linear", r"\sigma(s) = s", KEY),
    ("relu", r"\sigma(s) = \mathrm{ReLU}(s)", QUERY),
    ("relu2", r"\sigma(s) = \mathrm{ReLU}(s)^2", VAL),
    ("relu3", r"\sigma(s) = \mathrm{ReLU}(s)^3", ACCENT),
]


class MLPStore(VScene):
    def construct(self):
        header = H(r"4.\; The MLP as a nonlinear memory store", size=48).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        lin = M(r"f(x) = \sum_{i=1}^{M} v_i\,", r"(k_i^\top x)", size=52).shift(UP * 1.6)
        non = M(r"f(x) = \sum_{i=1}^{M} v_i\,", r"\sigma", r"(k_i^\top x)", size=52).move_to(lin)
        non[1].set_color(SIG)
        with self.voice(
            "Second lever: recall selectively. In the stores so far, every memory contributes in proportion to "
            "its similarity, even when that similarity is tiny. What if we passed each similarity through a "
            "nonlinear function first, one that squashes weak matches toward zero, but leaves a perfect match alone?"
        ) as tr:
            self.play(Write(lin), run_time=1.5)
            self.wait(tr.frac(0.35))
            self.play(TransformMatchingTex(lin, non), run_time=1.5)
        self.play(non.animate.scale(0.75).next_to(header, DOWN, buff=0.25))

        # ---- sigma graph + bar chart morphing together ----
        gax = Axes(x_range=[-0.6, 1.05, 0.5], y_range=[-0.6, 1.05, 0.5], x_length=4.8, y_length=4.4, tips=False,
                   axis_config={"color": DIM, "include_numbers": True, "font_size": 22}).shift(LEFT * 4 + DOWN * 1.0)
        gx = M("s", size=30, color=SOFT).next_to(gax.x_axis, RIGHT, buff=0.1)
        sims = similarity_sample(40, seed=21)
        base_y = -1.3
        base = Line(RIGHT * 0.3, RIGHT * 6.7, color=DIM, stroke_width=1.5).shift(UP * base_y)

        def state(i):
            key, tex, col = ACTS[i]
            f = SCORE_FNS[key]
            g = gax.plot(lambda s: float(f(np.array(s))), x_range=[-0.6, 1.0, 0.005], color=col, stroke_width=5)
            lab = M(tex, size=34, color=col).next_to(gax, UP, buff=0.1)
            bars = bar_chart(f(sims) / f(np.array(1.0)), width=6.4, height=2.8).shift(RIGHT * 3.5 + UP * base_y)
            return g, lab, bars

        g, lab, bars = state(0)
        bl = T("recall weights, $M=40$", size=28, color=SOFT).next_to(base, DOWN, buff=1.1)
        with self.voice(
            "On the left is the function we apply to each similarity. On the right, the recall weights for forty "
            "random memories, with the red target at one. With the plain linear store, all the grey bars "
            "come straight through."
        ):
            self.play(Create(gax), FadeIn(gx), Create(base), FadeIn(bl))
            self.play(Create(g), Write(lab), LaggedStart(*[GrowFromEdge(b, DOWN if b.get_center()[1] > base_y else UP)
                                                           for b in bars], lag_ratio=0.03), run_time=2)
        with self.voice(
            "Use a ReLU, and every negative similarity is zeroed out. About half the interference disappears."
        ):
            g2, lab2, bars2 = state(1)
            self.play(Transform(g, g2), Transform(lab, lab2), Transform(bars, bars2), run_time=2)
        pt = Dot(gax.c2p(0.1, 0.01), color=INK)
        ptl = M(r"0.1 \mapsto 0.01", size=32).next_to(gax.c2p(0.1, 0.01), UP + LEFT * 0.3, buff=0.45)
        with self.voice(
            "Now square it. A similarity of one stays one. But a typical stray similarity of about point one becomes "
            "point zero one. Weak matches are crushed ten times harder than the strong one."
        ) as tr:
            g3, lab3, bars3 = state(2)
            self.play(Transform(g, g3), Transform(lab, lab3), Transform(bars, bars3), run_time=2)
            self.play(FadeIn(pt, scale=2), Write(ptl))
        with self.voice("Cube it, and the grey bars all but vanish."):
            g4, lab4, bars4 = state(3)
            self.play(Transform(g, g4), Transform(lab, lab4), Transform(bars, bars4), FadeOut(pt), FadeOut(ptl), run_time=2)
        self.play(FadeOut(VGroup(gax, gx, g, lab, bars, base, bl)))

        # ---- reveal: this is an MLP ----
        mat = M(r"f(x)", r"=", r"V^\top", r"\,\sigma(", r"K", r"x)", size=60).shift(UP * 1.0)
        mat[2].set_color(VAL)
        mat[4].set_color(KEY)
        with self.voice(
            "Now write that in matrix form. Stack the keys as the rows of K, and the values as the columns of V "
            "transpose. Then f of x is V transpose, sigma of K x. And that's exactly a two layer MLP."
        ) as tr:
            self.play(ReplacementTransform(non, mat), run_time=2)
            self.wait(tr.frac(0.35))
            self.play(Indicate(mat[4], color=KEY), Indicate(mat[2], color=VAL))

        def layer(n, x, h):
            return VGroup(*[Circle(0.18, color=SOFT, stroke_width=2).move_to([x, y, 0])
                            for y in np.linspace(h / 2, -h / 2, n)])

        inp, hid, out = layer(4, -3.5, 2.2), layer(8, 0, 3.0), layer(4, 3.5, 2.2)
        net = VGroup(inp, hid, out).shift(DOWN * 1.0)
        e1 = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1, color=KEY, stroke_opacity=0.35)
                      for b in hid for a in inp])
        e2 = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1, color=VAL, stroke_opacity=0.35)
                      for a in hid for b in out])
        lx = M("x", size=36).next_to(inp, LEFT)
        lo = M("f(x)", size=36).next_to(out, RIGHT)
        j = 2
        hi_in = VGroup(*[Line(a.get_center(), hid[j].get_center(), stroke_width=4, color=KEY) for a in inp])
        hi_out = VGroup(*[Line(hid[j].get_center(), b.get_center(), stroke_width=4, color=VAL) for b in out])
        tk = T(r"incoming weights = key $k_i$", size=28, color=KEY).move_to([-2.4, hid.get_top()[1] + 0.45, 0])
        tv = T(r"outgoing weights = value $v_i$", size=28, color=VAL).move_to([2.4, hid.get_top()[1] + 0.45, 0])
        self.play(mat.animate.scale(0.7).next_to(header, DOWN, buff=0.25))
        with self.voice(
            "Each hidden neuron is one memory. Its incoming weights are the key. Its outgoing weights are the value. "
            "The width of the hidden layer is the number of memories."
        ) as tr:
            self.play(Create(e1), Create(e2), FadeIn(inp), FadeIn(hid), FadeIn(out), FadeIn(lx), FadeIn(lo), run_time=2)
            self.play(hid[j].animate.set_fill(SIG, 1).set_stroke(SIG), Create(hi_in), Write(tk), run_time=1.2)
            self.play(Create(hi_out), Write(tv), run_time=1.2)
        note = T(r"trained ReLU MLPs: often only $\sim$1\% of hidden units fire per input", size=28, color=SOFT)
        note.to_edge(DOWN, buff=0.3)
        with self.voice(
            "This matches what we see in trained models. Facts are stored in superposition across neurons, "
            "and in trained ReLU MLPs, often only around one percent of the hidden units fire for a given input. "
            "Selective recall, in action."
        ):
            self.play(FadeIn(note, shift=UP * 0.2))
        with self.voice(
            "A fun historical footnote: an early version of the original transformer paper even described its "
            "feed forward layer as a kind of attention. That passage was later dropped."
        ):
            self.play(FadeOut(note))
            q = VGroup(T(r"early arXiv versions of \emph{Attention Is All You Need}", size=26, color=ACCENT),
                       T(r"described the feed-forward layer as ``a kind of attention''", size=26, color=ACCENT)
                       ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.25)
            self.play(FadeIn(q))
        self.clear_all()
        self.add(header)

        # ---- capacity of higher powers ----
        rule = M(r"\sigma(s) = \mathrm{ReLU}(s)^p", r"\;\Rightarrow\;",
                 r"\frac{\|\text{interference}\|}{\|\text{signal}\|} \sim \sqrt{\frac{M}{d_k^{\,p}}}", size=46)
        rule.next_to(header, DOWN, buff=0.4)
        rows = [
            (r"$\sigma(s)$", r"capacity", r"at $d_k = 64$", INK),
            (r"$s$ (linear)", r"$d_k$", r"64", KEY),
            (r"$\mathrm{ReLU}(s)$", r"$2d_k$", r"128", QUERY),
            (r"$\mathrm{ReLU}(s)^2$", r"$2d_k^2/3$", r"$\approx 2{,}700$", VAL),
            (r"SwiGLU", r"$\approx 2d_k^2/3$", r"$\approx 2{,}900$", SWI),
            (r"$\mathrm{ReLU}(s)^3$", r"$2d_k^3/15$", r"$\approx 35{,}000$", ACCENT),
        ]
        table = VGroup()
        for a, b, c, col in rows:
            table.add(VGroup(T(a, size=32, color=col), T(b, size=32, color=col), T(c, size=32, color=col)))
        for r in table:
            for i, cell in enumerate(r):
                cell.move_to([[-3.2, 0.2, 3.2][i], 0, 0])
        table.arrange(DOWN, buff=0.28).next_to(rule, DOWN, buff=0.5)
        for r in table:
            for i, cell in enumerate(r):
                cell.set_x([-3.2, 0.2, 3.2][i])
        hline = Line(LEFT * 4.6, RIGHT * 4.6, color=DIM).next_to(table[0], DOWN, buff=0.14)
        why = VGroup(
            M(r"\text{stray match: }\; \tfrac{1}{\sqrt{d_k}} \;\longmapsto\; \Big(\tfrac{1}{\sqrt{d_k}}\Big)^{p} = \tfrac{1}{\sqrt{d_k^{\,p}}}",
              size=44, color=NOISE),
            M(r"\text{target: }\; 1 \;\longmapsto\; 1^p = 1", size=44, color=SIG),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT).next_to(rule, DOWN, buff=0.8)
        with self.voice(
            "In general, raise the ReLU to the power p. Then the ratio becomes root M over d to the p."
        ):
            self.play(Write(rule), run_time=2.5)
        with self.voice(
            "Here's why. A typical stray similarity is about one over root d. Raised to the power p, it shrinks to "
            "one over root d to the p. The target's similarity is one, and one to any power is still one."
        ) as tr:
            self.play(Write(why[0]), run_time=2)
            self.wait(tr.frac(0.2))
            self.play(Write(why[1]), run_time=1.2)
        with self.voice(
            "So it's the same random walk as before, with d replaced by d to the p. Capacity grows from d, to d "
            "squared, to d cubed, give or take a small constant."
        ) as tr:
            self.play(Indicate(rule[2], color=VAL, scale_factor=1.1), run_time=1.5)
        self.play(FadeOut(why))
        with self.voice(
            "With sixty four dimensional keys: a linear store holds sixty four. ReLU doubles that to one twenty eight. "
            "Squared ReLU jumps to about twenty seven hundred. And cubed ReLU, about thirty five thousand."
        ) as tr:
            self.play(FadeIn(table[0]), Create(hline))
            for i in (1, 2, 3, 5):
                self.play(FadeIn(table[i], shift=RIGHT * 0.2), run_time=tr.frac(0.12))
                self.wait(tr.frac(0.08))
        glu = VGroup(
            M(r"\mathrm{ReLU}(Kx)^2 = \mathrm{ReLU}(Kx) \odot Kx", size=36),
            M(r"\text{SwiGLU: }\; \sigma(s) = \mathrm{Swish}(s)\cdot s = \frac{s^2}{1+e^{-s}}", size=36),
        ).arrange(DOWN, buff=0.3).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(rule), VGroup(table, hline).animate.next_to(header, DOWN, buff=0.5))
        glu.to_edge(DOWN, buff=0.5)
        with self.voice(
            "Squared ReLU might look exotic, but it's a gated linear unit in disguise: ReLU of K x, times K x. "
            "And SwiGLU, the activation in most modern language models, behaves almost the same way, landing "
            "around twenty nine hundred."
        ):
            self.play(Write(glu[0]), run_time=1.5)
            self.play(Write(glu[1]), FadeIn(table[4], shift=RIGHT * 0.2), run_time=1.5)
        self.play(FadeOut(VGroup(table, hline, glu)))

        # ---- plot ----
        plot = loglog_axes(x_max_exp=5.5, y_min_exp=-3, y_max_exp=1, w=10, h=4.8).shift(DOWN * 0.1 + RIGHT * 0.4)
        ax = plot[0]
        curves = VGroup(*[curve_on(ax, k, c) for k, _, c in ACTS[:2]], curve_on(ax, "relu2", VAL), curve_on(ax, "relu3", ACCENT))
        dots = VGroup(*[mc_dots(ax, k, c) for k, c in
                        [("linear", KEY), ("relu", QUERY), ("relu2", VAL), ("relu3", ACCENT), ("swiglu", SWI)]])
        caps = [(64, KEY), (128, QUERY), (2731, VAL), (34953, ACCENT)]
        cross = VGroup(*[Dot(ax.c2p(np.log10(m), 0), radius=0.08, color=c) for m, c in caps])
        labs = VGroup(*[T(n, size=26, color=c).next_to(ax.c2p(np.log10(m), 0), d, buff=0.15)
                        for n, (m, c), d in zip(["linear", "ReLU", "ReLU$^2$", "ReLU$^3$"], caps,
                                                [UP + LEFT * 0.6, DOWN + RIGHT * 0.6, DOWN + RIGHT * 0.6, DOWN + RIGHT * 0.6])])
        with self.voice(
            "On the log log plot, each power tilts the line down and pushes its crossing point further right. "
            "The dots are simulations. They sit right on the predictions, with SwiGLU, in orange, hugging squared ReLU.",
            gap=0.15,
        ) as tr:
            self.play(Create(plot), run_time=1.2)
            for c, l, x in zip(curves, labs, cross):
                self.play(Create(c), FadeIn(l), FadeIn(x, scale=2), run_time=tr.frac(0.12))
            self.play(FadeIn(dots))
        self.clear_all()
