from common import *


def histogram(samples, ax, bins, color):
    counts, edges = np.histogram(samples, bins=bins, density=True)
    g = VGroup()
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        if c <= 0:
            continue
        r = Rectangle(width=ax.c2p(b, 0)[0] - ax.c2p(a, 0)[0], height=ax.c2p(0, c)[1] - ax.c2p(0, 0)[1],
                      stroke_width=0, fill_color=color, fill_opacity=0.85)
        r.move_to(ax.c2p((a + b) / 2, 0), aligned_edge=DOWN)
        g.add(r)
    return g


def walk_seed(n, step, target):
    for seed in range(500):
        th = np.random.default_rng(seed).uniform(0, TAU, n)
        steps = step * np.stack([np.cos(th), np.sin(th)], 1)
        if abs(np.linalg.norm(steps.sum(0)) - target) < 0.03:
            return steps
    return steps


class Interference(VScene):
    def construct(self):
        header = H(r"2.\; Recall picks up interference", size=48).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        eq = M(r"f(k_j)", r"=", r"v_j", r"+", r"\sum_{i \ne j} v_i\,(k_i^\top k_j)", size=60)
        eq[2].set_color(SIG)
        eq[4].set_color(NOISE)
        bs = Brace(eq[2], DOWN, color=SIG)
        bn = Brace(eq[4], DOWN, color=NOISE)
        ts = T("signal", size=34, color=SIG).next_to(bs, DOWN)
        tn = T("interference", size=34, color=NOISE).next_to(bn, DOWN)
        with self.voice(
            "So how much damage does sharing do? Query the store with one of its own keys, k j. "
            "The sum splits in two. The term where i equals j gives back v j. That's the signal."
        ) as tr:
            self.play(Write(eq[:4]), Write(eq[4]), run_time=2)
            self.play(GrowFromCenter(bs), Write(ts))
        with self.voice(
            "Every other memory chips in its own value, weighted by how much its key overlaps with k j. "
            "That's interference."
        ):
            self.play(GrowFromCenter(bn), Write(tn))
        self.play(VGroup(eq, bs, bn, ts, tn).animate.scale(0.8).next_to(header, DOWN, buff=0.2))
        top = VGroup(eq, bs, bn, ts, tn)

        # ---- random keys are nearly orthogonal in high dimensions ----
        ax = Axes(x_range=[-1, 1, 0.5], y_range=[0, 3.6, 1], x_length=9, y_length=3.6, tips=False,
                  axis_config={"color": DIM}, x_axis_config={"numbers_to_include": [-1, -0.5, 0, 0.5, 1],
                                                            "font_size": 26}).shift(DOWN * 0.6)
        ax.y_axis.set_opacity(0)
        xlab = T(r"overlap $k_i^\top k_j$ between two random keys", size=28, color=SOFT).next_to(ax, DOWN, buff=0.4)
        bins = np.linspace(-1, 1, 61)
        s3 = (unit_vectors(6000, d=3, seed=1) * unit_vectors(6000, d=3, seed=2)).sum(1)
        s64 = (unit_vectors(6000, d=64, seed=1) * unit_vectors(6000, d=64, seed=2)).sum(1)
        h3 = histogram(s3, ax, bins, KEY)
        h64 = histogram(s64, ax, bins, KEY)
        dlab = M(r"d_k = 3", size=40, color=KEY).move_to(ax.c2p(0.7, 1.8))
        dlab64 = M(r"d_k = 64", size=40, color=KEY).move_to(ax.c2p(0.7, 1.8))
        with self.voice(
            "How big are those overlaps? Suppose the keys are random. In three dimensions, two random directions "
            "can overlap by just about anything."
        ):
            self.play(Create(ax), FadeIn(xlab))
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in h3], lag_ratio=0.02), Write(dlab), run_time=2)
        sd = 1 / np.sqrt(64)
        b1 = DashedLine(ax.c2p(-sd, 0), ax.c2p(-sd, 2.3), color=INK, stroke_width=2)
        b2 = DashedLine(ax.c2p(sd, 0), ax.c2p(sd, 2.3), color=INK, stroke_width=2)
        sdl = M(r"\pm \tfrac{1}{\sqrt{d_k}} = \pm\tfrac18", size=36).next_to(b1.get_end(), LEFT, buff=0.15)
        with self.voice(
            "But in high dimensions, random vectors are almost always nearly perpendicular. With sixty four "
            "dimensions, the typical overlap is only about plus or minus one eighth. One over the square root of d."
        ) as tr:
            self.play(Transform(h3, h64), Transform(dlab, dlab64), run_time=2.5)
            self.play(Create(b1), Create(b2), Write(sdl))
        self.play(FadeOut(VGroup(ax, xlab, h3, dlab, b1, b2, sdl)))

        # ---- bar chart: target vs everyone else ----
        def chart(n):
            w = similarity_sample(n, seed=11)
            c = bar_chart(w, width=11, height=2.6).shift(DOWN * 2.0)
            return c

        base = Line(LEFT * 5.7, RIGHT * 5.7, color=DIM, stroke_width=1.5).shift(DOWN * 2.0)
        c32 = chart(32)
        nlab = M(r"M = 32", size=40).to_corner(UR, buff=0.6).shift(DOWN * 1.8)
        tlab = T("the memory we asked for", size=28, color=SIG).next_to(c32[0], UP, buff=0.15).shift(RIGHT * 1.4)
        with self.voice(
            "Here's that as a picture. Each bar is one stored memory's weight in the recall. The red bar is the "
            "memory we asked for, with weight one. Every grey bar is some other memory leaking in."
        ):
            self.play(Create(base), FadeIn(nlab))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN if b.get_center()[1] > -1.6 else UP) for b in c32],
                                  lag_ratio=0.04), run_time=2)
            self.play(FadeIn(tlab))
        cur = c32
        with self.voice(
            "With thirty two memories, the grey bars are small. But store more, and there are more and more of them."
        ) as tr:
            for n in (128, 512):
                nxt = chart(n)
                self.play(Transform(cur, nxt), Transform(nlab, M(rf"M = {n}", size=40).move_to(nlab)),
                          tlab.animate.next_to(nxt[0], UP, buff=0.15).shift(RIGHT * 1.4), run_time=tr.frac(0.35))
        self.play(FadeOut(VGroup(cur, base, nlab, tlab, top)))

        # ---- random walk ----
        U = 2.6
        steps = walk_seed(64, 1 / 8, 1.0) * U
        origin = LEFT * 3.2 + DOWN * 1.2
        sig = Arrow(origin, origin + RIGHT * U, buff=0, color=SIG, stroke_width=7)
        sigl = T("signal: length 1", size=28, color=SIG).next_to(sig, DOWN, buff=0.2)
        start = RIGHT * 1.6 + DOWN * 1.2
        pts = np.vstack([[0, 0], np.cumsum(steps, 0)])
        pts3 = [start + np.array([p[0], p[1], 0]) for p in pts]
        n_tr = ValueTracker(0)

        def path():
            k = int(n_tr.get_value())
            g = VGroup()
            for i in range(k):
                g.add(Line(pts3[i], pts3[i + 1], color=NOISE, stroke_width=3))
            return g

        def resultant():
            k = max(int(n_tr.get_value()), 1)
            return Arrow(start, pts3[k], buff=0, color=INK, stroke_width=5, max_tip_length_to_length_ratio=0.15)

        def readout():
            k = int(n_tr.get_value())
            ln = np.linalg.norm(pts[k]) / U
            return VGroup(M(rf"M = {k}", size=34), M(rf"\|\text{{interference}}\| \approx {ln:.2f}", size=34, color=NOISE)) \
                .arrange(DOWN, aligned_edge=LEFT).to_corner(UR, buff=0.5).shift(DOWN * 1.6)

        walk, res, ro = always_redraw(path), always_redraw(resultant), always_redraw(readout)
        with self.voice(
            "Each one is small, but they add up. Every interfering value points in its own random direction, "
            "so adding them is a random walk. And a random walk of M steps only gets about the square root of M "
            "step lengths from where it started."
        ) as tr:
            self.play(GrowArrow(sig), FadeIn(sigl))
            self.add(walk, res, ro)
            self.play(n_tr.animate.set_value(64), run_time=tr.left(1) - 0.3, rate_func=linear)
        with self.voice(
            "Each step has length about one over root d. So after M steps, the interference has grown to about "
            "root M, over root d."
        ) as tr:
            self.wait(tr.frac(0.2))
            self.play(Indicate(sig, color=SIG))
        walk.clear_updaters(); res.clear_updaters(); ro.clear_updaters()
        self.clear_all()
        self.add(header)

        # ---- reading the formula ----
        ratio = M(r"\frac{\|\text{interference}\|}{\|\text{signal}\|}", r"\approx", r"\sqrt{\frac{M}{d_k}}",
                  size=62, substrings_to_isolate=["M", "d_k"]).shift(DOWN * 0.2 + LEFT * 2.2)
        ratio.get_part_by_tex("M").set_color(NOISE)
        ratio.get_part_by_tex("d_k").set_color(KEY)
        frac = ratio[0]
        num = keep_in_frame(T("noise from all the other memories", size=32, color=NOISE).next_to(frac, UP, buff=0.7))
        den = keep_in_frame(T("the memory you asked for (size 1)", size=32, color=SIG).next_to(frac, DOWN, buff=0.7))
        mlab = T("how many memories crowd in", size=32, color=NOISE)
        keep_in_frame(mlab.next_to(ratio.get_part_by_tex("M"), RIGHT, buff=0.7).shift(UP * 1.0))
        dlab = T("room in key space", size=32, color=KEY)
        keep_in_frame(dlab.next_to(ratio.get_part_by_tex("d_k"), RIGHT, buff=0.7).shift(DOWN * 1.0))
        arrows = VGroup(
            Arrow(num.get_bottom(), frac.get_top() + LEFT * 0.4, buff=0.12, color=NOISE, stroke_width=3),
            Arrow(den.get_top(), frac.get_bottom() + LEFT * 0.4, buff=0.12, color=SIG, stroke_width=3),
            Arrow(mlab.get_left(), ratio.get_part_by_tex("M").get_right(), buff=0.12, color=NOISE, stroke_width=3),
            Arrow(dlab.get_left(), ratio.get_part_by_tex("d_k").get_right(), buff=0.12, color=KEY, stroke_width=3),
        )
        with self.voice(
            "So here's how to read this formula. On top, the noise from every other memory. On the bottom, the "
            "memory you asked for, with size one."
        ) as tr:
            self.play(Write(ratio), run_time=2)
            self.play(FadeIn(num), GrowArrow(arrows[0]))
            self.play(FadeIn(den), GrowArrow(arrows[1]))
        with self.voice(
            "The noise grows with M, the number of memories crowding in, but only like its square root, because "
            "random errors partly cancel. And it shrinks with d, because in more dimensions, random keys overlap less."
        ) as tr:
            self.play(FadeIn(mlab), GrowArrow(arrows[2]), run_time=1)
            self.wait(tr.frac(0.35))
            self.play(FadeIn(dlab), GrowArrow(arrows[3]), run_time=1)
        cap = T(r"recall fails when the ratio reaches 1: \; $M \approx d_k = 64$", size=40, color=SIG).to_edge(DOWN, buff=0.4)
        with self.voice(
            "Recall falls apart when the ratio reaches one. Here that's when M equals d. So the outer product store "
            "holds about d memories: sixty four."
        ):
            self.play(Write(cap))
        self.clear_all()

        # ---- two levers ----
        h = T("How do we reduce interference?", size=44).to_edge(UP, buff=0.6)
        left = VGroup(
            T(r"1.\; bigger $d_k$", size=40, color=KEY),
            T(r"compute and memory grow with it", size=28, color=SOFT),
        ).arrange(DOWN, buff=0.25)
        right = VGroup(
            T(r"2.\; be smarter", size=40, color=VAL),
            T(r"write carefully: account for the other memories", size=28, color=SOFT),
            T(r"recall selectively: ignore weak matches", size=28, color=SOFT),
        ).arrange(DOWN, buff=0.25)
        cols = VGroup(left, right).arrange(DOWN, buff=0.7).next_to(h, DOWN, buff=0.6)
        with self.voice(
            "So how do we do better? There are two levers. One is to make the keys longer: increase d. "
            "That works, but compute and memory grow with it."
        ):
            self.play(Write(h))
            self.play(FadeIn(left, shift=UP * 0.2))
        with self.voice(
            "The other lever is to be smarter. Write more carefully, taking the other memories into account. "
            "And recall more selectively, so weak matches stop leaking in. "
            "The rest of this story pulls on that second lever, and each pull rebuilds a piece of the transformer.",
            gap=0.15,
        ) as tr:
            self.play(FadeIn(right[0], shift=UP * 0.2))
            self.play(FadeIn(right[1], shift=UP * 0.2))
            self.play(FadeIn(right[2], shift=UP * 0.2))
            self.wait(tr.frac(0.2))
            self.play(left.animate.set_opacity(0.35))
        self.clear_all()
