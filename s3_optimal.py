from common import *


class OptimalLinear(VScene):
    def construct(self):
        header = H(r"3.\; The optimal linear memory store", size=48).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        obj = M(r"\min_W \sum_{i=1}^{M} \|W k_i - v_i\|_2^2", size=56).shift(UP * 1.2)
        sol = M(r"W", r"=", r"V^\top K", r"\,(K^\top K)^{-1}", size=60).next_to(obj, DOWN, buff=0.8)
        sol[3].set_color(ACCENT)
        outer = M(r"W_{\text{outer}} = V^\top K", size=40, color=SOFT).next_to(sol, DOWN, buff=0.6)
        with self.voice(
            "First lever: write carefully. Instead of blindly adding outer products, choose the matrix W that "
            "fits every memory at once, minimizing the total recall error."
        ):
            self.play(Write(obj), run_time=2)
        with self.voice(
            "That's ordinary least squares, and its solution is V transpose K, times the inverse of K transpose K. "
            "It's the outer product store, with one extra factor."
        ) as tr:
            self.play(Write(sol), run_time=2)
            self.play(FadeIn(outer, shift=UP * 0.2))
            self.play(Indicate(sol[3], color=ACCENT, scale_factor=1.15))

        # ---- whitening ----
        self.play(FadeOut(VGroup(obj, outer)), sol.animate.scale(0.7).next_to(header, DOWN, buff=0.3))
        r = np.random.default_rng(4)
        raw = r.standard_normal((220, 2)) * np.array([1.5, 0.35])
        rot = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
        pts = raw @ rot.T
        cov = pts.T @ pts / len(pts)
        w, U = np.linalg.eigh(cov)
        white = U @ np.diag(w ** -0.5) @ U.T
        cloud = VGroup(*[Dot([p[0] * 1.1, p[1] * 1.1, 0], radius=0.04, color=KEY) for p in pts]).shift(DOWN * 0.8)
        lab_before = T("keys crowd into a few directions", size=30, color=SOFT).to_edge(DOWN, buff=0.4)
        lab_after = T("after whitening: every direction counts equally", size=30, color=SOFT).to_edge(DOWN, buff=0.4)
        with self.voice(
            "That factor whitens the keys. If many keys share a direction, the plain outer product store counts "
            "that direction over and over. The inverse turns crowded directions down and sparse ones up, until "
            "the keys look evenly spread."
        ) as tr:
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in cloud], lag_ratio=0.005), FadeIn(lab_before), run_time=1.5)
            self.wait(tr.frac(0.25))
            wp = pts @ white.T * 0.9
            cloud2 = VGroup(*[Dot([q[0] * 1.1, q[1] * 1.1, 0], radius=0.04, color=KEY) for q in wp]).shift(DOWN * 0.8)
            self.play(Transform(cloud, cloud2), FadeTransform(lab_before, lab_after), run_time=2.5)
        self.play(FadeOut(VGroup(cloud, lab_after)))

        # ---- dual basis: exact recall when M <= d ----
        pl = NumberPlane(x_range=[-1.5, 1.8, 1], y_range=[-1.2, 1.6, 1], x_length=5.8, y_length=4.9,
                         background_line_style={"stroke_color": DIM, "stroke_opacity": 0.5, "stroke_width": 1},
                         axis_config={"stroke_color": FAINT, "stroke_width": 1.5}).shift(LEFT * 3 + DOWN * 0.9)
        k1, k2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
        Kinv = np.linalg.inv(np.stack([k1, k2]))
        d1, d2 = Kinv[:, 0], Kinv[:, 1]

        def ar(v, col, sw=6):
            return Arrow(pl.c2p(0, 0), pl.c2p(*v), buff=0, color=col, stroke_width=sw, max_tip_length_to_length_ratio=0.15)

        a1, a2 = ar(k1, KEY), ar(k2, KEY)
        l1 = M("k_1", color=KEY, size=36).next_to(a1.get_end(), DOWN, buff=0.12)
        l2 = M("k_2", color=KEY, size=36).next_to(a2.get_end(), UP, buff=0.1)
        b1, b2 = ar(d1, ACCENT), ar(d2, ACCENT)
        m1 = M(r"\tilde k_1", color=ACCENT, size=36).next_to(b1.get_end(), DR, buff=0.05)
        m2 = M(r"\tilde k_2", color=ACCENT, size=36).next_to(b2.get_end(), LEFT, buff=0.1)
        ra1 = RightAngle(Line(pl.c2p(0, 0), pl.c2p(*d1)), Line(pl.c2p(0, 0), pl.c2p(*k2)), length=0.25, color=INK)
        ra2 = RightAngle(Line(pl.c2p(0, 0), pl.c2p(*k1)), Line(pl.c2p(0, 0), pl.c2p(*d2)), length=0.25, color=INK)
        leak = M(r"\text{outer product: } k_2^\top k_1 = 0.5 \;\Rightarrow\; \text{leak}", size=34, color=NOISE)
        dual = M(r"\tilde k_i^\top k_j = \begin{cases} 1 & i = j \\ 0 & i \ne j\end{cases}", size=40, color=ACCENT)
        Wd = M(r"W = \sum_i v_i\,\tilde k_i^\top", size=40)
        ex = M(r"W k_j = v_j \text{ exactly, for } M \le d_k", size=36, color=SIG)
        right = VGroup(leak, dual, Wd, ex).arrange(DOWN, buff=0.45, aligned_edge=LEFT).shift(RIGHT * 3.2 + DOWN * 0.8)
        with self.voice(
            "With fewer memories than dimensions, the result is easy to picture. Here are two keys that aren't "
            "perpendicular. The plain store leaks half of one memory into the other."
        ):
            self.play(Create(pl), GrowArrow(a1), GrowArrow(a2), Write(l1), Write(l2), run_time=1.5)
            self.play(Write(leak))
        with self.voice(
            "The optimal store reads each memory with a dual vector instead, k tilde. Each dual vector is perpendicular "
            "to every other key, but still has overlap one with its own."
        ) as tr:
            self.play(GrowArrow(b1), Write(m1), Create(ra1), run_time=1.2)
            self.play(GrowArrow(b2), Write(m2), Create(ra2), run_time=1.2)
            self.play(Write(dual))
        with self.voice(
            "So querying with k one, only the first memory responds. Recall is exact, as long as there are at "
            "most d memories."
        ):
            self.play(Write(Wd))
            self.play(Write(ex))
        self.clear_all()
        self.add(header)

        # ---- M > d: the trace argument ----
        def cells(keep, n=16, w=0.62, h=1.1):
            g = VGroup()
            for i in range(n):
                box = Rectangle(width=w, height=h, stroke_color=INK, stroke_width=1.5)
                sig = Rectangle(width=w, height=h * keep, stroke_width=0, fill_color=SIG, fill_opacity=0.85)
                sig.align_to(box, DOWN)
                noise = Rectangle(width=w, height=h * (1 - keep), stroke_width=0, fill_color=NOISE, fill_opacity=0.6)
                noise.align_to(box, UP)
                g.add(VGroup(box, sig, noise) if keep < 1 else VGroup(box, sig))
            return g.arrange(RIGHT, buff=0.08)

        row_a = cells(1.0).shift(UP * 0.6)
        row_b = cells(0.5).shift(UP * 0.6)
        lab_a = T(r"$M = d_k$: every memory keeps all of its own signal", size=34).next_to(row_a, DOWN, buff=0.4)
        lab_b = T(r"$M = 2d_k$: each keeps about half; the rest is shared noise", size=34).next_to(row_a, DOWN, buff=0.4)
        each = M(r"\text{each memory keeps} \approx \frac{d_k}{M} \text{ of its signal}", size=44, color=SIG)
        each.next_to(lab_a, DOWN, buff=0.5)
        with self.voice(
            "Past d memories, exact recall is impossible. There are only d dimensions of room, and now more than d "
            "memories have to share it."
        ):
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in row_a], lag_ratio=0.05), FadeIn(lab_a), run_time=1.5)
        with self.voice(
            "Roughly speaking, each memory keeps about d over M of its own signal. Double the memories, and each "
            "keeps about half. The rest of its share turns into interference."
        ) as tr:
            self.play(Write(each), run_time=1.5)
            self.wait(tr.frac(0.15))
            self.play(Transform(row_a, row_b), Transform(lab_a, lab_b), run_time=1.5)
        self.play(FadeOut(VGroup(row_a, lab_a, each)))

        ratio = M(r"\frac{\|\text{interference}\|}{\|\text{signal}\|}", r"\approx", r"\sqrt{\frac{M}{d_k} - 1}",
                  size=72, substrings_to_isolate=["M", "d_k", "- 1"]).shift(UP * 0.4 + LEFT * 1.8)
        ratio.get_part_by_tex("M").set_color(NOISE)
        ratio.get_part_by_tex("d_k").set_color(KEY)
        minus = ratio.get_part_by_tex("- 1")
        minus.set_color(ACCENT)
        ml = T(r"zero noise until $M$ passes $d_k$", size=36, color=ACCENT).next_to(ratio, DOWN, buff=1.0).align_to(minus, LEFT).shift(LEFT * 1.0)
        ma = Arrow(ml.get_top(), minus.get_bottom() + DOWN * 0.35, buff=0.1, color=ACCENT, stroke_width=4)
        reach = T(r"ratio reaches 1 at $M = 2d_k = 128$", size=40, color=SIG).to_edge(DOWN, buff=0.7)
        with self.voice(
            "That's what this formula says. It looks just like the outer product store's, with one change: minus one, "
            "inside the root."
        ) as tr:
            self.play(Write(ratio), run_time=2)
            self.play(Circumscribe(minus, color=ACCENT))
        with self.voice(
            "That minus one keeps the noise at exactly zero while M is at most d. And it pushes the point where noise "
            "equals signal from M equals d, out to M equals two d. One hundred twenty eight memories."
        ):
            self.play(FadeIn(ml), GrowArrow(ma))
            self.play(Write(reach))
        self.play(FadeOut(VGroup(ratio, ml, ma, reach)))

        # ---- plot ----
        plot = loglog_axes(x_max_exp=3.5, y_min_exp=-2, y_max_exp=1, w=9.5, h=5).shift(DOWN * 0.3 + RIGHT * 0.4)
        ax = plot[0]
        xs = np.logspace(0.5, 3.5, 300)
        outer_c = loglog_curve(ax, xs, np.sqrt(xs / D), KEY, y_min_exp=-2)
        opt_x = np.logspace(np.log10(64.3), 3.5, 300)
        opt_c = loglog_curve(ax, opt_x, np.sqrt(opt_x / D - 1), ACCENT, y_min_exp=-2)
        drop = DashedLine(ax.c2p(np.log10(64), -2), ax.c2p(np.log10(64), np.log10(0.0712)), color=ACCENT, stroke_width=2)
        hl = T("outer product", size=28, color=KEY).move_to(ax.c2p(1.2, 0.35))
        ol = T("optimal linear", size=28, color=ACCENT).move_to(ax.c2p(3.0, -0.4))
        z = T("zero interference", size=24, color=ACCENT).next_to(drop, LEFT, buff=0.1)
        p1 = Dot(ax.c2p(np.log10(64), 0), color=KEY, radius=0.08)
        p2 = Dot(ax.c2p(np.log10(128), 0), color=ACCENT, radius=0.08)
        p1l = M("64", size=30, color=KEY).next_to(p1, UL, buff=0.08)
        p2l = M("128", size=30, color=ACCENT).next_to(p2, DR, buff=0.08)
        dots = VGroup(mc_dots(ax, "linear", KEY, -2), mc_dots(ax, "optimal", ACCENT, -2))
        with self.voice(
            "Here are both stores side by side, with simulated points on top. The dashed line is where noise equals signal. "
            "The outer product store crosses it at sixty four."
        ) as tr:
            self.play(Create(plot), run_time=1.5)
            self.play(Create(outer_c), FadeIn(hl), run_time=1.5)
            self.play(FadeIn(p1, scale=2), Write(p1l))
        with self.voice(
            "The optimal store has zero interference all the way to sixty four, then crosses at one twenty eight. "
            "Writing carefully doubles the capacity."
        ):
            self.play(Create(drop), FadeIn(z), Create(opt_c), FadeIn(ol), run_time=2)
            self.play(FadeIn(dots), FadeIn(p2, scale=2), Write(p2l))
        with self.voice(
            "That's a real gain. But a factor of two won't get us to a million. For that, we need to change how "
            "the store is read.", gap=0.15,
        ):
            pass
        self.clear_all()
