from common import *

POS, NEG = SIG, KEY


def heat(vals, size=0.3):
    vals = np.atleast_2d(vals)
    m = np.abs(vals).max() or 1
    g = VGroup()
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            x = vals[r, c] / m
            col = interpolate_color(ManimColor(BG), ManimColor(POS if x > 0 else NEG), min(1, abs(x)) ** 0.7)
            sq = Square(size, stroke_width=1, stroke_color=LINE, fill_color=col, fill_opacity=1)
            sq.move_to([c * size, -r * size, 0])
            g.add(sq)
    return g.center()


def plane(shift):
    p = NumberPlane(
        x_range=[-1.6, 1.6, 1], y_range=[-1.6, 1.6, 1], x_length=4.2, y_length=4.2,
        background_line_style={"stroke_color": DIM, "stroke_opacity": 0.5, "stroke_width": 1},
        axis_config={"stroke_color": FAINT, "stroke_width": 1.5},
    )
    return p.shift(shift)


def vec(p, xy, color, start=(0, 0), stroke_width=6):
    return Arrow(p.c2p(*start), p.c2p(start[0] + xy[0], start[1] + xy[1]), buff=0, color=color,
                 stroke_width=stroke_width, max_tip_length_to_length_ratio=0.18)


class OuterProduct(VScene):
    def construct(self):
        header = H(r"1.\; The outer-product memory store", size=48).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        kp, vp = plane(LEFT * 3.7 + DOWN * 0.6), plane(RIGHT * 3.7 + DOWN * 0.6)
        klab = T("key space", size=30, color=KEY).next_to(kp, UP, buff=0.15)
        vlab = T("value space", size=30, color=VAL).next_to(vp, UP, buff=0.15)
        K = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.8, 0.6])]
        V = [np.array([1.1, 0.6]), np.array([-0.7, 1.0]), np.array([0.3, -1.1])]
        ka = [vec(kp, k, KEY) for k in K]
        va = [vec(vp, v, VAL) for v in V]
        kl = [M(rf"k_{i+1}", size=36, color=KEY).next_to(ka[i].get_end(), UR * 0.6, buff=0.08) for i in range(3)]
        vl = [M(rf"v_{i+1}", size=36, color=VAL).next_to(va[i].get_end(), UP if i < 2 else DR * 0.6, buff=0.08)
              for i in range(3)]

        with self.voice(
            "Let's begin with the simplest memory there is. A memory is a pair of vectors. A key, which is what "
            "you'll search by, and a value, which is what you want back."
        ) as tr:
            self.play(Create(kp), Create(vp), FadeIn(klab), FadeIn(vlab), run_time=1.5)
            for i in range(2):
                self.play(GrowArrow(ka[i]), Write(kl[i]), run_time=0.7)
                self.play(GrowArrow(va[i]), Write(vl[i]), run_time=0.7)
        planes = VGroup(kp, vp, klab, vlab, *ka[:2], *va[:2], *kl[:2], *vl[:2])

        # ---- writing with outer products ----
        self.play(planes.animate.set_opacity(0.0), run_time=0.6)
        self.remove(planes)
        rng = np.random.default_rng(3)
        v6, k6 = rng.standard_normal(6), rng.standard_normal(6)
        vcol = heat(v6[:, None]).set_stroke(VAL, 2)
        krow = heat(k6[None, :]).set_stroke(KEY, 2)
        grid = heat(np.outer(v6, k6))
        vcol.move_to(LEFT * 4.2 + DOWN * 0.2)
        krow.next_to(vcol, RIGHT, buff=0.5).align_to(vcol, UP)
        eq = M("=", size=60).next_to(krow, RIGHT, buff=0.5).match_y(vcol)
        grid.next_to(eq, RIGHT, buff=0.5)
        vcl = M("v", color=VAL).next_to(vcol, DOWN)
        krl = M(r"k^\top", color=KEY).next_to(krow, DOWN)
        gl = M(r"v\,k^\top", size=44).next_to(grid, DOWN)
        rule = T(r"write: add $v_i\,k_i^\top$ for every memory", size=34).to_edge(UP, buff=1.2)

        with self.voice(
            "How do we store pairs like these in a single matrix? The simplest answer: for every pair, "
            "add the outer product, the value times the key transposed."
        ) as tr:
            self.play(Write(rule), run_time=1.2)
            self.play(FadeIn(vcol), Write(vcl), FadeIn(krow), Write(krl), run_time=1)
        with self.voice(
            "An outer product is a grid. Every entry is one coordinate of v, times one coordinate of k."
        ) as tr:
            self.play(Write(eq))
            row_anims = []
            for r in range(6):
                row = VGroup(*grid[r * 6:(r + 1) * 6])
                row_anims.append(AnimationGroup(Indicate(vcol[r], color=INK, scale_factor=1.15),
                                                FadeIn(row, lag_ratio=0.2)))
            self.play(LaggedStart(*row_anims, lag_ratio=0.5), run_time=tr.left(1) - 0.6)
            self.play(Write(gl), run_time=0.5)

        trio = VGroup(*[heat(np.outer(rng.standard_normal(6), rng.standard_normal(6)), size=0.22) for _ in range(3)])
        total = heat(sum(np.outer(rng.standard_normal(6), rng.standard_normal(6)) for _ in range(3)), size=0.3)
        pluses = [M("+", size=48) for _ in range(2)]
        seq = VGroup(trio[0], pluses[0], trio[1], pluses[1], trio[2], M("=", size=48), total).arrange(RIGHT, buff=0.35)
        seq.move_to(UP * 0.5)
        under = VGroup(*[M(rf"v_{i+1}k_{i+1}^\top", size=32).next_to(trio[i], DOWN) for i in range(3)])
        wl = M("W", size=44).next_to(total, DOWN)
        formula = M(r"W", r"=", r"\sum_{i=1}^{M} v_i\,k_i^\top", r"= V^\top K", size=52).to_edge(DOWN, buff=0.6)

        with self.voice(
            "To store many memories, just add up their outer products. W is the sum, over all M memories, of v i "
            "k i transposed. Stack the keys as the rows of a matrix K, and the values as rows of V, and this is "
            "simply V transpose times K."
        ) as tr:
            self.play(FadeOut(VGroup(vcol, krow, vcl, krl, eq, gl)), ReplacementTransform(grid, trio[0]), run_time=1)
            self.play(FadeIn(under[0]), FadeIn(pluses[0]), FadeIn(trio[1]), FadeIn(under[1]),
                      FadeIn(pluses[1]), FadeIn(trio[2]), FadeIn(under[2]), run_time=1.2)
            self.play(Write(seq[5]), FadeIn(total, shift=LEFT * 0.3), Write(wl), run_time=1)
            self.play(Write(formula), run_time=2)
        self.play(FadeOut(VGroup(seq, under, wl, rule)), formula.animate.to_edge(UP, buff=1.1).scale(0.8))

        # ---- recall ----
        recall = M(r"f(x) = Wx", r"= \sum_{i=1}^{M} v_i\,", r"(k_i^\top x)", size=52).next_to(formula, DOWN, buff=0.4)
        recall[2].set_color(QUERY)
        sim_note = T("similarity of key $i$ to the query", size=28, color=QUERY).next_to(recall, DOWN, buff=0.35).align_to(recall[2], LEFT)
        with self.voice(
            "Now, to recall, multiply W by a query vector x. Since W is a sum, W x is a sum too. "
            "Each stored value, weighted by the dot product between its key and the query. "
            "In other words, recall is a similarity weighted average of everything in the store."
        ) as tr:
            self.play(Write(recall[:2]), run_time=1.5)
            self.play(Write(recall[2]), run_time=1)
            self.play(FadeIn(sim_note, shift=UP * 0.2))
        self.play(FadeOut(VGroup(formula, sim_note)), recall.animate.scale(0.7).to_edge(UP, buff=0.3), FadeOut(header))

        planes.set_opacity(1)
        for m in (kp, vp):
            m.background_lines.set_stroke(opacity=0.5)
        self.play(FadeIn(planes), run_time=1)
        x_arrow = vec(kp, K[0], QUERY, stroke_width=9)
        xl = M("x", color=QUERY, size=40).next_to(x_arrow.get_end(), DOWN, buff=0.15)
        w1 = M(r"k_1^\top x = 1", size=32, color=QUERY)
        w2 = M(r"k_2^\top x = 0", size=32, color=QUERY)
        VGroup(w1, w2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(UP * 0.4)
        out1 = vec(vp, V[0], INK, stroke_width=9)
        with self.voice(
            "Query with the first key. Its similarity to itself is one. The second key is perpendicular, so its "
            "similarity is zero. The second value gets weight zero, and we get back exactly v one. Perfect recall."
        ) as tr:
            self.play(GrowArrow(x_arrow), Write(xl))
            self.play(Write(w1), run_time=0.8)
            self.play(Write(w2), va[1].animate.set_opacity(0.25), vl[1].animate.set_opacity(0.25), run_time=0.8)
            self.play(TransformFromCopy(va[0], out1), run_time=1.2)
            self.play(Flash(out1.get_end(), color=INK), run_time=0.6)

        # ---- a third memory doesn't fit ----
        w3 = M(r"k_3^\top x = 0.8", size=32, color=SIG).next_to(w2, DOWN, aligned_edge=LEFT, buff=0.3)
        scaled3 = vec(vp, 0.8 * V[2], NOISE, start=tuple(V[0]))
        result = vec(vp, V[0] + 0.8 * V[2], SIG, stroke_width=7)
        target = DashedLine(vp.c2p(0, 0), vp.c2p(*V[0]), color=INK, stroke_width=3)
        with self.voice(
            "But now store a third memory. In two dimensions, there's no room for a third perpendicular key. "
            "Query with k one again, and the third key overlaps with it, by point eight. So the answer comes back "
            "as v one, plus point eight of v three. The memory is contaminated."
        ) as tr:
            self.play(va[1].animate.set_opacity(1), vl[1].animate.set_opacity(1), run_time=0.4)
            self.play(GrowArrow(ka[2]), Write(kl[2]), GrowArrow(va[2]), Write(vl[2]), run_time=1.2)
            self.wait(tr.frac(0.15))
            self.play(Write(w3), run_time=0.8)
            self.play(TransformFromCopy(va[2], scaled3), run_time=1.2)
            self.add(target)
            self.play(ReplacementTransform(out1, result), FadeIn(target), run_time=1.2)
        with self.voice(
            "In d dimensions, you can line up at most d perfectly perpendicular keys. "
            "Every memory after that leaks into the others."
        ):
            dl = T(r"at most $d_k$ orthogonal keys fit in $\mathbb{R}^{d_k}$", size=34).to_edge(DOWN, buff=0.35)
            self.play(Write(dl))
        self.clear_all()

        # ---- aside: gradient descent writes memories ----
        aside = T(r"Aside: gradient descent writes memories", size=38, color=ACCENT).to_edge(UP, buff=0.8)
        upd = M(r"W \leftarrow W", r"\;-\;\eta\,\bar y", r"\,x^\top", size=64)
        upd[1].set_color(VAL)
        upd[2].set_color(KEY)
        b1 = Brace(upd[1], DOWN, color=VAL)
        b2 = Brace(upd[2], DOWN, color=KEY)
        t1 = T("value", size=32, color=VAL).next_to(b1, DOWN)
        t2 = T("key", size=32, color=KEY).next_to(b2, DOWN)
        ybar = M(r"\bar y = \partial L / \partial y", size=32, color=SOFT).to_edge(DOWN, buff=1.0)
        with self.voice(
            "A quick aside, because this isn't just a toy. Take one step of gradient descent on a linear layer. "
            "The update is minus the learning rate, times the gradient at the output, times the input transposed. "
            "That's an outer product. Every training step writes a memory into the weights, with the layer's input as the key."
        ) as tr:
            self.play(FadeIn(aside))
            self.play(Write(upd), FadeIn(ybar), run_time=2)
            self.wait(tr.frac(0.3))
            self.play(GrowFromCenter(b1), Write(t1), GrowFromCenter(b2), Write(t2), run_time=1.2)
        self.clear_all()
