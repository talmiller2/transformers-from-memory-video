from common import *


class Intro(VScene):
    def construct(self):
        # ---- credit card ----
        title = VGroup(
            H(r"You could have invented", size=64),
            H(r"transformers from memory", size=64),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        sub = VGroup(
            T(r"An animated adaptation of the post by Alex Wang", size=36, color=SOFT),
            T(r"\texttt{alexwang.ai/posts/associative-memory}", size=36, color=SOFT),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        disc = VGroup(
            T(r"Independent adaptation. Not made by, endorsed by,", size=36, color=SOFT),
            T(r"or affiliated with Alex Wang.", size=36, color=SOFT),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        text = VGroup(title, sub, disc).arrange(DOWN, buff=0.55, aligned_edge=LEFT).to_edge(LEFT, buff=0.7)
        kea = ImageMobject(str(ROOT / "assets" / "kea.png")).set_height(5.8).to_edge(RIGHT, buff=0.0).shift(RIGHT * 0.9 + UP * 0.5)
        card = Group(kea, text)
        with self.voice(
            "You could have invented transformers from memory. This is an animated adaptation of Alex Wang's post "
            "of the same name. It's an independent project, not made by, or affiliated with, Alex Wang."
        ) as tr:
            self.play(FadeIn(kea, shift=LEFT * 0.3), Write(title), run_time=1.5)
            self.play(FadeIn(sub, shift=UP * 0.2), run_time=1)
            self.wait(tr.frac(0.3))
            self.play(FadeIn(disc), run_time=0.6)

        # ---- the hook: a model recalls rules ----
        ask = T(r"Remind me the rules to the game.", size=34)
        ask_box = SurroundingRectangle(ask, corner_radius=0.2, buff=0.25, color=QUERY, fill_color=QUERY, fill_opacity=0.12)
        q = VGroup(ask_box, ask).to_edge(UP, buff=1.0).shift(RIGHT * 2)
        lines = VGroup(*[
            RoundedRectangle(width=w, height=0.16, corner_radius=0.08, stroke_width=0, fill_color=FAINT, fill_opacity=0.8)
            for w in [6.5, 5.8, 6.2, 4.1, 6.4, 5.5, 3.2]
        ]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        ans_box = SurroundingRectangle(lines, corner_radius=0.2, buff=0.35, color=SOFT)
        ans = VGroup(ans_box, lines).next_to(q, DOWN, buff=0.6).to_edge(LEFT, buff=1.5)

        with self.voice(
            "Imagine you're running late for a friend's board game night, and you ask your favorite AI chat to remind "
            "you of the rules to the game. It just recalls them. The setup, the turns, the scoring."
        ) as tr:
            self.play(FadeOut(card), FadeIn(q, shift=LEFT * 0.3), run_time=0.8)
            self.play(Create(ans_box), run_time=0.6)
            self.play(LaggedStart(*[GrowFromEdge(l, LEFT) for l in lines], lag_ratio=0.25), run_time=tr.left(1) * 0.8)

        # ---- where is it stored? a stack of layers ----
        blocks = VGroup()
        for i in range(3):
            a = RoundedRectangle(width=2.6, height=0.6, corner_radius=0.1, color=ACCENT, fill_opacity=0.15)
            a.add(T("attention", size=26))
            m = RoundedRectangle(width=2.6, height=0.6, corner_radius=0.1, color=VAL, fill_opacity=0.15)
            m.add(T("MLP", size=26))
            blocks.add(a, m)
        blocks.arrange(UP, buff=0.18).scale(0.9).to_edge(RIGHT, buff=1.2)
        dots = M(r"\vdots", color=SOFT).next_to(blocks, UP, buff=0.1)
        with self.voice(
            "Somewhere inside a stack of matrix multiplications, that knowledge was written down. "
            "And something about the question pulled it back out."
        ) as tr:
            self.play(ans.animate.scale(0.6).to_edge(LEFT, buff=0.6), q.animate.scale(0.8).to_edge(LEFT, buff=0.6).shift(UP * 0.4),
                      run_time=1)
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.15), FadeIn(dots), run_time=2)
            arrow = Arrow(ans.get_right(), blocks.get_left(), color=SOFT, buff=0.3)
            self.play(GrowArrow(arrow))
        with self.voice(
            "So how do models remember so much? And how much can they actually store?"
        ):
            cred = VGroup(
                T(r"How do models remember so much?", size=34),
                T(r"How much can they store?", size=34),
            ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
            self.play(FadeIn(cred[0], shift=UP * 0.2))
            self.play(FadeIn(cred[1], shift=UP * 0.2))

        # ---- every layer is a key-value store ----
        with self.voice(
            "The central idea is that each of the big layers in a transformer, the linear maps, the MLPs, and attention "
            "itself, can be read as an associative memory. A store of key value pairs, where you look things up by similarity."
        ) as tr:
            self.play(FadeOut(VGroup(q, ans, arrow, cred)), run_time=0.6)
            self.play(LaggedStart(*[Indicate(b, color=INK, scale_factor=1.05) for b in blocks], lag_ratio=0.1),
                      run_time=tr.frac(0.45))
            store = RoundedRectangle(width=2.4, height=1.6, corner_radius=0.2, color=INK)
            store_lbl = M("f", size=60).move_to(store)
            k = Arrow(LEFT * 5.2, LEFT * 1.4, color=KEY, buff=0, stroke_width=6)
            kl = M(r"\text{key } k", color=KEY, size=36).next_to(k, UP)
            v = Arrow(RIGHT * 1.4, RIGHT * 5.2, color=VAL, buff=0, stroke_width=6)
            vl = M(r"\text{value } v", color=VAL, size=36).next_to(v, UP)
            schema = VGroup(k, kl, store, store_lbl, v, vl).shift(LEFT * 1.6)
            self.play(FadeOut(blocks), FadeOut(dots), run_time=0.6)
            self.play(GrowArrow(k), Write(kl), run_time=0.8)
            self.play(Create(store), Write(store_lbl), run_time=0.8)
            self.play(GrowArrow(v), Write(vl), run_time=0.8)

        lad = ladder(height=4.6).move_to(RIGHT * 1.0 + DOWN * 1.0)
        with self.voice(
            "Seen that way, you can ask each layer one very concrete question. How many memories can it hold, "
            "before recall falls apart?"
        ):
            self.play(schema.animate.scale(0.8).to_corner(UL, buff=0.35), run_time=1)
            self.play(FadeIn(lad), run_time=1)
            qm = T("?", size=80, color=SOFT).next_to(lad, LEFT, buff=1.6)
            self.play(Write(qm))

        rungs = [
            (64, r"linear map: 64", KEY),
            (2731, r"squared-ReLU MLP: $\approx 2{,}700$", VAL),
            (1.2e6, r"softmax attention: $>$1 million", SIG),
        ]
        with self.voice(
            "And the answers climb dramatically. With sixty four dimensional keys, a plain linear map holds about "
            "sixty four memories. A squared ReLU MLP holds about twenty seven hundred. And softmax attention, "
            "more than a million."
        ) as tr:
            self.play(FadeOut(qm), run_time=0.4)
            for val, lab, col in rungs:
                r = ladder_rung(lad, val, lab, col)
                self.play(FadeIn(r[0], scale=2), Write(r[1]), run_time=tr.frac(0.22))
                self.wait(tr.frac(0.06))
        with self.voice(
            "The goal is to see exactly where each of those numbers comes from. And along the way, to rebuild the "
            "pieces of a transformer, one fix at a time.",
            gap=0.15,
        ):
            pass
        self.clear_all()
