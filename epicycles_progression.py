"""
3Blue1Brown 风格：从 1 到 4 个圆盘递进演绎各类摆线 (Epicycles Progression)
完全保持与 fourier_epicycles_drawing.py 一致的圆盘、向量、轨迹与画中画样式，
依次平滑过渡展示：
  第 1 幕 (1 个圆盘) -> 标准圆 (Circle)
  第 2 幕 (2 个圆盘) -> 三尖内摆线 (Deltoid, 半径比 2:1, 转速 1:-2)
  第 3 幕 (3 个圆盘) -> 四叶回旋花瓣线 (4-Cusp Floral Curve, 转速 1, -3, 5)
  第 4 幕 (4 个圆盘) -> 五星交叠星芒线 (Pentagram Star-Loop, 转速 1, -4, 6, -9)

运行方式：
  manim -pql epicycles_progression.py EpicyclesProgressionScene      # 低分辨率快速预览
  manim -pqh epicycles_progression.py EpicyclesProgressionScene      # 1080P 高清渲染
"""

import numpy as np
from manim import *

# ==========================================
# 1. 核心参数与 4 幕圆盘组合配置
# ==========================================
CONFIG = {
    # 每一幕完整绘制一周的时长（秒）
    "stage_draw_duration": 8.0,
    
    # 幕与幕之间平滑变形过渡的时长（秒）
    "transition_duration": 1.4,
    
    # 轨迹超采样细分点数（保证在任何帧率下线条都极致丝滑）
    "path_subsamples": 2000,
    
    # 画面布局与配色（与主程序完全一致）
    "center_offset": RIGHT * 1.5,  # 整体向右平移，为左上角画中画留出空间
    "zoom_factor": 0.25,           # 局部画中画放大倍数（0.25 对应 4 倍放大）
    "display_width": 4.6,
    "display_height": 3.0,
    "box_color": TEAL_C,           # 青蓝色取景框
    "path_color": "#22d3ee",       # 流光青色轨迹
}

# 定义 4 幕圆盘组合：每一幕由若干个 (半径 r, 转速 k, 初始相位 phi) 组成
STAGES = [
    {
        "name": "1 Disk: Circle",
        "epicycles": [
            # (radius, freq, phase)
            (2.2, 1, 0.0),
        ],
    },
    {
        "name": "2 Disks: Deltoid (2:1, 1:-2)",
        "epicycles": [
            (1.6, 1, 0.0),
            (0.8, -2, 0.0),
        ],
    },
    {
        "name": "3 Disks: 4-Cusp Floral Curve",
        "epicycles": [
            (1.6, 1, 0.0),
            (0.65, -3, 0.0),
            (0.32, 5, 0.0),
        ],
    },
    {
        "name": "4 Disks: Pentagram Star-Loop",
        "epicycles": [
            (1.55, 1, 0.0),
            (0.70, -4, 0.0),
            (0.32, 6, 0.0),
            (0.16, -9, 0.0),
        ],
    },
]


# ==========================================
# 2. Manim 递进演化场景类 (含 ZoomedScene 画中画)
# ==========================================
class EpicyclesProgressionScene(ZoomedScene):
    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=CONFIG["zoom_factor"],
            zoomed_display_height=CONFIG["display_height"],
            zoomed_display_width=CONFIG["display_width"],
            image_frame_stroke_width=2,
            zoomed_camera_config={
                "default_frame_stroke_width": 2,
                "default_frame_stroke_color": CONFIG["box_color"],
            },
            **kwargs,
        )

    @staticmethod
    def build_coeffs_and_freqs(epicycle_tuples):
        """将 (radius, freq, phase) 列表转化为复数系数 coeffs 与频率数组 freqs（含直流质心 c0=0）"""
        coeffs = [0.0 + 0.0j]
        freqs = [0]
        for r, k, phi in epicycle_tuples:
            coeffs.append(r * np.exp(1j * phi))
            freqs.append(int(k))
        return np.array(coeffs, dtype=complex), np.array(freqs, dtype=int)

    @staticmethod
    def build_epicycle_vgroup(coeffs, freqs, t, center_offset):
        """根据给定时刻 t 构建与之前样式完全一致的圆盘与向量组"""
        group = VGroup()
        curr_pos = np.array([coeffs[0].real, coeffs[0].imag, 0.0]) + center_offset

        for i, (c, k) in enumerate(zip(coeffs[1:], freqs[1:])):
            radius = np.abs(c)
            angle = np.angle(c) + 2 * np.pi * k * t
            vec = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
            next_pos = curr_pos + vec

            # 圆盘轨道环（与上一个代码文件样式完全一致）
            if radius > 0.008:
                circle_opacity = max(0.12, 0.45 - (i / max(1, len(coeffs))) * 0.25)
                circle = Circle(
                    radius=radius,
                    stroke_color=GOLD_B if i == 0 else GREY_B,
                    stroke_width=1.2,
                    stroke_opacity=circle_opacity,
                ).move_to(curr_pos)
                group.add(circle)

            # 旋转向量臂（与上一个代码文件样式完全一致）
            if radius > 0.06:
                arrow = Arrow(
                    start=curr_pos,
                    end=next_pos,
                    buff=0,
                    stroke_width=max(1.2, min(2.4, radius * 2.2)),
                    max_tip_length_to_length_ratio=min(0.25, 0.14 / max(radius, 0.06)),
                    color=WHITE,
                    stroke_opacity=0.88,
                )
                group.add(arrow)
            elif radius > 0.008:
                line = Line(
                    start=curr_pos,
                    end=next_pos,
                    stroke_width=1.0,
                    color=WHITE,
                    stroke_opacity=0.72,
                )
                group.add(line)

            curr_pos = next_pos

        return group

    def construct(self):
        center_offset = CONFIG["center_offset"]
        n_sub = CONFIG["path_subsamples"]
        t_dense = np.linspace(0.0, 1.0, n_sub, endpoint=True)

        # 全局时间进度追踪器与当前阶段的系数容器
        time_tracker = ValueTracker(0.0)
        active_state = {
            "coeffs": None,
            "freqs": None,
            "dense_path_3d": None,
        }

        def update_active_stage_data(stage_idx):
            coeffs, freqs = self.build_coeffs_and_freqs(STAGES[stage_idx]["epicycles"])
            exp_matrix = np.exp(1j * 2 * np.pi * np.outer(freqs, t_dense))
            z_dense = np.dot(coeffs, exp_matrix)
            dense_path_3d = np.column_stack([
                z_dense.real + center_offset[0],
                z_dense.imag + center_offset[1],
                np.zeros(n_sub),
            ])
            active_state["coeffs"] = coeffs
            active_state["freqs"] = freqs
            active_state["dense_path_3d"] = dense_path_3d

        # 初始化第 1 幕数据
        update_active_stage_data(0)

        def get_current_pen_tip_pos():
            t = time_tracker.get_value()
            coeffs = active_state["coeffs"]
            freqs = active_state["freqs"]
            current_z = np.sum(coeffs * np.exp(1j * 2 * np.pi * freqs * t))
            return np.array([current_z.real, current_z.imag, 0.0]) + center_offset

        # 画笔发光尖端
        pen_tip = Dot(radius=0.04, color=WHITE, fill_opacity=1.0)
        pen_tip.move_to(get_current_pen_tip_pos())

        # 激活左上角画中画特写 (Zoomed Camera)
        self.activate_zooming(animate=False)
        self.zoomed_display.to_corner(UL, buff=0.35)
        self.zoomed_display.display_frame.set_color(CONFIG["box_color"])
        self.zoomed_camera.frame.move_to(pen_tip.get_center())
        self.zoomed_camera.frame.add_updater(
            lambda frame: frame.move_to(pen_tip.get_center())
        )

        # 第 1 幕初始圆盘淡入
        current_static_epicycles = self.build_epicycle_vgroup(
            active_state["coeffs"], active_state["freqs"], 0.0, center_offset
        )
        self.add(pen_tip)
        self.play(FadeIn(current_static_epicycles), run_time=0.8)

        # 依次演绎 4 幕（1、2、3、4 个圆盘）
        for stage_idx in range(len(STAGES)):
            time_tracker.set_value(0.0)

            # 移除静态过渡圆盘，挂载实时动态圆盘与轨迹
            self.remove(current_static_epicycles)

            dynamic_epicycles = always_redraw(
                lambda: self.build_epicycle_vgroup(
                    active_state["coeffs"],
                    active_state["freqs"],
                    time_tracker.get_value(),
                    center_offset,
                )
            )

            drawn_path = VMobject(
                stroke_color=CONFIG["path_color"],
                stroke_width=2.6,
            )

            def update_drawn_path(mob):
                t = time_tracker.get_value()
                idx = max(1, int(t * (n_sub - 1)))
                pts = list(active_state["dense_path_3d"][: idx + 1])
                pts.append(get_current_pen_tip_pos())
                if len(pts) >= 2:
                    mob.set_points_as_corners(pts)

            drawn_path.add_updater(update_drawn_path)
            pen_tip.clear_updaters()
            pen_tip.add_updater(lambda mob: mob.move_to(get_current_pen_tip_pos()))

            self.add(dynamic_epicycles, drawn_path, pen_tip)

            # 完整旋转一周绘出当前摆线
            self.play(
                time_tracker.animate.set_value(1.0),
                run_time=CONFIG["stage_draw_duration"],
                rate_func=linear,
            )
            self.wait(1.0)

            # 锁定完成态静态快照，准备平滑过渡到下一幕
            drawn_path.clear_updaters()
            pen_tip.clear_updaters()
            end_epicycles = self.build_epicycle_vgroup(
                active_state["coeffs"], active_state["freqs"], 1.0, center_offset
            )
            self.remove(dynamic_epicycles)
            self.add(end_epicycles)

            # 如果还有下一幕，执行圆盘平滑变形 + 旧轨迹优雅淡出 + 画笔滑移就位
            if stage_idx + 1 < len(STAGES):
                update_active_stage_data(stage_idx + 1)
                time_tracker.set_value(0.0)
                next_start_epicycles = self.build_epicycle_vgroup(
                    active_state["coeffs"], active_state["freqs"], 0.0, center_offset
                )
                next_tip_pos = get_current_pen_tip_pos()

                self.play(
                    ReplacementTransform(end_epicycles, next_start_epicycles),
                    FadeOut(drawn_path),
                    pen_tip.animate.move_to(next_tip_pos),
                    run_time=CONFIG["transition_duration"],
                    rate_func=smooth,
                )
                current_static_epicycles = next_start_epicycles
                self.wait(0.3)
            else:
                # 最后一幕完成后停留展示
                self.wait(2.0)