"""
3Blue1Brown 风格：均轮与本轮 (Fourier Epicycles) 拟合任意图像轮廓与连笔签名
支持：
1. 黑底白线连笔签名图（如生成的“阿斌”一笔签）：自动通过『最小跨度无缝桥接算法』串联内外回环，零跳线完美还原书法笔画；
2. 人物剪影图（如 Bad Apple 博丽灵梦）：自动通过『智能背景泛洪剥离 (smartBg)』提取纯净外轮廓；
3. 2400 点超采样丝滑轨迹绘制，彻底消除低帧率 (-pql 15fps) 下的折线撕裂感。

运行方式：
  manim -pql fourier_epicycles_drawing.py FourierDrawingScene      # 低分辨率快速预览
  manim -pqh fourier_epicycles_drawing.py FourierDrawingScene      # 1080P 高清渲染
"""

import os
import cv2
import numpy as np
from manim import *

# ==========================================
# 1. 核心参数配置
# ==========================================
CONFIG = {
    # 待拟合的图像路径（可填入保存的 "abin.jpg" 或 "dongfang.png"）
    "image_path": "naiwa_black.png",
    
    # 提取模式: "auto" (自动识别连笔线稿或实心剪影) | "line_art" (连笔签名/空心线稿) | "silhouette" (实心剪影)
    "extract_mode": "auto",
    
    # 智能背景剥离色差容差（针对实心剪影模式，默认 26）
    "color_tolerance": 26,
    
    # 圆盘（向量）总数量：连笔签名建议 220 ~ 300 个，细节极其丝滑
    "num_vectors": 50,
    
    # 动画完整绘制一周的时长（秒）
    "draw_duration": 10.0,
    
    # 轮廓等弧长重采样点数（设为 1500 点，充分保留连笔转折细节）
    "num_samples": 1500,
    
    # 轨迹超采样细分点数（解决 Manim -pql 15帧/秒下采样不足导致的折线撕裂问题）
    "path_subsamples": 2400,
    
    # 是否显示半透明目标参考轮廓底图
    "show_target_contour": False,
    
    # 画面布局与缩放比例
    "target_height": 5.2,          # 轮廓在 Manim 画面中的最大尺寸
    "center_offset": RIGHT * 1.5,  # 整体向右平移，为左上角画中画留出空间
    
    # 局部画中画 (Zoomed Inset) 配置
    "zoom_factor": 0.20,           # 放大倍数倒数（0.20 对应 5 倍局部特写）
    "display_width": 4.6,          # 画中画窗口宽度
    "display_height": 3.0,         # 画中画窗口高度
    "box_color": TEAL_C,           # 3Blue1Brown 标志性青蓝色取景框
    "path_color": "YELLOW_C",       # 流光青色 (也可改为 YELLOW_C)
}


# ==========================================
# 2. 图像轮廓提取（支持连笔签名无缝桥接 + 智能背景剥离）与 DFT
# ==========================================
class ContourFourierAnalyzer:
    """
    自适应轮廓提取器：
    - 针对黑底白线连笔签名（如“阿斌”一笔签）：采用『最小跨度无缝桥接算法』，在笔画宽度最近点（~10px）处将字内回环无缝嵌入主回路，杜绝任何跨画布飞线；
    - 针对实心剪影（如 Bad Apple）：采用『四角采样 + 边框泛洪 + 5x5 闭运算』提取单一外轮廓。
    """

    @staticmethod
    def _extract_line_art_bridged(gray: np.ndarray, bg_is_white: bool) -> np.ndarray:
        """提取连笔签名/线稿的所有内外回环，并在最近笔画宽度处无缝桥接为单条连续闭合曲线"""
        h, w = gray.shape
        work_gray = gray.copy()
        # 自动屏蔽生图右下角可能存在的微小水印区域
        if not bg_is_white:
            work_gray[int(h * 0.88):, int(w * 0.88):] = 0
            _, binary = cv2.threshold(work_gray, 100, 255, cv2.THRESH_BINARY)
        else:
            work_gray[int(h * 0.88):, int(w * 0.88):] = 255
            _, binary = cv2.threshold(work_gray, 150, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        min_len = max(30.0, 0.02 * (h + w))
        valid_contours = [
            c.reshape(-1, 2).astype(np.float64)
            for c in contours
            if cv2.arcLength(c, True) > min_len
        ]
        if not valid_contours:
            return None

        valid_contours.sort(key=lambda c: len(c), reverse=True)
        main_loop = valid_contours[0]
        remaining = valid_contours[1:]

        # 迭代将每个内部镂空回环在距离最近处（仅隔一个笔画宽度）桥接并入主回路
        while remaining:
            best_dist = float("inf")
            best_rem_idx = -1
            best_main_pt_idx = -1
            best_sub_pt_idx = -1

            step_m = max(1, len(main_loop) // 800)
            main_sample = main_loop[::step_m]

            for r_idx, sub_c in enumerate(remaining):
                step_s = max(1, len(sub_c) // 300)
                sub_sample = sub_c[::step_s]
                diff = main_sample[:, None, :] - sub_sample[None, :, :]
                dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
                min_flat = np.argmin(dists)
                m_i, s_i = np.unravel_index(min_flat, dists.shape)
                d = dists[m_i, s_i]
                if d < best_dist:
                    best_dist = d
                    best_rem_idx = r_idx
                    best_main_pt_idx = m_i * step_m
                    best_sub_pt_idx = s_i * step_s

            sub_c = remaining.pop(best_rem_idx)
            # 若离主轮廓过远（可能是孤立噪点），则跳过
            if best_dist > 0.08 * max(h, w):
                continue

            sub_ordered = np.vstack([sub_c[best_sub_pt_idx:], sub_c[: best_sub_pt_idx + 1]])
            main_loop = np.vstack([
                main_loop[: best_main_pt_idx + 1],
                sub_ordered,
                main_loop[best_main_pt_idx:],
            ])

        return main_loop

    @staticmethod
    def _extract_silhouette_smart(img: np.ndarray) -> np.ndarray:
        """网页版同款 smartBg 智能背景泛洪剥离算法（适合实心人物剪影）"""
        max_dim = 280
        h0, w0 = img.shape[:2]
        scale_img = min(max_dim / w0, max_dim / h0, 1.0)
        w = int(round(w0 * scale_img))
        h = int(round(h0 * scale_img))
        img_small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        if img_small.ndim == 3 and img_small.shape[2] == 4 and np.min(img_small[:, :, 3]) < 200:
            mask = (img_small[:, :, 3] > 25).astype(np.uint8) * 255
        else:
            bgr = (
                cv2.cvtColor(img_small, cv2.COLOR_GRAY2BGR).astype(np.float32)
                if img_small.ndim == 2
                else img_small[:, :, :3].astype(np.float32)
            )
            sz = min(8, w // 4, h // 4)
            corners = np.vstack([
                bgr[:sz, :sz].reshape(-1, 3),
                bgr[:sz, w - sz :].reshape(-1, 3),
                bgr[h - sz :, :sz].reshape(-1, 3),
                bgr[h - sz :, w - sz :].reshape(-1, 3),
            ])
            bg_color = np.median(corners, axis=0)
            tol = CONFIG["color_tolerance"]
            dist = np.linalg.norm(bgr - bg_color, axis=2)
            bg_candidate = (dist <= (tol + 8)).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(bg_candidate, connectivity=4)
            border_labels = set(
                np.concatenate([labels[0, :], labels[h - 1, :], labels[:, 0], labels[:, w - 1]])
            )
            border_labels.discard(0)
            is_flood_bg = np.isin(labels, list(border_labels))
            mask = (~is_flood_bg).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        closed_mask[:2, :] = 0
        closed_mask[-2:, :] = 0
        closed_mask[:, :2] = 0
        closed_mask[:, -2:] = 0

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest_c = max(contours, key=cv2.contourArea)
        return largest_c.reshape(-1, 2).astype(np.float64)

    @staticmethod
    def get_contour_points(image_path: str, num_samples: int = 1500) -> np.ndarray:
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                h, w = gray.shape
                corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
                bg_is_white = np.mean(corners) > 127

                # 判断前景像素占比：若前景占比很小 (< 22%) 或黑底白线，说明是连笔签名/线稿！
                fg_ratio = (
                    np.mean(gray < 127) if bg_is_white else np.mean(gray > 127)
                )
                mode = CONFIG.get("extract_mode", "auto")
                is_line_art = (mode == "line_art") or (mode == "auto" and (not bg_is_white or fg_ratio < 0.22))

                if is_line_art:
                    raw_pts = ContourFourierAnalyzer._extract_line_art_bridged(gray, bg_is_white)
                    mode_name = "连笔签名内外环无缝桥接模式"
                else:
                    raw_pts = ContourFourierAnalyzer._extract_silhouette_smart(img)
                    mode_name = "智能背景剥离剪影模式"

                if raw_pts is not None and len(raw_pts) > 15:
                    print(f"\n[Fourier] ✅ 已启用『{mode_name}』，成功提取连续闭合轨迹 ({len(raw_pts)} 点)")

                    if np.linalg.norm(raw_pts[0] - raw_pts[-1]) > 1e-3:
                        raw_pts = np.vstack([raw_pts, raw_pts[0]])

                    # 等弧长均匀插值重采样
                    diffs = np.diff(raw_pts, axis=0)
                    dists = np.hypot(diffs[:, 0], diffs[:, 1])
                    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
                    total_len = cum_dist[-1]

                    if total_len > 0:
                        target_dists = np.linspace(0, total_len, num_samples, endpoint=False)
                        sx = np.interp(target_dists, cum_dist, raw_pts[:, 0])
                        sy = np.interp(target_dists, cum_dist, raw_pts[:, 1])

                        sy = -sy
                        sx -= np.mean(sx)
                        sy -= np.mean(sy)

                        scale = CONFIG["target_height"] / max(np.ptp(sx), np.ptp(sy))
                        return (sx * scale) + 1j * (sy * scale)

        print(f"\n[Fourier] ⚠️ 未找到图片 '{image_path}'，当前使用默认心形轮廓演示。")
        t = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        sx = 16 * np.sin(t) ** 3
        sy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        sx -= np.mean(sx)
        sy -= np.mean(sy)
        scale = CONFIG["target_height"] / max(np.ptp(sx), np.ptp(sy))
        return (sx * scale) + 1j * (sy * scale)

    @staticmethod
    def compute_dft_coefficients(points_complex: np.ndarray, num_vectors: int):
        N = len(points_complex)
        coeffs = np.fft.fft(points_complex) / N
        freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)

        c0_idx = np.where(freqs == 0)[0][0]
        other_indices = [i for i in range(N) if i != c0_idx]
        other_indices.sort(key=lambda i: np.abs(coeffs[i]), reverse=True)

        active_count = min(num_vectors, N)
        selected_indices = [c0_idx] + other_indices[: active_count - 1]

        return coeffs[selected_indices], freqs[selected_indices]


# ==========================================
# 3. Manim 可视化场景类 (超采样丝滑轨迹 + 画中画特写)
# ==========================================
class FourierDrawingScene(ZoomedScene):
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

    def construct(self):
        # 1. 提取轮廓点集并计算傅里叶系数
        complex_points = ContourFourierAnalyzer.get_contour_points(
            CONFIG["image_path"],
            num_samples=CONFIG["num_samples"],
        )
        coeffs, freqs = ContourFourierAnalyzer.compute_dft_coefficients(
            complex_points,
            num_vectors=CONFIG["num_vectors"],
        )

        center_offset = CONFIG["center_offset"]

        # 2. 绘制半透明目标参考轮廓底图
        if CONFIG["show_target_contour"]:
            target_pts_3d = [
                np.array([z.real, z.imag, 0.0]) + center_offset
                for z in complex_points
            ]
            target_pts_3d.append(target_pts_3d[0])
            ref_contour = VMobject(
                stroke_color=GREY_B,
                stroke_width=1.2,
                stroke_opacity=0.22,
            ).set_points_as_corners(target_pts_3d)
            self.add(ref_contour)

        # 3. 预计算 2400 个超采样平滑轨迹点
        n_sub = CONFIG["path_subsamples"]
        t_dense = np.linspace(0.0, 1.0, n_sub, endpoint=True)
        exp_matrix = np.exp(1j * 2 * np.pi * np.outer(freqs, t_dense))
        z_dense = np.dot(coeffs, exp_matrix)
        dense_path_3d = np.column_stack([
            z_dense.real + center_offset[0],
            z_dense.imag + center_offset[1],
            np.zeros(n_sub),
        ])

        # 4. 时间进度追踪器 (0 -> 1)
        time_tracker = ValueTracker(0.0)

        def get_current_pen_tip_pos():
            t = time_tracker.get_value()
            current_z = np.sum(coeffs * np.exp(1j * 2 * np.pi * freqs * t))
            return np.array([current_z.real, current_z.imag, 0.0]) + center_offset

        pen_tip = Dot(radius=0.035, color=WHITE, fill_opacity=1.0)
        pen_tip.add_updater(lambda mob: mob.move_to(get_current_pen_tip_pos()))

        # 5. 超采样丝滑轨迹线
        drawn_path = VMobject(
            stroke_color=CONFIG["path_color"],
            stroke_width=2.6,
        )

        def update_drawn_path(mob):
            t = time_tracker.get_value()
            idx = max(1, int(t * (n_sub - 1)))
            pts = list(dense_path_3d[: idx + 1])
            pts.append(get_current_pen_tip_pos())
            if len(pts) >= 2:
                mob.set_points_as_corners(pts)

        drawn_path.add_updater(update_drawn_path)

        # 6. 动态圆盘与旋转向量臂
        def create_epicycles():
            t = time_tracker.get_value()
            group = VGroup()

            curr_pos = np.array([coeffs[0].real, coeffs[0].imag, 0.0]) + center_offset

            for i, (c, k) in enumerate(zip(coeffs[1:], freqs[1:])):
                radius = np.abs(c)
                angle = np.angle(c) + 2 * np.pi * k * t
                vec = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
                next_pos = curr_pos + vec

                if radius > 0.008:
                    circle_opacity = max(0.08, 0.42 - (i / len(coeffs)) * 0.32)
                    circle = Circle(
                        radius=radius,
                        stroke_color=GOLD_B if i == 0 else GREY_B,
                        stroke_width=1.0,
                        stroke_opacity=circle_opacity,
                    ).move_to(curr_pos)
                    group.add(circle)

                if radius > 0.06:
                    arrow = Arrow(
                        start=curr_pos,
                        end=next_pos,
                        buff=0,
                        stroke_width=max(1.0, min(2.2, radius * 2.5)),
                        max_tip_length_to_length_ratio=min(0.25, 0.12 / max(radius, 0.06)),
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

        epicycles = always_redraw(create_epicycles)
        self.add(epicycles, drawn_path, pen_tip)

        # 7. 激活左上角画中画特写 (Zoomed Camera)
        self.activate_zooming(animate=False)
        self.zoomed_display.to_corner(UL, buff=0.35)
        self.zoomed_display.display_frame.set_color(CONFIG["box_color"])

        self.zoomed_camera.frame.add_updater(
            lambda frame: frame.move_to(pen_tip.get_center())
        )

        # 8. 渲染完整周期动画
        self.wait(0.5)
        self.play(
            time_tracker.animate.set_value(1.0),
            run_time=CONFIG["draw_duration"],
            rate_func=linear,
        )
        self.wait(2.0)