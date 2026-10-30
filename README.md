# 傅里叶级数可视化

古希腊天文学，人们认为行星在做匀速圆周运动，然而有人观测到火星逆行的现象，当时人们为了解释这种现象发明了本轮和均轮。现在我们知道这套理论是错的，但重要的是这种本轮套均轮，均轮套均轮的方式能拟合任意封闭曲线，并且它和Fourier变换有着千丝万缕的联系。

在b站看了漫士的视频[《【漫士】半小时速通傅里叶变换：千年的圆上圆游戏》](https://www.bilibili.com/video/BV18Ma96JEoZ/?spm_id_from=333.337.search-card.all.click&vd_source=0d64b23e402c0efa7f8f76cef7f8cdb5)，自己简单整理了下思路，用manim做了一个可视化脚本。可以识别本地图片轮廓(过于复杂的图片最好让生图大模型先处理下，搞成类似Bad Apple那种轮廓清晰的图)，用这种方式绘制各种图形

## 1. 项目文件结构与核心功能

| 文件名称 | 核心功能 |
| :--- | :--- |
| `fourier_epicycles_drawing.py` | 主程序（Manim）。读取本地图片并提取轮廓，用一组旋转圆盘拟合并描绘出该图形，支持连笔线稿与实心剪影两种模式。 |
| `epicycles_progression.py` | 演示程序（Manim）。用 1~4 个圆盘依次画出圆、三尖内摆线、四叶花瓣线与五星星芒线。 |
| `ournotes.md` | 理论笔记。推导均轮本轮拟合与傅里叶级数之间的等价关系。 |
| `requirements.txt` | Python 依赖清单（manim、opencv-python、numpy 等）。 |
| `image_material/` | 示例素材图片（签名、剪影等）。 |
| `media/` | Manim 渲染输出的视频目录，运行后自动生成。 |

## 2. 环境配置指南 

本项目基于 **Python 3.9 ~ 3.12** 与 **Manim Community** 构建，依赖清单统一记录在同目录下的 `requirements.txt` 中。下面两种方式任选其一。

### 2.1 方式一：Conda（推荐，省去手动配置 C 库）

1. **激活或创建 Conda 虚拟环境**：
   ```bash
   conda create -n fourier python=3.10 -y
   conda activate fourier
   ```

2. **安装系统级图形库 Cairo 与视频编码器 FFmpeg**（强烈推荐通过 `conda-forge` 频道安装，自动配齐底层 C 动态链接库）：
   ```bash
   conda install -c conda-forge cairo pycairo ffmpeg -y
   ```

3. **安装 `requirements.txt` 中的 Python 依赖**：
   ```bash
   pip install -r requirements.txt
   ```

### 2.2 方式二：原生 Python + pip

不使用 Conda 时，用 `venv` 建虚拟环境、用 `pip` 装包，但需要自己先装好系统的 **FFmpeg**（视频编码必需）。

1. **创建并激活虚拟环境**：
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **安装系统级视频编码器 FFmpeg**（按平台任选一条）：
   ```bash
   # Windows（winget）
   winget install Gyan.FFmpeg
   
   # macOS（Homebrew）
   brew install ffmpeg
   
   # Ubuntu / Debian
   sudo apt install ffmpeg
   ```

3. **安装 `requirements.txt` 中的 Python 依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**（能打印出版本号即可）：
   ```bash
   manim --version
   ffmpeg -version
   ```

> 提示：Windows 上若 `pip install -r requirements.txt` 在编译 `pycairo` 时报缺少 Cairo C 库，直接改用 **3.1 的 Conda 方式**最省事；macOS / Linux 若报 `pango` 相关错误，用 `brew install pango` 或 `sudo apt install libpango1.0-dev` 补上即可。



## 3. 使用方法与运行指南 

### 3.1  `fourier_epicycles_drawing.py`)

示例图片放在 `image_material/` 目录下（如连笔签名 `qianming.jpg`、人物剪影 `dongfang.png`）。把 `CONFIG["image_path"]` 改成对应路径（如 `"image_material/qianming.jpg"`）后，在终端执行对应命令；若路径不存在，程序会自动回退为内置心形曲线演示。

* **低分辨率快速预览（480P 15fps）**：
  
  ```bash
  manim -pql fourier_epicycles_drawing.py FourierDrawingScene
  ```
* **1080P 全高清渲染导出（1920×1080 60fps）**：
  
  ```bash
  manim -pqh fourier_epicycles_drawing.py FourierDrawingScene
  ```
* **4K 超清渲染导出（3840×2160 60fps）**：
  
  ```bash
  manim -pqk fourier_epicycles_drawing.py FourierDrawingScene
  ```

渲染完成的 `.mp4` 视频文件会自动保存在当前目录下的 `media/videos/fourier_epicycles_drawing/` 对应分辨率文件夹中。

#### 核心配置项 (`CONFIG`) 调参说明

| 参数名 | 默认值 | 说明与调参建议 |
| :--- | :--- | :--- |
| `image_path` | `"qianming.jpg"` | 本地图片文件名或路径。若文件不存在，程序会自动回退为内置的高阶心形曲线演示。 |
| `extract_mode` | `"auto"` | • `"auto"`：根据背景与前景像素占比自动判断连笔线稿或实心剪影。<br>• `"line_art"`：强制启用连笔签名内外环无缝桥接模式。<br>• `"silhouette"`：强制启用实心剪影外轮廓提取模式。 |
| `num_vectors` | `150` | **圆盘（向量）总数量**：<br>• 高对称几何图案（如万花筒写轮眼）：设为 `8 ~ 16` 即可高度还原；<br>• 呆萌剪影 / 简笔画：推荐 `50 ~ 160`；<br>• 复杂连笔书法签名 / 动漫发丝肖像：推荐 `150 ~ 280`。 |
| `path_stroke_width` | `9.0` | **画笔轨迹线宽**：<br>• 设为 `2.5 ~ 3.0`：呈现细线轮廓描边效果；<br>• 设为 `6.5 ~ 9.5`：绘制连笔签名时直接将笔画两侧描边融合涂满，呈现饱满的**实心书法笔画**效果。 |
| `path_color` | `YELLOW_C` | 轨迹线条颜色（可使用 Manim 内置色值如 `YELLOW_C`、`TEAL_C`，或十六进制颜色如 `"#22d3ee"`）。 |
| `draw_duration` | `12.0` | 完整描绘一周的动画时长（秒），可根据圆盘复杂度在 `10.0 ~ 20.0` 之间调节。 |
| `zoom_factor` | `0.20` | 左上角画中画特写窗口的视口缩放比例（`0.20` 代表 5 倍局部放大，数值越小放大倍数越高）。 |
| `show_target_contour` | `False` | 是否在开场前预先显示半透明的目标轮廓参考底图（默认 `False` 保持全黑纯净画布）。 |

### 3.2 运行 1~4 圆盘递进摆线动画 (`epicycles_progression.py`)

该脚本无需外部图片，直接运行即可：

```bash
# 快速预览
manim -pql epicycles_progression.py EpicyclesProgressionScene

# 1080P 60帧高清导出
manim -pqh epicycles_progression.py EpicyclesProgressionScene
```

若需自定义每一幕的几何图案，直接编辑脚本中的 `STAGES` 列表即可，列表中每个元组格式为 `(半径 r, 转速 k, 初始相位 phi)`：

```python
STAGES = [
    {"name": "1 Disk: Circle", "epicycles": [(2.2, 1, 0.0)]},
    {"name": "2 Disks: Deltoid", "epicycles": [(1.6, 1, 0.0), (0.8, -2, 0.0)]},
    {"name": "3 Disks: 4-Cusp Floral", "epicycles": [(1.6, 1, 0.0), (0.65, -3, 0.0), (0.32, 5, 0.0)]},
    {"name": "4 Disks: Pentagram Star", "epicycles": [(1.55, 1, 0.0), (0.70, -4, 0.0), (0.32, 6, 0.0), (0.16, -9, 0.0)]},
]
```

