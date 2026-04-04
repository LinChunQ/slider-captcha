# Slider Captcha Recognizer (OpenCV Edition)

## Demo

演示地址：http://118.145.160.23:5000/

这是一个基于 **OpenCV** 实现的高精度滑块验证码识别方案。该方案专门针对带有复杂内部纹路、干扰背景的滑块进行了优化，识别率极高且性能卓越。

## 🚀 核心功能 (Core Features)

为了解决传统边缘检测干扰多的问题，本项目采用了以下进阶策略：

1.  **Alpha 通道蒙版提取**：直接从小图的透明度通道（Alpha Channel）提取滑块轮廓。这种方法能 **100% 忽略** 滑块内部的任何彩色干扰贴图。
2.  **带状区域裁剪 (Band ROI)**：根据滑块在小图中的 Y 坐标，在大图中动态切出一个横向搜索带。这极大地缩小了搜索空间，不仅提升了速度，还彻底杜绝了画面顶部或底部的干扰误报。
3.  **局部双边滤波 (Bilateral Filter)**：在裁剪后的搜索带应用双边滤波。相比高斯模糊，它能在磨平背景细小杂色的同时，**锁死并强化** 缺口的锐利边缘。
4.  **形态学加粗 (Dilation)**：对边缘进行膨胀处理，增加特征重合度，使匹配算法对轻微的变形具有更强的鲁棒性。

## 🛠️ 处理流程可视化

项目的识别逻辑分为以下步骤：

  * **小图端**：`Alpha提取` -\> `Canny边缘` -\> `Dilation加粗`
  * **大图端**：`带状裁剪` -\> `双边滤波降噪` -\> `Canny边缘`
  * **匹配端**：`模板匹配 (MatchTemplate)` -\> `带掩码 (Mask) 运算`

-----

## 📦 快速开始

### 1\. 环境准备

确保已安装 `opencv-python` 和 `numpy`：

```bash
pip install opencv-python numpy
```

### 2\. 代码集成

将 `SliderCaptchaService` 类引入你的项目：

```python
from slider_captcha import SliderCaptchaService

service = SliderCaptchaService(debug=True)
# small_b64 和 big_b64 为图片的 Base64 编码字符串（支持带/不带 data:image 前缀）
result = service.recognize(small_b64, big_b64)

if result:
    print(f"识别成功！滑块应滑动的 X 坐标: {result['x']}")
```

## 📂 项目结构

  * `slider_captcha.py`: 核心识别逻辑封装类。
  * `main.py`: 测试脚本，支持批量从 JSON 加载数据并实时可视化调试。
  * `debug_output/`: 开启 `debug=True` 后，程序会自动保存每一步的处理过程图（如 `small_edge.png`, `band_edge.png`），方便调优。

## ⚙️ 调优参数建议

如果在特定场景下识别偏离，可以尝试微调以下参数：

  * **`cv2.Canny(..., 30, 150)`**: 调整高低阈值来控制边缘灵敏度。
  * **`cv2.bilateralFilter(..., d=9, sigmaColor=75, ...)`**: 增加 `d` 值可增强降噪能力，但会消耗更多 CPU。
  * **`band_y` 偏移量**: 在切片时可以给 `band_y` 增加一些上下浮动（Padding），以应对某些前端渲染导致的坐标偏移。

-----

**注意**：本工具仅用于自动化测试及学习研究，请勿用于任何违反服务条款或法律法规的活动。
