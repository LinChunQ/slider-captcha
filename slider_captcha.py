import cv2
import numpy as np
import base64
import os


class SliderCaptchaService:
    def __init__(self, debug=True):
        self.debug = debug
        self.debug_dir = "debug_output"
        if self.debug and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def _save_debug(self, group_index, step_name, img):
        """保存中间过程图，文件名包含组索引和步骤名"""
        if self.debug and img is not None:
            filename = f"group_{group_index}_{step_name}.png"
            path = os.path.join(self.debug_dir, filename)
            cv2.imwrite(path, img)

    def _decode_image(self, input_data, flags):
        if isinstance(input_data, str):
            if input_data.startswith("data:image"):
                input_data = input_data.split(",")[1]
            img_data = base64.b64decode(input_data)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, flags)
        return None

    def recognize(self, small_b64, big_b64, index=0):
        # 1. 解码 (UNCHANGED 才能拿到 Alpha 通道)
        small_img = self._decode_image(small_b64, cv2.IMREAD_UNCHANGED)
        big_img = self._decode_image(big_b64, cv2.IMREAD_COLOR)
        if small_img is None or big_img is None: return None

        # 如果小图没有 Alpha 通道（如 JPG），自动补全为不透明
        if small_img.ndim == 2:
            # 灰度图 → BGRA
            small_img = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGRA)
        elif small_img.shape[2] == 3:
            # BGR → BGRA，alpha 全部设为 255（不透明）
            small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2BGRA)
            small_img[:, :, 3] = 255

        # 2. 小图处理：提取 Alpha 通道并定位有效内容
        # 这里的 alpha 就是滑块的形状蒙版
        alpha = small_img[:, :, 3]
        _, alpha_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        # 获取滑块在小图中的实际边界（去除四周多余透明像素）
        x_box, y_box, w_box, h_box = cv2.boundingRect(alpha_mask)

        # 3. 小图处理优化
        small_bgr = cv2.cvtColor(small_img, cv2.COLOR_BGRA2BGR)
        # 裁剪出蒙版和灰度图
        mask_roi = alpha[y_box:y_box + h_box, x_box:x_box + w_box]

        # 核心技巧：结合 Alpha 蒙版生成的边缘，完全无视内部贴图
        tpl_edge = cv2.Canny(mask_roi, 30, 150)
        # 加粗边缘，增加匹配容错
        kernel = np.ones((3, 3), np.uint8)
        tpl_edge = cv2.dilate(tpl_edge, kernel, iterations=1)

        # 4. 大图处理：灰度与带状裁剪（先切再滤）
        big_gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
        big_h, big_w = big_img.shape[:2]
        band_y = max(0, y_box)
        band_h_val = min(big_h - band_y, h_box)

        # 如果带状区域比模板小，扩大到���图高度
        if band_h_val < h_box or big_w < w_box:
            band_y = 0
            band_h_val = big_h

        # 第一步：先裁剪出带状区域（此时是原始灰度，保留了所有细节）
        band_roi_raw = big_gray[band_y:band_y + band_h_val, :]

        # 第二步：在带状区域应用双边滤波 (Bilateral Filter)
        band_smooth = cv2.bilateralFilter(band_roi_raw, d=9, sigmaColor=75, sigmaSpace=75)

        # 第三步：对平滑后的区域进行边缘检测
        band_edge = cv2.Canny(band_smooth, 50, 150)

        # 可选优化：对边缘进行一次形态学膨胀
        kernel = np.ones((3, 3), np.uint8)
        band_edge = cv2.dilate(band_edge, kernel, iterations=1)

        # 5. 匹配（确保模板不大于搜索图）
        th, tw = tpl_edge.shape[:2]
        bh, bw = band_edge.shape[:2]
        if th > bh or tw > bw:
            # 模板比搜索区域大，无法匹配
            best_x = 0
        else:
            res = cv2.matchTemplate(band_edge, tpl_edge, cv2.TM_SQDIFF_NORMED, mask=mask_roi)
            _, _, min_loc, _ = cv2.minMaxLoc(res)
            best_x = min_loc[0]

        # 6. 返回结果，包含小图的处理过程
        return {
            "x": best_x,
            "small_alpha": alpha_mask,  # 小图蒙版
            "small_edge": tpl_edge,  # 小图边缘
            "big_edge": band_edge,  # 大图带状边缘
            "result": big_img.copy(),
            "band_y": band_y,
            "w_box": w_box,
            "h_box": h_box,
            "best_x": best_x
        }
