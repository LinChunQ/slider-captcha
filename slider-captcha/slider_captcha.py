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

        # 2. 小图处理：提取 Alpha 通道并定位有效内容
        # 这里的 alpha 就是滑块的形状蒙版
        alpha = small_img[:, :, 3]
        _, alpha_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        # 获取滑块在小图中的实际边界（去除四周多余透明像素）
        x_box, y_box, w_box, h_box = cv2.boundingRect(alpha_mask)

        # 3. 小图处理：灰度与边缘
        small_bgr = cv2.cvtColor(small_img, cv2.COLOR_BGRA2BGR)
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        # 裁剪出滑块核心区域
        tpl_roi = small_gray[y_box:y_box + h_box, x_box:x_box + w_box]
        tpl_edge = cv2.Canny(tpl_roi, 50, 150)
        # 对应的蒙版也要裁剪
        mask_roi = alpha[y_box:y_box + h_box, x_box:x_box + w_box]

        # 4. 大图处理：灰度、模糊与带状裁剪
        big_gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
        big_blur = cv2.GaussianBlur(big_gray, (3, 3), 0)

        big_h, big_w = big_img.shape[:2]
        band_y = max(0, min(y_box, big_h - 1))
        band_h = min(h_box, big_h - band_y)

        band_roi = big_blur[band_y:band_y + band_h, :]
        band_edge = cv2.Canny(band_roi, 50, 150)

        # 5. 匹配
        # 注意：这里我们只匹配裁剪后的 tpl_edge
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