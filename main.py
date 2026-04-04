import json
import cv2
from slider_captcha import SliderCaptchaService


def run_full_debug():
    service = SliderCaptchaService(debug=True)
    with open("test_data.json", 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    for i, item in enumerate(data_list):
        res = service.recognize(item["smallImage"], item["bigImage"], index=i)

        if res:
            # 准备结果图绘制
            final_img = res["result"]
            x, y, w, h = int(res["best_x"]), res["band_y"], res["w_box"], res["h_box"]
            cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            print("大图中缺口距离：",res["x"])

            print(f"[{i + 1}] 展示全流程... 空格/回车下一组，ESC退出")

            # --- 展示小图处理过程 ---
            cv2.imshow("A. Small Alpha Mask", res["small_alpha"])
            cv2.imshow("B. Small Canny Edge", res["small_edge"])

            # --- 展示大图处理过程 ---
            cv2.imshow("C. Big Band Edge", res["big_edge"])
            cv2.imshow("D. Final Result", final_img)

            # 窗口布局（根据你的屏幕分辨率调整坐标）
            cv2.moveWindow("A. Small Alpha Mask", 100, 100)  # 左上
            cv2.moveWindow("B. Small Canny Edge", 400, 100)  # 中上
            cv2.moveWindow("C. Big Band Edge", 100, 300)  # 左下
            cv2.moveWindow("D. Final Result", 100, 500)  # 最下

            key = cv2.waitKey(0)
            if key == 27: break
        else:
            print(f"[{i + 1}] 识别失败")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_full_debug()