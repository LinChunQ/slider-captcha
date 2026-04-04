import base64
import io
import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template

from slider_captcha import SliderCaptchaService

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB，允许上传大图片

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.json")


def _load_test_data():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _ndarray_to_b64(img):
    """将 numpy ndarray 转为带前缀的 base64 PNG 字符串"""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def _recognize(small_b64, big_b64, uuid=None, move_length=None):
    service = SliderCaptchaService(debug=False)
    res = service.recognize(small_b64, big_b64)
    if res is None:
        return {"success": False}

    final_img = res["result"].copy()
    x, y, w, h = int(res["best_x"]), res["band_y"], res["w_box"], res["h_box"]
    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    result_b64 = _ndarray_to_b64(final_img)

    accuracy = None
    if move_length is not None:
        big_w = res["result"].shape[1]
        predicted_ratio = res["best_x"] / big_w
        accuracy = round(1 - abs(predicted_ratio - move_length), 4)

    return {
        "success": True,
        "offset": int(res["best_x"]),
        "small_alpha": _ndarray_to_b64(res["small_alpha"]),
        "small_edge": _ndarray_to_b64(res["small_edge"]),
        "big_edge": _ndarray_to_b64(res["big_edge"]),
        "result": result_b64,
        "uuid": uuid,
        "moveLength": move_length,
        "accuracy": accuracy,
        "band_y": res["band_y"],
        "w_box": res["w_box"],
        "h_box": res["h_box"],
        "bigW": res["result"].shape[1],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/test-cases", methods=["GET"])
def get_test_cases():
    data = _load_test_data()
    return jsonify([
        {
            "uuid": item["uuid"],
            "moveLength": item["moveLength"],
            "success": item["success"],
        }
        for item in data
    ])


@app.route("/api/demo-preview", methods=["POST"])
def api_demo_preview():
    try:
        body = request.get_json()
        idx = int(body.get("index", 0))
        data_list = _load_test_data()

        if 0 <= idx < len(data_list):
            item = data_list[idx]
            big_b64 = item.get("bigImage", "")

            # 关键修正：如果 JSON 里的字符串没有前缀，手动加上
            if big_b64 and not big_b64.startswith("data:image"):
                big_b64 = "data:image/png;base64," + big_b64

            return jsonify({
                "success": True,
                "bigImage": big_b64
            })
        return jsonify({"success": False, "error": "Index out of range"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/demo-recognize", methods=["POST"])
def api_demo_recognize():
    try:
        body = request.get_json()
        idx = int(body.get("index", 0))
        data_list = _load_test_data()

        if 0 <= idx < len(data_list):
            item = data_list[idx]
            # 调用你之前的 _recognize 函数，它会返回完整的分析步骤图
            result = _recognize(
                item["smallImage"],
                item["bigImage"],
                uuid=item.get("uuid"),
                move_length=item.get("moveLength")
            )
            return jsonify(result)
        return jsonify({"success": False, "error": "Index out of range"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    small_b64 = body.get("smallImage", "").strip().replace("\n", "").replace("\r", "")
    big_b64 = body.get("bigImage", "").strip().replace("\n", "").replace("\r", "")

    if not small_b64 or not big_b64:
        return jsonify({"success": False, "error": "smallImage and bigImage are required"}), 400

    try:
        return jsonify(_recognize(small_b64, big_b64))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/custom-sample", methods=["GET"])
def api_custom_sample():
    """返回第一条测试数据的原始 base64，供自定义面板体验用"""
    data = _load_test_data()
    if not data:
        return jsonify({"success": False, "error": "No test data"}), 404
    item = data[0]
    small = item.get("smallImage", "")
    big = item.get("bigImage", "")
    if small and not small.startswith("data:image"):
        small = "data:image/png;base64," + small
    if big and not big.startswith("data:image"):
        big = "data:image/png;base64," + big
    return jsonify({"success": True, "smallImage": small, "bigImage": big})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
