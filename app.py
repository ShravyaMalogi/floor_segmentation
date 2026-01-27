from flask import (
    Flask,
    render_template,
    request,
    redirect,
    jsonify,
    send_from_directory
)
from PIL import Image
import os
import cv2
import numpy as np

from room_processing import *
from texture_mapping import (
    get_wall_corners,
    map_texture,
    load_texture,
    image_resize
)
from wall_segmentation.segmenation import wall_segmenting, build_model
from wall_estimation.estimation import wall_estimation


# --------------------------------------------------
# App init
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_FOLDER = os.path.join(BASE_DIR, "static", "IMG")
DATA_FOLDER = os.path.join(BASE_DIR, "static", "data")
TEXTURE_LIBRARY = os.path.join(BASE_DIR, "test_images", "textures")

ROOM_IMAGE = os.path.join(IMG_FOLDER, "room.jpg")
TEXTURED_ROOM_PATH = os.path.join(IMG_FOLDER, "textured_room.jpg")

MASK_PATH = os.path.join(DATA_FOLDER, "image_mask.npy")
CORNERS_PATH = os.path.join(DATA_FOLDER, "corners_estimation.npy")

os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# --------------------------------------------------
# Load model once
# --------------------------------------------------
model = build_model()


# --------------------------------------------------
# Home
# --------------------------------------------------
@app.route("/")
def main():
    return redirect("/room")


# --------------------------------------------------
# Upload room image + segmentation
# --------------------------------------------------
@app.route("/prediction", methods=["POST"])
def predict_image_room():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Load image safely
        img = Image.open(file.stream).convert("RGB")
        img_np = np.asarray(img)

        if img_np.shape[0] > 600:
            img_np = image_resize(img_np, height=600)

        Image.fromarray(img_np).save(ROOM_IMAGE)

        # ---- ML PIPELINE ----
        mask1 = wall_segmenting(model, ROOM_IMAGE)
        estimation_map = wall_estimation(ROOM_IMAGE)
        corners = get_wall_corners(estimation_map)

        mask2 = np.zeros(mask1.shape, dtype=np.uint8)
        for pts in corners:
            cv2.fillPoly(mask2, [np.array(pts)], 255)

        mask = mask1 & np.bool_(mask2)

        np.save(MASK_PATH, mask)
        np.save(CORNERS_PATH, np.array(corners))

        # Tell frontend to reload
        return jsonify({
            "status": "success",
            "reload": True
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"error": "Server error"}), 500


# --------------------------------------------------
# Room visualizer
# --------------------------------------------------
@app.route("/room")
def room():
    textures = [
        f for f in os.listdir(TEXTURE_LIBRARY)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if os.path.isfile(TEXTURED_ROOM_PATH):
        room_image = "/static/IMG/textured_room.jpg"
    elif os.path.isfile(ROOM_IMAGE):
        room_image = "/static/IMG/room.jpg"
    else:
        room_image = ""

    return render_template(
        "index.html",
        room=room_image,
        textures=textures,
        enable_textures=os.path.isfile(ROOM_IMAGE)
    )


# --------------------------------------------------
# Apply texture (CLICK)
# --------------------------------------------------
@app.route("/result_textured", methods=["POST"])
def result_textured():
    try:
        data = request.get_json()
        texture_name = os.path.basename(data.get("texture", ""))
        texture_path = os.path.join(TEXTURE_LIBRARY, texture_name)

        if not os.path.isfile(ROOM_IMAGE):
            return jsonify({"state": "error", "msg": "Upload room first"}), 400

        if not os.path.isfile(texture_path):
            return jsonify({"state": "error", "msg": "Texture not found"}), 400

        img = load_img(ROOM_IMAGE)
        corners = np.load(CORNERS_PATH)
        mask = np.load(MASK_PATH)

        texture = load_texture(texture_path, 6, 6)
        textured = map_texture(texture, img, corners, mask)
        out = brightness_transfer(img, textured, mask)

        save_image(out, TEXTURED_ROOM_PATH)

        return jsonify({
            "state": "success",
            "room_path": "/static/IMG/textured_room.jpg"
        })

    except Exception as e:
        print("TEXTURE ERROR:", e)
        return jsonify({"state": "error", "msg": "Texture processing failed"}), 500


# --------------------------------------------------
# Serve textures
# --------------------------------------------------
@app.route("/textures/<path:filename>")
def serve_texture(filename):
    return send_from_directory(TEXTURE_LIBRARY, filename)


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(port=9000, debug=True)
