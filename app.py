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

# Import the new tiling functions
from floor_texture_tiling import apply_tiled_texture_to_floor, tile_texture


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
# Configuration for texture tiling
# --------------------------------------------------
DEFAULT_TILE_SIZE = (100, 100)  # Width, Height in pixels


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

        return jsonify({
            "status": "success",
            "reload": True
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"error": "Server error"}), 500


# --------------------------------------------------
# Apply texture with TILING (NEW!)
# --------------------------------------------------
@app.route("/apply_texture", methods=["POST"])
def apply_texture():
    """
    Apply a tiled texture to the floor.
    This uses repeating tile patterns instead of stretching/warping.
    """
    try:
        data = request.get_json()
        texture_name = data.get("texture")
        tile_size = data.get("tile_size", DEFAULT_TILE_SIZE)
        
        if not texture_name:
            return jsonify({"error": "No texture specified"}), 400
        
        # Validate files exist
        if not os.path.isfile(ROOM_IMAGE):
            return jsonify({"error": "No room image found. Upload a room first."}), 400
        
        if not os.path.isfile(MASK_PATH):
            return jsonify({"error": "No floor mask found. Process room image first."}), 400
        
        # Load room image and floor mask
        room_img = cv2.imread(ROOM_IMAGE)
        floor_mask = np.load(MASK_PATH)
        
        # Convert boolean mask to uint8 if needed
        if floor_mask.dtype == bool:
            floor_mask = floor_mask.astype(np.uint8) * 255
        
        # Load texture
        texture_path = os.path.join(TEXTURE_LIBRARY, texture_name)
        if not os.path.isfile(texture_path):
            return jsonify({"error": f"Texture not found: {texture_name}"}), 404
        
        texture = cv2.imread(texture_path)
        if texture is None:
            return jsonify({"error": "Failed to load texture"}), 500
        
        # Apply tiled texture to floor
        print(f"Applying tiled texture: {texture_name} with tile size: {tile_size}")
        result = apply_tiled_texture_to_floor(
            room_img, 
            floor_mask, 
            texture,
            tile_size=tuple(tile_size) if isinstance(tile_size, list) else tile_size
        )
        
        # Save result
        cv2.imwrite(TEXTURED_ROOM_PATH, result)
        
        return jsonify({
            "status": "success",
            "textured_room": "/static/IMG/textured_room.jpg",
            "message": f"Texture applied with repeating tile pattern"
        })
    
    except Exception as e:
        print("TEXTURE APPLICATION ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
    )


# --------------------------------------------------
# Static file serving
# --------------------------------------------------
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


# --------------------------------------------------
# Tile size configuration (optional)
# --------------------------------------------------
@app.route("/set_tile_size", methods=["POST"])
def set_tile_size():
    """Allows frontend to configure tile size dynamically"""
    try:
        data = request.get_json()
        tile_width = data.get("width", 100)
        tile_height = data.get("height", 100)
        
        global DEFAULT_TILE_SIZE
        DEFAULT_TILE_SIZE = (tile_width, tile_height)
        
        return jsonify({
            "status": "success",
            "tile_size": DEFAULT_TILE_SIZE
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
