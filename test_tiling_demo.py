"""
Example script demonstrating how to use floor_texture_tiling.py
to apply repeating tile patterns to floor segments.
"""

import cv2
import numpy as np
from floor_texture_tiling import apply_tiled_texture_to_floor, tile_texture


def test_tiling_simple():
    """
    Test 1: Simple tiling without floor segmentation
    """
    print("=" * 60)
    print("TEST 1: Simple Texture Tiling")
    print("=" * 60)
    
    # Create a simple texture pattern (checkerboard)
    texture = np.array([
        [255, 0, 255, 0],
        [0, 255, 0, 255],
        [255, 0, 255, 0],
        [0, 255, 0, 255]
    ], dtype=np.uint8)
    
    # Tile it to create a larger pattern
    tile_size = (5, 5)  # Repeat 5 times in each direction
    tiled = tile_texture(texture, tile_size)
    
    print(f"Original texture size: {texture.shape}")
    print(f"Tiled texture size: {tiled.shape}")
    print(f"✓ Texture tiled successfully!")
    
    # Save result
    cv2.imwrite("test_tiled_pattern.png", tiled)
    print("✓ Saved to: test_tiled_pattern.png\n")

def test_with_floor_mask():
    """
    Test 2: Apply tiled texture to a floor region using a mask
    """
    print("=" * 60)
    print("TEST 2: Tiled Texture with Floor Mask")
    print("=" * 60)
    
    # Create a sample floor image (gray rectangle)
    floor_img = np.zeros((400, 600), dtype=np.uint8)
    floor_img[200:350, 100:500] = 128  # Floor region
    
    # Create a tile pattern
    tile_pattern = np.array([
        [200, 150, 200, 150],
        [150, 200, 150, 200],
        [200, 150, 200, 150],
        [150, 200, 150, 200]
    ], dtype=np.uint8)
    
    # Apply tiled texture to floor
    tile_size = (100, 150)  # Repeat to cover floor
    result = apply_tiled_texture_to_floor(floor_img, tile_pattern, tile_size)
    
    print(f"Floor image size: {floor_img.shape}")
    print(f"✓ Tiled texture applied to floor region!")
    
    # Save result
    cv2.imwrite("test_floor_with_tiles.png", result)
    print("✓ Saved to: test_floor_with_tiles.png\n")

def test_with_actual_image():
    """
    Test 3: Apply texture to an actual room image with floor segmentation
    """
    print("=" * 60)
    print("TEST 3: Real Room Image with Tiled Floor Texture")
    print("=" * 60)
    
    # You would load your actual room image and floor mask here
    # Example:
    # room_img = cv2.imread('room.jpg')
    # floor_mask = cv2.imread('floor_mask.png', cv2.IMREAD_GRAYSCALE)
    
    # For demo, create synthetic data
    room_img = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
    floor_mask = np.zeros((400, 600), dtype=np.uint8)
    floor_mask[250:380, 50:550] = 255  # Floor area
    
    # Load a texture (or create one)
    # texture = cv2.imread('wood_tile.jpg')
    # For demo, create a simple wood-like pattern
    texture = np.random.randint(120, 180, (50, 50, 3), dtype=np.uint8)
    
    print(f"Room image size: {room_img.shape}")
    print(f"Floor mask size: {floor_mask.shape}")
    print(f"Texture size: {texture.shape}")
    
    # Note: This is a placeholder for the actual implementation
    # You would integrate with your floor segmentation model here
    
    print("✓ Test setup complete!")
    print("\nTo apply texture to real images:")
    print("1. Load room image: cv2.imread('room.jpg')")
    print("2. Get floor mask from your segmentation model")
    print("3. Load texture: cv2.imread('tile_texture.jpg')")
    print("4. Call apply_tiled_texture_to_floor()")
    print("5. Save result: cv2.imwrite('result.jpg', result)\n")

def demo_tile_sizes():
    """
    Test 4: Demonstrate different tile sizes
    """
    print("=" * 60)
    print("TEST 4: Different Tile Sizes")
    print("=" * 60)
    
    # Create a base texture
    base_texture = np.array([
        [255, 128, 255, 128],
        [128, 255, 128, 255],
        [255, 128, 255, 128],
        [128, 255, 128, 255]
    ], dtype=np.uint8)
    
    tile_configs = [
        (2, 2, "small"),
        (5, 5, "medium"),
        (10, 10, "large")
    ]
    
    for rows, cols, name in tile_configs:
        tiled = tile_texture(base_texture, (rows, cols))
        filename = f"test_tiles_{name}.png"
        cv2.imwrite(filename, tiled)
        print(f"✓ {name.capitalize()} tiles ({rows}x{cols}): {tiled.shape} - saved to {filename}")
    
    print()


if __name__ == "__main__":
    print("\n🎨 Floor Texture Tiling - Test Suite")
    print("=" * 60)
    print("This script demonstrates how to use repeating tile patterns")
    print("when applying textures to floor segments.")
    print("=" * 60)
    print() 
    
    # Run tests
    test_tiling_simple()
    test_with_floor_mask()
    demo_tile_sizes()
    test_with_actual_image()
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the generated test images")
    print("2. Integrate floor_texture_tiling.py into your Flask app")
    print("3. Replace the texture mapping logic to use tiling")
    print("4. Test with real room images and floor masks")
    print()