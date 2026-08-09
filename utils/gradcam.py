import numpy as np
from PIL import Image, ImageFilter


def _jet(values):
    values = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam(model, image, transform):
    """
    Lightweight explanation map for Vercel.

    This is an occlusion-sensitivity heatmap rather than gradient Grad-CAM.
    Each highlighted region shows where covering the image caused the
    predicted class confidence to drop the most.
    """
    base = image.convert("RGB").resize((224, 224))
    original = transform(base)
    target_index, original_probs = model.predict(original)
    original_score = float(original_probs[target_index])

    # 7x7 = 49 model calls; fast enough for a small ResNet18.
    patch = 32
    stride = 32
    drops = np.zeros((7, 7), dtype=np.float32)

    # Use the image mean as a neutral occlusion value in pixel space.
    neutral = np.full((patch, patch, 3), 0.5, dtype=np.float32)
    base_arr = np.asarray(base, dtype=np.float32) / 255.0

    for row in range(7):
        for col in range(7):
            x0 = col * stride
            y0 = row * stride
            x1 = min(x0 + patch, 224)
            y1 = min(y0 + patch, 224)

            occluded = base_arr.copy()
            occluded[y0:y1, x0:x1] = neutral[:y1-y0, :x1-x0]

            occluded_img = Image.fromarray(
                np.clip(occluded * 255, 0, 255).astype(np.uint8)
            )

            _, probs = model.predict(transform(occluded_img))
            drops[row, col] = max(
                0.0,
                original_score - float(probs[target_index])
            )

    # Smooth and normalize the coarse map.
    maximum = float(drops.max())
    if maximum > 1e-8:
        drops /= maximum

    heat_small = Image.fromarray(
        np.clip(drops * 255, 0, 255).astype(np.uint8),
        mode="L"
    )
    heat = heat_small.resize((224, 224), Image.Resampling.BICUBIC)
    heat = np.asarray(heat, dtype=np.float32) / 255.0

    heat_rgb = _jet(heat)
    original_rgb = np.asarray(base, dtype=np.float32) / 255.0

    visualization = (
        0.58 * original_rgb +
        0.42 * heat_rgb
    ) * 255.0

    return np.clip(
        visualization, 0, 255
    ).astype(np.uint8)
