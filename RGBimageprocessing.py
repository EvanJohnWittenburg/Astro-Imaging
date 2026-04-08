import os
import numpy as np
import gc
from astropy.io import fits
from scipy.ndimage import zoom, shift
from scipy.signal import correlate2d
import cv2
from collections import Counter


data_folder = r'C:\Users\EvanW\Desktop\Astro7'
FAST_PREVIEW = False
RGB_LOG_SCALE = 2000
LUM_LOG_SCALE = 3000
LUMINANCE_BLEND_ALPHA = 0.2


def read_fits_folder(path):
    return [fits.getdata(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.fits')]

def create_master_frame(frames):
    return np.median(np.stack(frames), axis=0)

def calibrate_images(images, master_bias, master_dark):
    calibrated = []
    for img in images:
        if master_bias.shape != img.shape:
            scale_y = img.shape[0] / master_bias.shape[0]
            scale_x = img.shape[1] / master_bias.shape[1]
            bias_resized = zoom(master_bias, (scale_y, scale_x), order=1)
            dark_resized = zoom(master_dark, (scale_y, scale_x), order=1)
        else:
            bias_resized = master_bias
            dark_resized = master_dark
        calibrated.append(img - bias_resized - dark_resized)
    return calibrated

def stack_images(images):
    return np.nan_to_num(np.median(np.stack(images), axis=0), nan=0.0, posinf=0.0, neginf=0.0)

def align_images(reference, target):
    reference = reference.astype(np.float32)
    target = target.astype(np.float32)
    result = cv2.matchTemplate(reference, target, method=cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    shift_y = max_loc[1] - (reference.shape[0] - target.shape[0]) // 2
    shift_x = max_loc[0] - (reference.shape[1] - target.shape[1]) // 2
    return shift(target, shift=(shift_y, shift_x), mode='nearest')

def log_stretch(image, scale=1000):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0)
    return np.log1p(image * scale) / np.log1p(scale)

def normalize(image, stretch=True):
    image = np.nan_to_num(image, nan=0.0)
    low = np.percentile(image, 0.5)
    high = np.percentile(image, 99.5)
    if high - low < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    image = np.clip(image, low, high)
    image = image - low
    image = image / (high - low)
    if stretch:
        image = image ** 0.8
    return (image * 255).astype(np.uint8)

def save_as_bmp(image, filename):
    norm = normalize(image, stretch=True)
    cv2.imwrite(filename, norm)
    if filename.endswith('.bmp'):
        cv2.imwrite(filename.replace('.bmp', '.png'), norm)

def estimate_offset(ref, target, box_size=50):
    h, w = ref.shape
    min_std = np.inf
    best_box = (0, 0)
    for y in range(0, h - box_size, box_size):
        for x in range(0, w - box_size, box_size):
            ref_patch = ref[y:y+box_size, x:x+box_size]
            if np.std(ref_patch) < min_std:
                min_std = np.std(ref_patch)
                best_box = (y, x)
    y, x = best_box
    ref_val = np.median(ref[y:y+box_size, x:x+box_size])
    target_val = np.median(target[y:y+box_size, x:x+box_size])
    return ref_val - target_val

def create_rgb_image(r, g, b):
    r = normalize(r)
    g = normalize(g)
    b = normalize(b)
    return np.dstack((r, g, b))

def apply_luminance_overlay(rgb, luminance, alpha=0.5):
    rgb = rgb.astype(np.float32)
    lum = normalize(luminance, stretch=True).astype(np.float32)
    lrgb = rgb * (1 - alpha) + np.expand_dims(lum, axis=2) * alpha
    return np.clip(lrgb, 0, 255).astype(np.uint8)


bias_frames = read_fits_folder(os.path.join(data_folder, 'bias'))
dark_frames = read_fits_folder(os.path.join(data_folder, 'dark'))
master_bias = create_master_frame(bias_frames)
master_dark = create_master_frame(dark_frames)

stacked = {}
for f in ['red', 'green', 'blue']:
    folder = os.path.join(data_folder, f)
    raw_images = read_fits_folder(folder)
    images = calibrate_images(raw_images, master_bias, master_dark)
    stacked[f] = stack_images(images)
    gc.collect()
    stacked[f] = zoom(stacked[f], 2, order=0 if FAST_PREVIEW else 1)

folder = os.path.join(data_folder, 'luminance')
raw_images = read_fits_folder(folder)
images = calibrate_images(raw_images, master_bias, master_dark)
shapes = [img.shape for img in images]
shape_counts = Counter(shapes)
common_shape = shape_counts.most_common(1)[0][0]
images = [img for img in images if img.shape == common_shape]
stacked['luminance'] = stack_images(images)

for color in ['red', 'green', 'blue']:
    stacked[color] = align_images(stacked['luminance'], stacked[color])

stacked['green'] *= 1.15
stacked['blue'] *= 2.50
stacked['green'] += estimate_offset(stacked['red'], stacked['green'])
stacked['blue'] += estimate_offset(stacked['red'], stacked['blue'])

log_scaled = {c: log_stretch(stacked[c], scale=RGB_LOG_SCALE) for c in ['red', 'green', 'blue']}
l_scaled = log_stretch(stacked['luminance'], scale=LUM_LOG_SCALE)

save_as_bmp(log_scaled['red'], 'R.bmp')
save_as_bmp(log_scaled['green'], 'G.bmp')
save_as_bmp(log_scaled['blue'], 'B.bmp')
save_as_bmp(l_scaled, 'L.bmp')

rgb_image = create_rgb_image(log_scaled['red'], log_scaled['green'], log_scaled['blue'])
lrgb_image = apply_luminance_overlay(rgb_image, l_scaled, alpha=LUMINANCE_BLEND_ALPHA)

filename = 'LRGB_preview.bmp' if FAST_PREVIEW else 'LRGB_final.bmp'
cv2.imwrite(filename, lrgb_image)
cv2.imwrite(filename.replace('.bmp', '.png'), lrgb_image)
