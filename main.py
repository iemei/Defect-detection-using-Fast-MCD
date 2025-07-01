# Re-import required libraries after kernel reset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


# Re-load the image file
img_path = r'C:\Users\iemei\OneDrive\Pictures\Python figures\Ex9.png'
gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Redefine helper and pipeline functions
def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def remove_border_blobs(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    cleared = mask.copy().astype(np.uint8)
    inv = (cleared ^ 1).astype(np.uint8)
    flood = inv.copy()
    ffmask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    cleared[flood == 255] = 0
    return cleared.astype(bool)

def post_process(mask: np.ndarray, open_r: int = 3, close_r: int = 5) -> np.ndarray:
    m = (mask * 255).astype(np.uint8)
    if open_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_r+1,)*2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1,)*2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cleared = remove_border_blobs(m.astype(bool))
    inv = (~cleared).astype(np.uint8)
    flood = inv.copy()
    ffmask = np.zeros((inv.shape[0]+2, inv.shape[1]+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    holes = (flood == 0)
    return cleared | holes

def build_defect_free_image(gray: np.ndarray, orientation: str) -> np.ndarray:
    h, w = gray.shape
    result = np.zeros_like(gray, dtype=np.float32)
    if orientation == "vertical":
        for x in range(w):
            col = gray[:, x].astype(np.float32) / 255.0
            median_val = np.median([col[0], col[-1], np.mean(col)])
            result[:, x] = median_val
    else:
        for y in range(h):
            row = gray[y, :].astype(np.float32) / 255.0
            median_val = np.median([row[0], row[-1], np.mean(row)])
            result[y, :] = median_val
    return (result * 255).astype(np.uint8)

def fast_mcd_linear_with_overlay(gray: np.ndarray, axis_threshold: float = 25.0):
    out = {}
    gray_f = gray.astype(np.float32) / 255.0
    out['input'] = gray

    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, 3)
    Ixx, Iyy, Ixy = np.sum(gx**2), np.sum(gy**2), np.sum(gx*gy)
    M = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eig(M)
    axis_ratio = eigvals.max() / (eigvals.min() + 1e-8)

    struct_vec = eigvecs[:, np.argmin(eigvals)]
    out['axis_ratio'] = axis_ratio

    vertical = abs(struct_vec[0]) < abs(struct_vec[1])
    orientation = "vertical" if vertical else "horizontal"

    h, w = gray.shape
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    length = max(h, w)
    sx, sy = struct_vec
    pt1 = (int(w/8 - length*sx), int(h/8 - length*sy))
    pt2 = (int(w/8 + length*sx), int(h/8 + length*sy))
    cv2.line(overlay, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    out['dominant_line'] = overlay

    background = build_defect_free_image(gray, orientation)
    out['background'] = background

    flat = gray_f - (background.astype(np.float32) / 255.0)
    flat_vis = np.clip((flat - flat.min()) / (flat.max() - flat.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
    out['flat'] = flat_vis

    flat1d = flat.reshape(-1, 1)
    mcd = MinCovDet(support_fraction=0.75, random_state=0).fit(flat1d)
    mahal2 = mcd.mahalanobis(flat1d).reshape(h, w)
    chi_thr = chi2.ppf(1 - 1/(4*h*w), df=1)
    raw_mask = mahal2 > chi_thr
    out['raw'] = raw_mask

    final_mask = post_process(raw_mask)
    out['final'] = final_mask

    raw_mask_u8 = (final_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(raw_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(outlined_img, contours, -1, (0, 0, 255), thickness=2)
    out['outline_overlay'] = outlined_img

    return out

# Run integrated pipeline
final_results = fast_mcd_linear_with_overlay(gray_img)

# Visualise
imgs = [
    final_results['input'],
    cv2.cvtColor(final_results['dominant_line'], cv2.COLOR_BGR2RGB),
    final_results['background'],
    final_results['flat'],
    #final_results['raw'],
    final_results['final'],
    cv2.cvtColor(final_results['outline_overlay'], cv2.COLOR_BGR2RGB)
]
titles = [
    "Input image",
    "Dominant line",
    "Defect‑free background",
    "Flattened residual",
    #"Raw mask",
    "Final mask",
    "Outlined defect"
]
cmaps = ['gray', None, 'gray', 'gray', 'gray', 'gray', None]

plt.figure(figsize=(18, 10))
for idx, (im, title, cmap) in enumerate(zip(imgs, titles, cmaps)):
    ax = plt.subplot(2, 3, idx+1)
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
