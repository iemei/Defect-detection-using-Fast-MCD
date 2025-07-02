import cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

# ------------------------------------------------------------------
# Parameters you may tweak
# ------------------------------------------------------------------
AXIS_THRESHOLD = 25.0       # ratio ⇒ linear
SUPPORT_FRAC   = 0.75       # FAST‑MCD support
CHI_P_FLAT     = 1/(8)      # threshold p for flat (1 / 8HW)
CHI_P_LINEAR   = 1/(4)      # threshold p for linear (1 / 4HW)

# ------------------------------------------------------------------
# Helper: post‑processing (open + close + hole‑fill)
# ------------------------------------------------------------------
def post_process(mask, open_r=3, close_r=5):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_r+1,)*2)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1,)*2)
    m  = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_OPEN,  k1)
    m  = cv2.morphologyEx(m,                             cv2.MORPH_CLOSE, k2)
    # fill holes
    inv = cv2.bitwise_not(m)
    flood = inv.copy()
    ffmask = np.zeros((inv.shape[0]+2, inv.shape[1]+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0,0), 255)
    holes = cv2.bitwise_not(flood)
    final = cv2.bitwise_or(m, holes)
    return final.astype(bool)

# ------------------------------------------------------------------
# Helper: dominant‑line background for linear images
# ------------------------------------------------------------------
def build_background_linear(gray, orientation):
    h, w = gray.shape
    bg = np.zeros_like(gray, np.float32)
    if orientation == "vertical":
        for x in range(w):
            col = gray[:, x].astype(np.float32)
            med = np.median([col[0], col[-1], col.mean()])
            bg[:, x] = med
    else:                                           # horizontal
        for y in range(h):
            row = gray[y, :].astype(np.float32)
            med = np.median([row[0], row[-1], row.mean()])
            bg[y, :] = med
    return bg.astype(np.uint8)

# ------------------------------------------------------------------
# Linear branch
# ------------------------------------------------------------------
def process_linear(gray):
    h, w = gray.shape
    # dominant direction via Sobel moment tensor
    gx = cv2.Sobel(gray.astype(np.float32)/255.0, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray.astype(np.float32)/255.0, cv2.CV_64F, 0, 1, 3)
    Ixx, Iyy, Ixy = np.sum(gx**2), np.sum(gy**2), np.sum(gx*gy)
    M = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eig(M)
    struct_vec = eigvecs[:, np.argmin(eigvals)]
    orientation = "vertical" if abs(struct_vec[0]) < abs(struct_vec[1]) else "horizontal"

    bg   = build_background_linear(gray, orientation)
    flat = gray.astype(np.int16) - bg.astype(np.int16)

    # FAST‑MCD on residual
    mcd   = MinCovDet(support_fraction=SUPPORT_FRAC, random_state=0).fit(flat.reshape(-1,1))
    mahal = mcd.mahalanobis(flat.reshape(-1,1)).reshape(h, w)
    thr   = chi2.ppf(1 - CHI_P_LINEAR/(h*w), df=1)
    raw   = mahal > thr
    final = post_process(raw)

    return bg, flat, raw, final

# ------------------------------------------------------------------
# Flat branch
# ------------------------------------------------------------------
def process_flat(gray):
    h, w = gray.shape
    mcd   = MinCovDet(support_fraction=SUPPORT_FRAC, random_state=0).fit(gray.reshape(-1,1))
    mahal = mcd.mahalanobis(gray.reshape(-1,1)).reshape(h, w)
    thr   = chi2.ppf(1 - CHI_P_FLAT/(h*w), df=1)
    raw   = mahal > thr
    final = post_process(raw, open_r=1, close_r=3)
    return None, None, raw, final    # background/flat unused

# ------------------------------------------------------------------
# Classification + full pipeline
# ------------------------------------------------------------------
def classify_and_detect(gray):
    # Axis ratio from Sobel moment tensor
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    Ixx, Iyy, Ixy = np.sum(gx**2), np.sum(gy**2), np.sum(gx*gy)
    M = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    evals,_ = np.linalg.eig(M)
    axis_ratio = evals.max() / (evals.min() + 1e-8)

    if axis_ratio > AXIS_THRESHOLD:
        img_type = "linear"
        bg, flat, raw, mask = process_linear(gray)
    else:
        img_type = "flat"
        bg, flat, raw, mask = process_flat(gray)

    return img_type, bg, flat, raw, mask

# ------------------------------------------------------------------
# Demo / test
# ------------------------------------------------------------------
if __name__ == "__main__":
    # === HARD‑CODE your test image here ===
    IMG_PATH = r"C:\Users\iemei\OneDrive\Pictures\Python figures\Ex5.png"
    # IMG_PATH = r"C:\Users\iemei\OneDrive\Pictures\Python figures\Ex8.png"
    gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(IMG_PATH)

    typ, bg, flat, raw, final = classify_and_detect(gray)

    # Overlay contours
    contours, _ = cv2.findContours((final*255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, -1, (0,0,255), 2)

    # --- Display ---
    titles = ["Input", "Background", "Flattened", "Raw mask", "Final mask", "Overlay"]
    imgs   = [gray,
              bg if bg is not None else np.zeros_like(gray),
              flat if flat is not None else np.zeros_like(gray),
              raw.astype(np.uint8)*255,
              final.astype(np.uint8)*255,
              overlay[...,::-1]]

    plt.figure(figsize=(12,8))
    plt.suptitle(f"Image type detected: {typ}", fontsize=16)
    for i,(t,im) in enumerate(zip(titles,imgs)):
        ax = plt.subplot(2,3,i+1)
        ax.imshow(im, cmap='gray')
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
