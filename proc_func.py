import numpy as np
import math

def convolve(img, kernel):
    rws, cols, channels = img.shape
    kernel1 = kernel.copy()
    img1 = img.copy()
    rws1, cols1 = kernel.shape
    for r in range(0, rws1 // 2):
        for c in range(0, cols1):
            kernel1[r][c], kernel1[rws1 - r - 1][c] = kernel1[rws1 - r - 1][c], kernel1[r][c]
    for c in range(0, cols1 // 2):
            for r in range(0, rws1):
                kernel1[r][c], kernel1[r][cols1 - c - 1] = kernel1[r][cols1 - c - 1], kernel1[r][c]
    for ch in range(0, channels):
        for r in range(0, rws):
            for c in range(0, cols):
                sum = 0
                for dr in range(-(rws1 // 2), (rws1 // 2) + 1):
                    for dc in range(-(cols1 // 2), (cols1 // 2) + 1):
                        if r + dr >= 0 and r + dr < rws and c + dc >= 0 and c + dc < cols:
                            sum += kernel1[(rws1 // 2) + dr][(cols1 // 2) + dc] * img[r + dr][c + dc][ch]
                img1[r][c][ch] = int(max(min(sum, 255), 0))
    return img1

def median_blurring(img, sz):
    img1 = img.copy()
    rws, cols, channels = img.shape
    for ch in range(0, channels):
        for r in range(0, rws):
            for c in range(0, cols):
                vals = []
                for dr in range(-(sz // 2), (sz // 2) + 1):
                    for dc in range(-(sz // 2), (sz // 2) + 1):
                        if r + dr >= 0 and r + dr < rws and c + dc >= 0 and c + dc < cols:
                            vals.append(img[r + dr][c + dc][ch])
                        else:
                            vals.append(0)
                vals.sort()
                img1[r][c][ch] = vals[len(vals) // 2]
    return img1

def get_gaussian_weight(x, y, sigma):
    var = 2 * (sigma ** 2)
    norm = 1 / (math.pi * var)
    exp = -(x ** 2 + y ** 2) / var
    return norm * math.exp(exp)

def gaussian_blurring(img, sz = 9, sigma = 1.4):
    img1 = img.copy()
    img2 = img.copy()
    rws, cols, channels = img.shape
    kernel = np.zeros(sz)
    for di in range(-(sz // 2), (sz // 2) + 1):
        kernel[(sz // 2) + di] = get_gaussian_weight(abs(di), 0, sigma)
    sum1 = 0
    for x in kernel:
        sum1 += x
    for idx, x in enumerate(kernel):
        kernel[idx] = x / sum1
    for ch in range(0, channels):
        for r in range(0, rws):
            for c in range(0, cols):
                sum = 0
                for dr in range(0, 1):
                    for dc in range(-(sz // 2), (sz // 2) + 1):
                        if r + dr >= 0 and r + dr < rws and c + dc >= 0 and c + dc < cols:
                            sum += kernel[(sz // 2) + dc] * img[r + dr][c + dc][ch]
                sum = int(sum)
                img1[r][c][ch] = max(sum, min(sum, 255))
    for ch in range(0, channels):
            for r in range(0, rws):
                for c in range(0, cols):
                    sum = 0
                    for dr in range(-(sz // 2), (sz // 2) + 1):
                        for dc in range(0, 1):
                            if r + dr >= 0 and r + dr < rws and c + dc >= 0 and c + dc < cols:
                                sum += kernel[(sz // 2) + dr] * img1[r + dr][c + dc][ch]
                    sum = int(sum)
                    img2[r][c][ch] = max(0, min(sum, 255))
    return img2

def laplacian_edge_detection(img):
    img1 = gaussian_blurring(img)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    img2 = img1.copy()
    rws, cols, channels = img1.shape
    for ch in range(0, channels):
        for r in range(0, rws):
            for c in range(0, cols):
                sum = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if r + dr >= 0 and r + dr < rws and c + dc >= 0 and c + dc < cols:
                            sum += kernel[1 + dr][1 + dc] * img1[r + dr][c + dc][ch]
                img2[r][c][ch] = max(min(sum, 255), 0)
    return img2

def unsharp_masking(img, amp_factor):
    img1 = gaussian_blurring(img)
    rws, cols, channels = img1.shape
    img2 = np.zeros((rws, cols, channels), dtype=np.float64)
    for ch in range(0, channels):
        for r in range(0, rws):
            for c in range(0, cols):
                x = min(max(0, int(int(img[r][c][ch]) + amp_factor * (int(img[r][c][ch]) - int(img1[r][c][ch])))), 255)
                img2[r][c][ch] = x
    return img2.astype(np.uint8)