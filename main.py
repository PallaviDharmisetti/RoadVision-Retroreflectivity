import cv2
import numpy as np
import os

# -----------------------------
# GLOBAL SMOOTHING
# -----------------------------
ri_history = []
ALPHA = 0.3


def fix_path(path):
    path = path.strip().strip('"')
    path = path.replace("\\", "/")
    return path


# -----------------------------
# WEATHER + NIGHT DETECTION (UPDATED)
# -----------------------------
def detect_environment(gray):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    if brightness < 60:
        return "NIGHT"
    elif lap_var < 40 and brightness < 120:
        return "FOG/MIST"
    elif lap_var < 60:
        return "RAINY"
    else:
        return "CLEAR"


# -----------------------------
# ROAD SEGMENTATION
# -----------------------------
def road_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 40])
    upper = np.array([180, 80, 220])

    mask = cv2.inRange(hsv, lower, upper)

    h, w = frame.shape[:2]
    roi = np.zeros_like(mask)

    polygon = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.75), int(h * 0.6)),
        (int(w * 0.25), int(h * 0.6))
    ]], dtype=np.int32)

    cv2.fillPoly(roi, polygon, 255)

    return cv2.bitwise_and(mask, roi)


# -----------------------------
# LANE DETECTION
# -----------------------------
def detect_lanes(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60,
                            minLineLength=60, maxLineGap=40)

    mask = np.zeros_like(gray)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-5)

            if abs(slope) < 0.6:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    return mask


# -----------------------------
# SIGNBOARD DETECTION
# -----------------------------
def detect_signboards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 800 < area < 25000:
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)

            if len(approx) == 4:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask


# -----------------------------
# ROAD STUD DETECTION
# -----------------------------
def detect_studs(gray):
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh


# -----------------------------
# REFLECTION INDEX
# -----------------------------
def reflection_index(gray, mask):
    pixels = gray[mask == 255]

    if len(pixels) == 0:
        return 0.0

    mean = np.mean(pixels)
    std = np.std(pixels)

    ri = (0.7 * mean + 0.3 * std) / 255 * 100
    return round(min(ri, 100), 2)


# -----------------------------
# SMOOTHING
# -----------------------------
def smooth(value):
    ri_history.append(value)

    if len(ri_history) > 10:
        ri_history.pop(0)

    smoothed = ri_history[0]

    for v in ri_history[1:]:
        smoothed = (ALPHA * v) + ((1 - ALPHA) * smoothed)

    return round(smoothed, 2)


# -----------------------------
# CONDITION + LIFE PREDICTION (NEW)
# -----------------------------
def predict_condition_and_life(ri, env):
    if ri < 30:
        condition = "CRITICAL WEAR"
        life = "0-1 month"
        urgency = "IMMEDIATE REPAIR"
    elif ri < 55:
        condition = "MODERATE WEAR"
        life = "1-3 months"
        urgency = "SCHEDULE SOON"
    else:
        condition = "GOOD CONDITION"
        life = "3-6 months"
        urgency = "NO URGENT ACTION"

    if env != "CLEAR":
        condition += " (ENV AFFECTED)"

    return condition, life, urgency


# -----------------------------
# CONFIDENCE
# -----------------------------
def confidence(mask):
    return int(np.sum(mask == 255) / (mask.size + 1e-5) * 100)


# -----------------------------
# PROCESS FRAME
# -----------------------------
def process(frame):
    frame = cv2.resize(frame, (900, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    env = detect_environment(gray)

    rmask = road_mask(frame)
    lanes = detect_lanes(gray)
    signs = detect_signboards(frame)
    studs = detect_studs(gray)

    road_features = cv2.bitwise_and(lanes, lanes, mask=rmask)

    if np.sum(road_features) == 0:
        road_features = rmask

    ri_raw = reflection_index(gray, road_features)
    ri = smooth(ri_raw)

    condition, life, urgency = predict_condition_and_life(ri, env)
    conf = confidence(road_features)

    output = frame.copy()

    overlay = np.zeros_like(frame)
    overlay[road_features == 255] = [0, 255, 255]

    output = cv2.addWeighted(output, 1, overlay, 0.5, 0)

    output[signs == 255] = [0, 0, 255]
    output[studs == 255] = [255, 255, 0]

    # TEXT (UPDATED)
    cv2.putText(output, f"Reflection Index: {ri}/100", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(output, f"Condition: {condition}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(output, f"Environment: {env}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(output, f"Remaining Life: {life}", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(output, f"Repair Urgency: {urgency}", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    cv2.putText(output, f"Confidence: {conf}%", (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


# -----------------------------
# MAIN
# -----------------------------
def main():
    path = input("Enter image/video/webcam: ")
    path = fix_path(path)

    if path.lower() == "webcam":
        cap = cv2.VideoCapture(0)

    elif path.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(path)
        if frame is None:
            print("Invalid image")
            return

        output = process(frame)
        cv2.imshow("RoadVision System", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    else:
        if not os.path.exists(path):
            print("File not found")
            return

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print("Cannot open video")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process(frame)
        cv2.imshow("RoadVision System", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()