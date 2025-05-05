from ultralytics import YOLO
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
from shapely.geometry import Polygon

# --- Utility functions ---

def order_points(pts):
    # Orders 4 points: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    if not poly_1.is_valid or not poly_2.is_valid:
        return 0.0
    inter = poly_1.intersection(poly_2).area
    union = poly_1.union(poly_2).area
    return inter / union if union > 0 else 0.0

# --- Pipeline steps ---

def detect_corners(image_path):
    model = YOLO("models/corners-best-7k.pt")
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    boxes = results[0].boxes
    arr = boxes.xywh.cpu().numpy()
    points = arr[:, 0:2]
    corners = order_points(points)
    return corners

def four_point_transform(image_path, pts):
    img = Image.open(image_path)
    image = asarray(img)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # Display the warped image
    plt.figure(figsize=(8, 8))
    plt.imshow(warped)
    plt.title("Perspective Transformed Image")
    plt.axis('off')
    plt.show()
    return Image.fromarray(warped)

def plot_grid_on_transformed_image(image):
    # image: PIL Image
    w, h = image.size
    corners = np.array([[0,0], [w,0], [w,h], [0,h]])
    corners = order_points(corners)
    figure(figsize=(10, 10), dpi=80)
    plt.imshow(image)
    TL, TR, BR, BL = corners
    def interpolate(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / 8
        dy = (y1 - y0) / 8
        return [(x0 + i*dx, y0 + i*dy) for i in range(9)]
    ptsT = interpolate(TL, TR)
    ptsL = interpolate(TL, BL)
    ptsR = interpolate(TR, BR)
    ptsB = interpolate(BL, BR)
    for a, b in zip(ptsL, ptsR):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")
    for a, b in zip(ptsT, ptsB):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")
    plt.axis('off')
    plt.title("Grid Overlay on Transformed Image")
    plt.savefig('chessboard_transformed_with_grid.jpg')
    plt.show()
    plt.close()
    return ptsT, ptsL

def chess_pieces_detector(image):
    # image: PIL Image
    model = YOLO("models/piece-recog-ankita.pt")
    # Save to temp file for YOLO input
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        results = model.predict(source=tmp.name, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy()
    # Display detected pieces
    img_np = np.array(image)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    for box in detections:
        x1, y1, x2, y2 = box[:4]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='lime', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title("Detected Pieces")
    plt.axis('off')
    plt.show()
    return detections, boxes

def connect_square_to_detection(detections, boxes, square):
    di = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r',
          6: 'B', 7: 'K', 8: 'N', 9: 'P', 10: 'Q', 11: 'R'}
    list_of_iou = []
    for i in detections:
        box_x1, box_y1, box_x2, box_y2 = i[0], i[1], i[2], i[3]
        # cut high pieces
        if box_y2 - box_y1 > 60:
            box_complete = np.array([
                [box_x1, box_y1+40], [box_x2, box_y1+40],
                [box_x2, box_y2], [box_x1, box_y2]
            ])
        else:
            box_complete = np.array([
                [box_x1, box_y1], [box_x2, box_y1],
                [box_x2, box_y2], [box_x1, box_y2]
            ])
        list_of_iou.append(calculate_iou(box_complete, square))
    if not list_of_iou or max(list_of_iou) <= 0.15:
        return "1"
    num = int(np.argmax(list_of_iou))
    piece = int(boxes.cls[num].item())
    return di.get(piece, "1")

# --- Main pipeline ---

def main():
    image_path = 'images/aug_0_0.jpg'
    corners = detect_corners(image_path)
    transformed_image = four_point_transform(image_path, corners)
    ptsT, ptsL = plot_grid_on_transformed_image(transformed_image)
    detections, boxes = chess_pieces_detector(transformed_image)

    # Calculate grid points
    x = [ptsT[i][0] for i in range(9)]
    y = [ptsL[i][1] for i in range(9)]
    # Build all 64 squares
    FEN_annotation = []
    for row in range(8):
        line = []
        for col in range(8):
            square = np.array([
                [x[col], y[row]],
                [x[col+1], y[row]],
                [x[col+1], y[row+1]],
                [x[col], y[row+1]]
            ])
            line.append(square)
        FEN_annotation.append(line)
    # Detect pieces for each square
    board_FEN = []
    for line in FEN_annotation:
        line_to_FEN = []
        for square in line:
            piece_on_square = connect_square_to_detection(detections, boxes, square)
            line_to_FEN.append(piece_on_square)
        # Replace empty with '1'
        corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
        print(corrected_FEN)
        board_FEN.append(corrected_FEN)
    complete_board_FEN = [''.join(line) for line in board_FEN]
    to_FEN = '/'.join(complete_board_FEN)
    print("https://lichess.org/analysis/" + to_FEN)

if __name__ == "__main__":
    main()
