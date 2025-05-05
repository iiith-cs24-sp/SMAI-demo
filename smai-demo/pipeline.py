import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

# --- Constants ---
CHESSBOARD_SIZE = 8  # Number of squares on each side of a chessboard
SQUARE_SZ = 800  # Size of warped board
GRID_SIZE = 10  # Number of grid cells (including padding)
GRID_POINTS = GRID_SIZE + 1  # Number of points on grid (11 points for 10 cells)
CORNER_DETECTION_CONF_THRESHOLD = 0.08  # Confidence threshold for corner detection
PIECE_DETECTION_CONF_THRESHOLD = 0.25  # Confidence threshold for piece detection
PIECE_DETECTION_IOU_THRESHOLD = 0.45  # IoU threshold for piece detection
PIECE_DETECTION_IMG_SIZE = 640  # Image size for piece detection inference
PIECE_HEIGHT_THRESHOLD = 60  # Threshold for considering a piece "high"
PIECE_HEIGHT_ADJUSTMENT = 40  # Adjustment for high pieces
SQUARE_PIECE_IOU_THRESHOLD = 0.15  # Minimum IoU to consider a piece on a square
PADDING_START_INDEX = 1  # Index where padding ends and board begins
PADDING_END_INDEX = 9  # Index where board ends and padding begins
PLOT_LINEWIDTH = 2  # Line width for plotting

# --- Utility functions ---


def order_points(pts):
    """Order four points into TL,TR,BR,BL as in the training code"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(s)],  # TL
            pts[np.argmin(diff)],  # TR
            pts[np.argmax(s)],  # BR
            pts[np.argmax(diff)],  # BL
        ],
        dtype="float32",
    )


def warp_bbox(bbox, M):
    """Warp a bounding-box under perspective M (same as training code)"""
    # bbox = [x1,y1,x2,y2]
    pts = np.array(
        [
            [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
            ]
        ],
        dtype="float32",
    )
    warped = cv2.perspectiveTransform(pts, M)[0]
    xs, ys = warped[:, 0], ys[:, 1]
    return xs.min(), ys.min(), xs.max(), ys.max()


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
    print("Loading corner detection model...")
    model_path = os.path.abspath(os.path.join("models", "corners-best-7k.pt"))
    if not os.path.exists(model_path):
        print(f"ERROR: Corner model not found at {model_path}")
        return None

    model = YOLO(model_path)
    print(f"Detecting corners in {image_path}...")
    results = model.predict(
        source=image_path, conf=CORNER_DETECTION_CONF_THRESHOLD, verbose=True
    )

    # Visualize corner detection results
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("Original image with detected corners")

    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No corners detected!")
        return None

    arr = boxes.xywh.cpu().numpy()
    points = arr[:, 0:2]
    # Draw detected corners
    for x, y in points:
        plt.plot(x, y, "ro", markersize=10)

    corners = order_points(points)
    # Draw ordered corners with labels
    for i, (x, y) in enumerate(corners):
        plt.plot(x, y, "go", markersize=15)
        plt.text(x, y, str(i), fontsize=12, color="white")

    plt.axis("off")
    plt.show()
    return corners


def four_point_transform(image_path, pts):
    """Transform using the same padding approach as in training"""
    if pts is None:
        print("Cannot perform transformation without corner points")
        return None

    # Constants matching the training preprocessing
    pad_cells = 1
    pad_px = SQUARE_SZ // CHESSBOARD_SIZE * pad_cells
    out_sz = SQUARE_SZ + 2 * pad_px

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Source corners (from corner detection)
    src = order_points(pts)

    # Destination quad with positive padding (exactly as in training)
    dst = np.array(
        [
            [pad_px, pad_px],  # TL
            [SQUARE_SZ + pad_px, pad_px],  # TR
            [SQUARE_SZ + pad_px, SQUARE_SZ + pad_px],  # BR
            [pad_px, SQUARE_SZ + pad_px],  # BL
        ],
        dtype="float32",
    )

    # Perform perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_sz, out_sz))

    # Display the warped image
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Perspective Transformed Image (with padding)")
    plt.axis("off")
    plt.show()

    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def plot_grid_on_transformed_image(image):
    if image is None:
        print("Cannot plot grid on a None image")
        return None, None

    w, h = image.size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners = order_points(corners)
    figure(figsize=(10, 10), dpi=80)
    plt.imshow(image)
    plt.title("Grid Overlay on Transformed Image (10x10 with padding)")

    TL, TR, BR, BL = corners

    def interpolate(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / GRID_SIZE
        dy = (y1 - y0) / GRID_SIZE
        return [
            (x0 + i * dx, y0 + i * dy) for i in range(GRID_POINTS)
        ]  # 11 points for 10 cells

    ptsT = interpolate(TL, TR)
    ptsL = interpolate(TL, BL)
    ptsR = interpolate(TR, BR)
    ptsB = interpolate(BL, BR)

    for a, b in zip(ptsL, ptsR):
        plt.plot([a[0], b[0]], [a[1], b[1]], "r-", linestyle="--")
    for a, b in zip(ptsT, ptsB):
        plt.plot([a[0], b[0]], [a[1], b[1]], "r-", linestyle="--")

    # Mark the actual chess area (8x8 within the padding)
    plt.plot(
        [ptsT[PADDING_START_INDEX][0], ptsT[PADDING_END_INDEX][0]],
        [ptsT[PADDING_START_INDEX][1], ptsT[PADDING_END_INDEX][1]],
        "b-",
        linewidth=PLOT_LINEWIDTH,
    )
    plt.plot(
        [ptsB[PADDING_START_INDEX][0], ptsB[PADDING_END_INDEX][0]],
        [ptsB[PADDING_START_INDEX][1], ptsB[PADDING_END_INDEX][1]],
        "b-",
        linewidth=PLOT_LINEWIDTH,
    )
    plt.plot(
        [ptsL[PADDING_START_INDEX][0], ptsL[PADDING_END_INDEX][0]],
        [ptsL[PADDING_START_INDEX][1], ptsL[PADDING_END_INDEX][1]],
        "b-",
        linewidth=PLOT_LINEWIDTH,
    )
    plt.plot(
        [ptsR[PADDING_START_INDEX][0], ptsR[PADDING_END_INDEX][0]],
        [ptsR[PADDING_START_INDEX][1], ptsR[PADDING_END_INDEX][1]],
        "b-",
        linewidth=PLOT_LINEWIDTH,
    )

    plt.axis("off")
    plt.savefig("chessboard_transformed_with_grid.jpg")
    plt.show()

    return ptsT, ptsL


def chess_pieces_detector(image):
    if image is None:
        print("Cannot detect pieces on a None image")
        return None, None

    print("Loading chess piece detection model...")
    model_path = os.path.abspath(os.path.join("models", "piece-recog-ankita.pt"))
    if not os.path.exists(model_path):
        print(f"ERROR: Piece detection model not found at {model_path}")
        print(f"Looking for: {model_path}")
        return None, None

    model = YOLO(model_path)

    # Save the PIL image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        print(f"Saving image to temp file: {tmp.name}")
        image.save(tmp.name)
        print("Running piece detection with settings from training...")
        # Match training parameters
        results = model.predict(
            source=tmp.name,
            conf=PIECE_DETECTION_CONF_THRESHOLD,  # confidence threshold
            iou=PIECE_DETECTION_IOU_THRESHOLD,  # IoU threshold
            imgsz=PIECE_DETECTION_IMG_SIZE,  # image size for inference
            verbose=True,
        )

    if results is None or len(results) == 0:
        print("No results from piece detection!")
        return None, None

    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No pieces detected!")
        return None, None

    print(f"Detected {len(boxes)} pieces")
    detections = boxes.xyxy.cpu().numpy()

    # Display detected pieces with annotated image from results
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated)
    plt.title(f"Detected {len(boxes)} Chess Pieces")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return detections, boxes


def connect_square_to_detection(detections, boxes, square):
    if detections is None or boxes is None:
        return "1"  # Return empty square if no detections

    di = {
        0: "b",
        1: "k",
        2: "n",
        3: "p",
        4: "q",
        5: "r",
        6: "B",
        7: "K",
        8: "N",
        9: "P",
        10: "Q",
        11: "R",
    }

    list_of_iou = []
    for i in detections:
        box_x1, box_y1, box_x2, box_y2 = i[0], i[1], i[2], i[3]

        # cut high pieces
        if box_y2 - box_y1 > PIECE_HEIGHT_THRESHOLD:
            box_complete = np.array(
                [
                    [box_x1, box_y1 + PIECE_HEIGHT_ADJUSTMENT],
                    [box_x2, box_y1 + PIECE_HEIGHT_ADJUSTMENT],
                    [box_x2, box_y2],
                    [box_x1, box_y2],
                ]
            )
        else:
            box_complete = np.array(
                [[box_x1, box_y1], [box_x2, box_y1], [box_x2, box_y2], [box_x1, box_y2]]
            )

        try:
            iou = calculate_iou(box_complete, square)
            list_of_iou.append(iou)
        except Exception as e:
            print(f"Error calculating IOU: {e}")
            list_of_iou.append(0)

    if not list_of_iou or max(list_of_iou) <= SQUARE_PIECE_IOU_THRESHOLD:
        return "1"

    num = int(np.argmax(list_of_iou))
    try:
        piece = int(boxes.cls[num].item())
        return di.get(piece, "1")
    except Exception as e:
        print(f"Error getting piece class: {e}")
        return "1"


# --- Main pipeline ---


def main():
    image_path = "images/2.jpg"
    print(f"Starting chess board analysis on {image_path}")

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    # Step 1: Detect corners
    corners = detect_corners(image_path)
    if corners is None:
        print("Could not proceed due to corner detection failure")
        return

    # Step 2: Transform perspective with padding (matching training preprocessing)
    transformed_image = four_point_transform(image_path, corners)
    if transformed_image is None:
        print("Could not proceed due to perspective transform failure")
        return

    # Step 3: Plot grid
    ptsT, ptsL = plot_grid_on_transformed_image(transformed_image)
    if ptsT is None or ptsL is None:
        print("Could not proceed due to grid plotting failure")
        return

    # Step 4: Detect chess pieces
    detections, boxes = chess_pieces_detector(transformed_image)
    if detections is None:
        print("Could not proceed due to piece detection failure")
        return

    # Calculate grid points - using points 1-9 (skipping the padding)
    print("Creating chessboard grid from internal 8x8 region")
    x = [
        ptsT[i][0] for i in range(PADDING_START_INDEX, PADDING_END_INDEX + 1)
    ]  # Skip padding, use indices 1-9
    y = [
        ptsL[i][1] for i in range(PADDING_START_INDEX, PADDING_END_INDEX + 1)
    ]  # Skip padding, use indices 1-9

    # Build all 64 squares
    FEN_annotation = []
    for row in range(CHESSBOARD_SIZE):
        line = []
        for col in range(CHESSBOARD_SIZE):
            square = np.array(
                [
                    [x[col], y[row]],
                    [x[col + 1], y[row]],
                    [x[col + 1], y[row + 1]],
                    [x[col], y[row + 1]],
                ]
            )
            line.append(square)
        FEN_annotation.append(line)

    # Detect pieces for each square
    print("Mapping detected pieces to squares")
    board_FEN = []
    for line in FEN_annotation:
        line_to_FEN = []
        for square in line:
            piece_on_square = connect_square_to_detection(detections, boxes, square)
            line_to_FEN.append(piece_on_square)

        print(line_to_FEN)
        board_FEN.append(line_to_FEN)

    complete_board_FEN = ["".join(line) for line in board_FEN]
    to_FEN = "/".join(complete_board_FEN)

    print("\nFinal FEN notation:")
    print(to_FEN)
    print("\nView board at:")
    print("https://lichess.org/analysis/" + to_FEN)


if __name__ == "__main__":
    main()
