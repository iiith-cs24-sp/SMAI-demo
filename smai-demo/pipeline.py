import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from numpy import asarray
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

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
    print("Loading corner detection model...")
    # Use absolute paths to ensure model loading works correctly
    model_path = os.path.abspath(os.path.join("models", "corners-best-7k.pt"))
    if not os.path.exists(model_path):
        print(f"ERROR: Corner model not found at {model_path}")
        return None

    model = YOLO(model_path)
    print(f"Detecting corners in {image_path}...")
    results = model.predict(source=image_path, conf=0.08, verbose=True)

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
    if pts is None:
        print("Cannot perform transformation without corner points")
        return None

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
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Print information about the transformation
    print(f"Transforming image from points {rect} to standard rectangle")
    print(f"New dimensions: {maxWidth}x{maxHeight}")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Display the warped image
    plt.figure(figsize=(8, 8))
    plt.imshow(warped)
    plt.title("Perspective Transformed Image")
    plt.axis("off")
    plt.show()

    return Image.fromarray(warped)


def plot_grid_on_transformed_image(image):
    if image is None:
        print("Cannot plot grid on a None image")
        return None, None

    w, h = image.size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners = order_points(corners)
    figure(figsize=(10, 10), dpi=80)
    plt.imshow(image)
    plt.title("Grid Overlay on Transformed Image")

    TL, TR, BR, BL = corners

    def interpolate(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / 8
        dy = (y1 - y0) / 8
        return [(x0 + i * dx, y0 + i * dy) for i in range(9)]

    ptsT = interpolate(TL, TR)
    ptsL = interpolate(TL, BL)
    ptsR = interpolate(TR, BR)
    ptsB = interpolate(BL, BR)

    for a, b in zip(ptsL, ptsR):
        plt.plot([a[0], b[0]], [a[1], b[1]], "r-", linestyle="--")
    for a, b in zip(ptsT, ptsB):
        plt.plot([a[0], b[0]], [a[1], b[1]], "r-", linestyle="--")

    plt.axis("off")
    plt.savefig("chessboard_transformed_with_grid.jpg")
    plt.show()

    return ptsT, ptsL


def chess_pieces_detector(image):
    if image is None:
        print("Cannot detect pieces on a None image")
        return None, None

    print("Loading chess piece detection model...")
    # Use absolute paths to ensure model loading works correctly
    model_path = os.path.abspath(
        os.path.join("models", "piece-recog-shiram-chessrender360.pt")
    )
    if not os.path.exists(model_path):
        print(f"ERROR: Piece detection model not found at {model_path}")
        print(f"Looking for: {model_path}")
        return None, None

    model = YOLO(model_path)

    # Save the PIL image to a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        print(f"Saving image to temp file: {tmp.name}")
        image.save(tmp.name)
        print("Running piece detection with confidence threshold 0.25...")
        # Lower confidence threshold to make detection easier
        results = model.predict(source=tmp.name, conf=0.25, verbose=True)

    if results is None or len(results) == 0:
        print("No results from piece detection!")
        return None, None

    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No pieces detected!")
        return None, None

    print(f"Detected {len(boxes)} pieces")
    detections = boxes.xyxy.cpu().numpy()

    # Display detected pieces
    img_np = np.array(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.title(f"Detected {len(boxes)} Chess Pieces")

    # Use detection class names if available
    class_names = model.names if hasattr(model, "names") else None

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = box[:4]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2
        )
        plt.gca().add_patch(rect)

        # Add class label if available
        if class_names and boxes.cls is not None:
            cls_id = int(boxes.cls[i].item())
            label = f"{class_names[cls_id]}: {boxes.conf[i].item():.2f}"
            plt.text(
                x1, y1 - 5, label, fontsize=8, bbox=dict(facecolor="yellow", alpha=0.5)
            )

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
        if box_y2 - box_y1 > 60:
            box_complete = np.array(
                [
                    [box_x1, box_y1 + 40],
                    [box_x2, box_y1 + 40],
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

    if not list_of_iou or max(list_of_iou) <= 0.15:
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

    # Step 2: Transform perspective
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

    # Calculate grid points
    print("Creating chessboard grid")
    x = [ptsT[i][0] for i in range(9)]
    y = [ptsL[i][1] for i in range(9)]

    # Build all 64 squares
    FEN_annotation = []
    for row in range(8):
        line = []
        for col in range(8):
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

        corrected_FEN = (
            line_to_FEN  # No need to replace 'empty' as we're using '1' directly
        )
        print(corrected_FEN)
        board_FEN.append(corrected_FEN)

    complete_board_FEN = ["".join(line) for line in board_FEN]
    to_FEN = "/".join(complete_board_FEN)

    print("\nFinal FEN notation:")
    print(to_FEN)
    print("\nView board at:")
    print("https://lichess.org/analysis/" + to_FEN)


if __name__ == "__main__":
    main()
