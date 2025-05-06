import argparse
import os
import tempfile
import time
import glob
import sys
import concurrent.futures
from functools import lru_cache

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
SQUARE_PIECE_IOU_THRESHOLD = 0.10  # Minimum IoU to consider a piece on a square
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
    xs, ys = warped[:, 0], warped[:, 1]
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


def load_models(corner_model_path, piece_model_path):
    """Load corner and piece detection models once.

    Args:
        corner_model_path (str): Path to the corner detection model.
        piece_model_path (str): Path to the piece detection model.

    Returns:
        tuple: (corner_model, piece_model) or None if loading failed
    """
    print("Loading corner detection model...")
    if not os.path.exists(corner_model_path):
        print(f"ERROR: Corner model not found at {corner_model_path}")
        return None, None

    try:
        corner_model = YOLO(corner_model_path)
        corner_model.fuse()  # Fuse model layers for faster inference
        corner_model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Failed to load corner model: {e}")
        return None, None

    print("Loading chess piece detection model...")
    if not os.path.exists(piece_model_path):
        print(f"ERROR: Piece detection model not found at {piece_model_path}")
        return None, None

    try:
        piece_model = YOLO(piece_model_path)
        piece_model.fuse()  # Fuse model layers for faster inference
        piece_model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Failed to load piece model: {e}")
        return None, None

    return corner_model, piece_model


def detect_corners(image_path, corner_model, show_plots):
    """Detect corners using a pre-loaded model."""
    print(f"Detecting corners in {image_path}...")

    # Read image directly to avoid potential unnecessary conversions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Use direct image rather than loading from path again
    results = corner_model.predict(
        source=img, conf=CORNER_DETECTION_CONF_THRESHOLD, verbose=False
    )

    # Visualize corner detection results only if needed
    if show_plots:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title("Original image with detected corners")

        boxes = results[0].boxes
        if len(boxes) == 0:
            print("No corners detected!")
            plt.close()
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
    else:
        boxes = results[0].boxes
        if len(boxes) == 0:
            print("No corners detected!")
            return None

        arr = boxes.xywh.cpu().numpy()
        points = arr[:, 0:2]
        corners = order_points(points)

    return corners


def four_point_transform(image_path, pts, show_plots):
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

    # Display the warped image only if requested
    if show_plots:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title("Perspective Transformed Image (with padding)")
        plt.axis("off")
        plt.show()

    # Return directly as numpy array, no need for PIL conversion
    return warped


def plot_grid_on_transformed_image(image, show_plots):
    if image is None:
        print("Cannot plot grid on a None image")
        return None, None

    # Get dimensions from numpy array directly without conversion
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners = order_points(corners)

    if show_plots:
        figure(figsize=(10, 10), dpi=80)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

    if show_plots:
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
        if not os.path.exists("output"):
            os.makedirs("output")
        plt.savefig("output/chessboard_transformed_with_grid.jpg")
        plt.show()

    return ptsT, ptsL


def chess_pieces_detector(image, piece_model, show_plots):
    """Detect chess pieces using a pre-loaded model."""
    if image is None:
        print("Cannot detect pieces on a None image")
        return None, None

    # Use numpy array directly and avoid temporary file
    print("Running piece detection with settings from training...")
    # Match training parameters
    results = piece_model.predict(
        source=image,  # Pass numpy array directly
        conf=PIECE_DETECTION_CONF_THRESHOLD,  # confidence threshold
        iou=PIECE_DETECTION_IOU_THRESHOLD,  # IoU threshold
        imgsz=PIECE_DETECTION_IMG_SIZE,  # image size for inference
        verbose=False,  # Reduce logging for speed
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

    # Print detailed class information for debugging
    if hasattr(piece_model, "names"):
        classes = boxes.cls.cpu().numpy()
        for i, cls in enumerate(classes):
            cls_idx = int(cls)
            class_name = piece_model.names.get(cls_idx, f"Unknown-{cls_idx}")
            x1, y1, x2, y2 = detections[i]
            conf = boxes.conf[i].item()
            print(
                f"  Piece {i}: Class={cls_idx} ({class_name}), Confidence={conf:.3f}, Box=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
            )

    # Display detected pieces with annotated image from results only if needed
    if show_plots:
        annotated = results[0].plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected {len(boxes)} Chess Pieces")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return detections, boxes


@lru_cache(maxsize=8)
def get_piece_notation_map(piece_model):
    """Cache the piece notation map for reuse"""
    # Create a mapping from model class indices to chess notation
    piece_notation_map = {}
    if hasattr(piece_model, "names"):
        print("Model class names:")
        for idx, name in piece_model.names.items():
            print(f"  Class {idx}: {name}")

        # Create mapping from class names to FEN notation - support both naming conventions
        chess_notation = {
            # Hyphenated names
            "white-queen": "Q",
            "white-king": "K",
            "white-bishop": "B",
            "white-knight": "N",
            "white-rook": "R",
            "white-pawn": "P",
            "black-queen": "q",
            "black-king": "k",
            "black-bishop": "b",
            "black-knight": "n",
            "black-rook": "r",
            "black-pawn": "p",
            # Underscore names
            "white_queen": "Q",
            "white_king": "K",
            "white_bishop": "B",
            "white_knight": "N",
            "white_rook": "R",
            "white_pawn": "P",
            "black_queen": "q",
            "black_king": "k",
            "black_bishop": "b",
            "black_knight": "n",
            "black_rook": "r",
            "black_pawn": "p",
            # Common variations or typos
            "w-king": "K",
            "w-queen": "Q",
            "w-bishop": "B",
            "w-knight": "N",
            "w-rook": "R",
            "w-pawn": "P",
            "b-king": "k",
            "b-queen": "q",
            "b-bishop": "b",
            "b-knight": "n",
            "b-rook": "r",
            "b-pawn": "p",
            "wking": "K",
            "wqueen": "Q",
            "wbishop": "B",
            "wknight": "N",
            "wrook": "R",
            "wpawn": "P",
            "bking": "k",
            "bqueen": "q",
            "bbishop": "b",
            "bknight": "n",
            "brook": "r",
            "bpawn": "p",
        }

        # Special case for known problematic classes based on observed issues
        special_cases = {
            # If you know the exact class index for the white king, add it here
            # For example, if the white king is consistently misclassified:
            # 2: "K",  # Force class 2 to be recognized as white king
        }

        # Map each class index to its FEN notation
        for idx, name in piece_model.names.items():
            # Check special cases first
            if idx in special_cases:
                piece_notation_map[idx] = special_cases[idx]
                print(
                    f"  Applied special case mapping for class {idx}: {name} → {special_cases[idx]}"
                )
                continue

            # Normalize name by converting to lowercase
            normalized_name = name.lower()

            # Try direct lookup first
            if normalized_name in chess_notation:
                piece_notation_map[idx] = chess_notation[normalized_name]
            else:
                # Try replacing hyphens with underscores and vice versa
                alt_name_underscore = normalized_name.replace("-", "_")
                alt_name_hyphen = normalized_name.replace("_", "-")

                if alt_name_underscore in chess_notation:
                    piece_notation_map[idx] = chess_notation[alt_name_underscore]
                elif alt_name_hyphen in chess_notation:
                    piece_notation_map[idx] = chess_notation[alt_name_hyphen]
                else:
                    print(f"Warning: Unknown piece class '{name}' at index {idx}")
                    piece_notation_map[idx] = "1"  # Default to empty square

        print(f"Loaded class mapping from model: {piece_notation_map}")
    else:
        # Fallback to hardcoded mapping if model doesn't provide names
        # Adjusted default mapping to ensure white king is correctly mapped
        piece_notation_map = {
            0: "Q",  # white queen
            1: "K",  # white king
            2: "B",  # white bishop
            3: "N",  # white knight
            4: "R",  # white rook
            5: "P",  # white pawn
            6: "q",  # black queen
            7: "k",  # black king
            8: "b",  # black bishop
            9: "n",  # black knight
            10: "r",  # black rook
            11: "p",  # black pawn
        }
        print("Using default class mapping (model did not provide names)")

    # Add a verification step for the king piece
    has_white_king = "K" in piece_notation_map.values()
    has_black_king = "k" in piece_notation_map.values()

    if not has_white_king:
        print(
            "WARNING: No white king (K) found in piece mapping! This will cause incorrect FEN notation."
        )
        # Try to find the class most likely to be the white king
        for idx, name in (
            piece_model.names.items() if hasattr(piece_model, "names") else []
        ):
            if "king" in name.lower() and (
                "white" in name.lower()
                or "w-" in name.lower()
                or name.lower().startswith("w")
            ):
                piece_notation_map[idx] = "K"
                print(
                    f"  Auto-corrected: Mapping class {idx} ({name}) to white king (K)"
                )
                break

    if not has_black_king:
        print(
            "WARNING: No black king (k) found in piece mapping! This will cause incorrect FEN notation."
        )

    return piece_notation_map


def connect_square_to_detection(
    detections, boxes, square, assigned_pieces, piece_notation_map
):
    """Connect a square to a chess piece detection.

    Args:
        detections: Detected piece boxes as xyxy coordinates
        boxes: Detection boxes with class information
        square: The chess square polygon
        assigned_pieces: Set of already assigned piece indices
        piece_notation_map: Dictionary mapping class indices to chess notation

    Returns:
        Chess piece notation ('p', 'K', etc.) or '1' for empty square
    """
    if detections is None or boxes is None:
        return "1"  # Return empty square if no detections

    if assigned_pieces is None:
        assigned_pieces = set()

    list_of_iou = []
    list_of_confidence = []
    piece_indices = []

    # Get square's bottom y-coordinate for determining bottom placement
    square_bottom = max(square[2][1], square[3][1])  # Bottom y-coordinate of square

    for i, detection in enumerate(detections):
        # Skip already assigned pieces
        if i in assigned_pieces:
            list_of_iou.append(0)
            list_of_confidence.append(0)
            piece_indices.append(i)
            continue

        box_x1, box_y1, box_x2, box_y2 = (
            detection[0],
            detection[1],
            detection[2],
            detection[3],
        )

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

            # Use bottom box constraint - if the bottom of the piece is not in this square,
            # we don't want to consider it (no matter how much it overlaps)
            piece_bottom = box_y2

            # If the piece's bottom is not close to the square's bottom, reduce the IoU
            if abs(piece_bottom - square_bottom) > SQUARE_SZ / CHESSBOARD_SIZE / 2:
                iou = iou * 0.1  # Heavily penalize non-bottom placements

            list_of_iou.append(iou)
            # Store confidence value for this detection from boxes
            list_of_confidence.append(float(boxes.conf[i].item()))
            piece_indices.append(i)
        except Exception as e:
            print(f"Error calculating IOU: {e}")
            list_of_iou.append(0)
            list_of_confidence.append(0)
            piece_indices.append(i)

    if not list_of_iou or max(list_of_iou) <= SQUARE_PIECE_IOU_THRESHOLD:
        return "1"

    # Find all detections with reasonable IoU
    valid_detections = [
        (i, iou, conf)
        for i, (iou, conf) in enumerate(zip(list_of_iou, list_of_confidence))
        if iou > SQUARE_PIECE_IOU_THRESHOLD
    ]

    if not valid_detections:
        return "1"

    # Sort by IoU first (primary criteria)
    valid_detections.sort(key=lambda x: x[1], reverse=True)

    # If top candidates have similar IoU (within 10% of each other),
    # then choose the one with highest confidence
    top_iou = valid_detections[0][1]
    top_candidates = [det for det in valid_detections if det[1] >= top_iou * 0.9]

    if len(top_candidates) > 1:
        # Sort by confidence for candidates with similar IoU
        top_candidates.sort(key=lambda x: x[2], reverse=True)
        print(
            f"Multiple detections for square: Selected highest confidence {top_candidates[0][2]:.3f} vs {[f'{c[2]:.3f}' for c in top_candidates[1:]]}"
        )

    # Get the index of the best candidate
    best_idx = top_candidates[0][0]
    num = piece_indices[best_idx]
    assigned_pieces.add(num)  # Mark this piece as assigned

    try:
        piece = int(boxes.cls[num].item())
        return piece_notation_map.get(piece, "1")
    except Exception as e:
        print(f"Error getting piece class: {e}")
        return "1"


def process_image(image_path, corner_model, piece_model, show_plots):
    """Process a single chess board image using pre-loaded models.

    Args:
        image_path (str): Path to the chess board image to analyze.
        corner_model: Pre-loaded corner detection model.
        piece_model: Pre-loaded piece detection model.
        show_plots (bool): Whether to display matplotlib plots.

    Returns:
        str: FEN notation of the board or None if processing failed
    """
    print(f"\nProcessing image: {image_path}")
    start = time.time()

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return None

    # Step 1: Detect corners
    corners = detect_corners(image_path, corner_model, show_plots)
    if corners is None:
        print("Could not proceed due to corner detection failure")
        return None

    # Step 2: Transform perspective with padding (matching training preprocessing)
    transformed_image = four_point_transform(image_path, corners, show_plots)
    if transformed_image is None:
        print("Could not proceed due to perspective transform failure")
        return None

    # Step 3: Plot grid
    ptsT, ptsL = plot_grid_on_transformed_image(transformed_image, show_plots)
    if ptsT is None or ptsL is None:
        print("Could not proceed due to grid plotting failure")
        return None

    # Step 4: Detect chess pieces
    detections, boxes = chess_pieces_detector(
        transformed_image, piece_model, show_plots
    )
    if detections is None:
        print("Could not proceed due to piece detection failure")
        return None

    # Create a mapping from model class indices to chess notation
    piece_notation_map = get_piece_notation_map(piece_model)

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

    # Process squares from bottom to top to ensure tall pieces are assigned to bottom squares first
    # Create a flat list of all squares with their position info
    all_squares = []
    for row_idx, line in enumerate(FEN_annotation):
        for col_idx, square in enumerate(line):
            # Store square with its position information
            all_squares.append((square, row_idx, col_idx))

    # Sort squares by row in descending order (bottom to top)
    all_squares.sort(key=lambda x: -x[1])

    # Initialize tracking of assigned pieces
    assigned_pieces = set()

    # Process squares and build the board representation
    board_FEN = [["1" for _ in range(CHESSBOARD_SIZE)] for _ in range(CHESSBOARD_SIZE)]

    print("Mapping detected pieces to squares (bottom to top)")
    for square, row_idx, col_idx in all_squares:
        piece_on_square = connect_square_to_detection(
            detections, boxes, square, assigned_pieces, piece_notation_map
        )
        board_FEN[row_idx][col_idx] = piece_on_square
        if piece_on_square != "1":
            print(
                f"Assigned {piece_on_square} to square at row {row_idx}, col {col_idx}"
            )

    # Display the board
    for row in board_FEN:
        print(row)

    # Compile FEN string
    complete_board_FEN = []
    for line in board_FEN:
        # Optimize FEN generation by compressing consecutive empty squares
        fen_line = ""
        empty_count = 0
        for cell in line:
            if cell == "1":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_line += str(empty_count)
                    empty_count = 0
                fen_line += cell
        if empty_count > 0:
            fen_line += str(empty_count)
        complete_board_FEN.append(fen_line)

    to_FEN = "/".join(complete_board_FEN)

    print("\nFinal FEN notation:")
    print(to_FEN)
    print("\nView board at:")
    print("https://lichess.org/analysis/" + to_FEN)

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

    return to_FEN


def process_image_parallel(args):
    """Wrapper function for multiprocessing."""
    image_path, corner_model, piece_model, show_plots, img_num, total_images = args
    print(f"\n[{img_num}/{total_images}] Processing {os.path.basename(image_path)}")
    return image_path, process_image(image_path, corner_model, piece_model, show_plots)


def process_folder(
    folder_path, corner_model_path, piece_model_path, show_plots, num_workers=None
):
    """Process all images in a folder using parallel processing.

    Args:
        folder_path (str): Path to folder containing chess board images.
        corner_model_path (str): Path to the corner detection model.
        piece_model_path (str): Path to the piece detection model.
        show_plots (bool): Whether to display matplotlib plots.
        num_workers (int): Number of parallel workers. None means auto-detect.
    """
    if not os.path.isdir(folder_path):
        print(f"ERROR: Folder not found at {folder_path}")
        return

    print(f"Processing all images in folder: {folder_path}")
    start_total = time.time()

    # Load models once (will be shared across processes)
    corner_model, piece_model = load_models(corner_model_path, piece_model_path)
    if corner_model is None or piece_model is None:
        print("Failed to load models, aborting batch processing")
        return

    # Get all image files
    image_files = []
    for ext in ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext.upper()}")))

    # Make the list unique
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    # Determine optimal number of workers if not specified
    if num_workers is None:
        import multiprocessing

        num_workers = min(multiprocessing.cpu_count(), len(image_files))

    print(f"Found {len(image_files)} images to process using {num_workers} workers")

    # Process images in parallel
    results = {}

    # Create task arguments
    tasks = [
        (image_path, corner_model, piece_model, show_plots, i + 1, len(image_files))
        for i, image_path in enumerate(image_files)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for image_path, fen in executor.map(process_image_parallel, tasks):
            if fen:
                results[image_path] = fen
                print(f"✓ Completed {os.path.basename(image_path)}")
            else:
                print(f"✗ Failed {os.path.basename(image_path)}")

    # Summary
    end_total = time.time()
    print("\n=== Summary ===")
    print(f"Processed {len(results)}/{len(image_files)} images successfully")
    print(f"Total batch processing time: {end_total - start_total:.2f} seconds")
    print(
        f"Average time per image: {(end_total - start_total) / len(image_files):.2f} seconds"
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join(folder_path, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write results to file
    result_file = os.path.join(output_dir, "chess_results.txt")
    with open(result_file, "w") as f:
        for img_path, fen in results.items():
            f.write(
                f"{os.path.basename(img_path)}: {fen}\t Link: https://lichess.org/analysis/{fen}\n"
            )

    print(f"Results saved to {result_file}")


def pipeline(image_path, corner_model_path, piece_model_path, show_plots):
    """Run the chess board analysis pipeline on the given image path."""
    print(f"Starting chess board analysis on {image_path}")

    corner_model, piece_model = load_models(corner_model_path, piece_model_path)
    if corner_model is None or piece_model is None:
        print("Failed to load models, aborting processing")
        return

    # Process the image
    process_image(image_path, corner_model, piece_model, show_plots)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chess Board Analysis")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Path to the chess board image",
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        help="Path to folder containing chess board images",
    )
    parser.add_argument(
        "--corner-model",
        "-c",
        type=str,
        help="Path to corner detection model",
        default=os.path.abspath(os.path.join("models", "corners-best-7k.pt")),
    )
    parser.add_argument(
        "--piece-model",
        "-p",
        type=str,
        help="Path to piece detection model",
        default=os.path.abspath(
            os.path.join("models", "piece-recog-shiram-chessrender360.pt")
        ),
    )
    parser.add_argument(
        "--no-display", "-nd", action="store_true", help="Disable displaying plots"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )
    args = parser.parse_args()

    # Check that at least one of image or folder is provided
    if not args.image and not args.folder:
        print("ERROR: Either --image or --folder must be specified")
        parser.print_help()
        sys.exit(1)

    # Set matplotlib to use non-interactive backend if not showing plots
    if args.no_display:
        matplotlib_backend = plt.get_backend()
        plt.switch_backend("Agg")

    # Run processing based on input type
    if args.folder:
        process_folder(
            args.folder,
            args.corner_model,
            args.piece_model,
            not args.no_display,
            args.workers,
        )
    else:
        pipeline(args.image, args.corner_model, args.piece_model, not args.no_display)

    # Restore matplotlib backend if changed
    if args.no_display:
        plt.switch_backend(matplotlib_backend)
