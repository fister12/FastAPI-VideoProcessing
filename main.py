import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient, models
import uuid




# --- Configuration ---
# Directory to temporarily store uploaded video files
UPLOAD_DIR = Path("uploaded_videos")
# Directory to store extracted frames
FRAME_DIR = Path("extracted_frames")
# Qdrant collection name for storing frame feature vectors
COLLECTION_NAME = "video_frames"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Video Frame Processor and Similarity Search",
    description="API for uploading videos, extracting frames, computing feature vectors, "
                "and querying for similar frames using an in-memory Qdrant vector database.",
    version="1.0.0"
)

# --- Qdrant Client Initialization ---
# Initialize an in-memory Qdrant client for demonstration purposes.
# For production, you would connect to a persistent Qdrant instance.
qdrant_client = QdrantClient(":memory:")

# Define the vector size for our feature vectors (color histogram).
# A 3D RGB histogram with 8 bins per channel will have 8*8*8 = 512 bins.
VECTOR_SIZE = 512

@app.on_event("startup")
async def startup_event():
    """
    On application startup, ensure necessary directories exist and
    create the Qdrant collection if it doesn't already.
    """
    print("Starting up application...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Check if the collection exists, create if not
        collections = qdrant_client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            print(f"Qdrant collection '{COLLECTION_NAME}' created successfully.")
        else:
            print(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"Error initializing Qdrant client or collection: {e}")
        # In a real app, you might want to raise an exception or handle this more gracefully.

@app.on_event("shutdown")
async def shutdown_event():
    """
    On application shutdown, clean up temporary directories.
    """
    print("Shutting down application. Cleaning up temporary files...")
    # Clean up uploaded videos
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        print(f"Removed '{UPLOAD_DIR}' directory.")
    # Clean up extracted frames
    if FRAME_DIR.exists():
        shutil.rmtree(FRAME_DIR)
        print(f"Removed '{FRAME_DIR}' directory.")

# --- Helper Functions ---

def extract_frames(video_path: Path, interval_seconds: int, output_dir: Path) -> List[Path]:
    """
    Extracts frames from a video at a specified interval and saves them as images.

    Args:
        video_path (Path): Path to the input video file.
        interval_seconds (int): Interval in seconds at which to extract frames.
        output_dir (Path): Directory where extracted frames will be saved.

    Returns:
        List[Path]: A list of paths to the extracted frame images.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")

    frame_interval = int(fps * interval_seconds) # Calculate frames to skip
    if frame_interval == 0: # Ensure at least one frame is processed per second if FPS is very low
        frame_interval = 1

    extracted_frame_paths = []
    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from '{video_path}' at {interval_seconds}-second intervals...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break # End of video

        # Process frame only if it's at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_path.stem}_frame_{saved_count:05d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            extracted_frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    video_capture.release()
    print(f"Extracted {saved_count} frames from '{video_path}'.")
    return extracted_frame_paths

def compute_color_histogram(image_path: Path) -> List[float]:
    """
    Computes a 3D RGB color histogram as a feature vector for an image.
    The histogram is normalized.

    Args:
        image_path (Path): Path to the input image file.

    Returns:
        List[float]: The normalized color histogram (feature vector).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    # Convert BGR to RGB (OpenCV reads in BGR by default)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define bins for each channel (e.g., 8 bins per channel)
    h_bins = 8
    s_bins = 8
    v_bins = 8
    hist_size = [h_bins, s_bins, v_bins]

    # Define ranges for each channel (0-256 for RGB)
    # Using range for 0-255 for each channel
    ranges = [0, 256, 0, 256, 0, 256] # [R_min, R_max, G_min, G_max, B_min, B_max]

    # Compute histogram for BGR channels (OpenCV's default)
    # channels are 0, 1, 2 for B, G, R respectively
    hist = cv2.calcHist([image], [0, 1, 2], None, hist_size, ranges)

    # Normalize the histogram
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Flatten the 3D histogram into a 1D vector and convert to list of floats
    feature_vector = hist.flatten().tolist()

    # Ensure the vector size matches expectation
    if len(feature_vector) != VECTOR_SIZE:
        raise ValueError(f"Computed feature vector size {len(feature_vector)} does not match expected size {VECTOR_SIZE}")

    return feature_vector

async def process_and_store_frames_task(video_path: Path, interval_seconds: int, video_id: str):
    """
    Background task to process extracted frames, compute feature vectors,
    and store them in Qdrant.
    """
    try:
        # Create a unique directory for frames related to this video
        current_frame_output_dir = FRAME_DIR / video_id
        extracted_frame_paths = extract_frames(video_path, interval_seconds, current_frame_output_dir)

        points_to_insert = []
        for frame_path in extracted_frame_paths:
            try:
                feature_vector = compute_color_histogram(frame_path)
                points_to_insert.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()), # Use frame path as unique ID in Qdrant
                        vector=feature_vector,
                        payload={"frame_path": str(frame_path), "video_id": video_id}
                    )
                )
            except Exception as e:
                print(f"Error computing feature vector for {frame_path}: {e}")
                continue # Skip this frame and continue with others

        if points_to_insert:
            print(f"Upserting {len(points_to_insert)} points into Qdrant collection '{COLLECTION_NAME}'...")
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points_to_insert,
            )
            print("Feature vectors stored successfully in Qdrant.")
        else:
            print("No valid feature vectors to store in Qdrant.")

    except Exception as e:
        print(f"Error during background video processing task for {video_path}: {e}")
    finally:
        # Clean up the original uploaded video file after processing
        if video_path.exists():
            video_path.unlink()
            print(f"Cleaned up uploaded video: {video_path}")


# --- API Endpoints ---

@app.post("/upload_and_process_video/")
async def upload_video(
    video_file: UploadFile = File(...),
    interval_seconds: int = 1,
    background_tasks: BackgroundTasks = None
):
    """
    Uploads a video file, extracts frames, computes feature vectors,
    and stores them in the vector database.
    Processing happens in a background task to avoid blocking the API response.

    Args:
        video_file (UploadFile): The video file to upload (e.g., MP4).
        interval_seconds (int): The interval in seconds to extract frames (default: 1).
    """
    if not video_file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only video files (MP4, MOV, AVI, MKV) are supported."
        )

    # Generate a unique ID for this video upload
    video_id = f"{Path(video_file.filename).stem}_{os.urandom(4).hex()}"
    uploaded_video_path = UPLOAD_DIR / f"{video_id}{Path(video_file.filename).suffix}"

    try:
        # Save the uploaded video file
        with open(uploaded_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # Add the frame extraction, feature computation, and storage to a background task.
        # This allows the API to respond immediately while processing continues in the background.
        if background_tasks is not None:
            background_tasks.add_task(
                process_and_store_frames_task, uploaded_video_path, interval_seconds, video_id
            )
        else:
            # For direct call/testing
            await process_and_store_frames_task(uploaded_video_path, interval_seconds, video_id)

        return JSONResponse(
            status_code=202,
            content={
                "message": "Video upload successful. Frame extraction and processing initiated in background.",
                "video_id": video_id,
                "file_name": video_file.filename,
                "interval_seconds": interval_seconds
            }
        )
    except Exception as e:
        print(f"Error during video upload or processing initiation: {e}")
        # Clean up the uploaded file if an error occurs before background task starts
        if uploaded_video_path.exists():
            uploaded_video_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process video: {e}")



@app.get("/")
async def root():
    return {"message": "Welcome to the Video Processing API!"}

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)  # No content for favicon


@app.post("/query_similar_frames/")
async def query_similar_frames(
    query_image: UploadFile = File(...),
    limit: int = 5
):
    """
    Queries the vector database for frames similar to the provided query image.

    Args:
        query_image (UploadFile): The image file to use for similarity search.
        limit (int): The maximum number of similar frames to return (default: 5).

    Returns:
        JSONResponse: A list of similar frames with their paths and feature vectors.
    """
    if not query_image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only image files (JPG, PNG) are supported for query."
        )

    # Save query image temporarily to compute its feature vector
    query_image_path = UPLOAD_DIR / f"query_image_{os.urandom(4).hex()}{Path(query_image.filename).suffix}"
    try:
        with open(query_image_path, "wb") as buffer:
            shutil.copyfileobj(query_image.file, buffer)

        query_vector = compute_color_histogram(query_image_path)
    except Exception as e:
        if query_image_path.exists():
            query_image_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process query image: {e}")
    finally:
        if query_image_path.exists():
            query_image_path.unlink() # Clean up temporary query image

    try:
        print(f"Searching for similar frames in '{COLLECTION_NAME}' collection...")
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True # Include payload (frame_path) in results
        )

        similar_frames_data = []
        for hit in search_result:
            frame_path = hit.payload.get("frame_path") if hasattr(hit, 'payload') else None
            score = hit.score if hasattr(hit, 'score') else None
            frame_id = hit.id if hasattr(hit, 'id') else None
            if frame_path:
                similar_frames_data.append({
                    "frame_path": frame_path,
                    "similarity_score": score,
                    "frame_id": frame_id
                })

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Found {len(similar_frames_data)} similar frames.",
                "query_image_name": query_image.filename,
                "similar_frames": similar_frames_data
            }
        )

    except Exception as e:
        print(f"Error during similar frame query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query similar frames: {e}")


# --- How to Run (for local development) ---
# To run this application, save the code as `main.py` and run the following command in your terminal:
# uvicorn main:app --reload

# Then, you can access the API documentation (Swagger UI) at:
# http://127.0.0.1:8000/docs
# Or the alternative ReDoc documentation at:
# http://127.0.0.1:8000/redoc

# Example usage with curl:
# 1. Upload a video:
#    curl -X POST "http://127.0.0.1:8000/upload_and_process_video/?interval_seconds=2" \
#    -H "accept: application/json" \
#    -H "Content-Type: multipart/form-data" \
#    -F "video_file=@/path/to/your/video.mp4"
#    (Replace `/path/to/your/video.mp4` with the actual path to your video file)

# 2. Query similar frames with an image:
#    curl -X POST "http://127.0.0.1:8000/query_similar_frames/?limit=3" \
#    -H "accept: application/json" \
#    -H "Content-Type: multipart/form-data" \
#    -F "query_image=@/path/to/your/image.jpg"
#    (Replace `/path/to/your/image.jpg` with the actual path to an image file,
#    preferably one of the extracted frames for testing similarity.)

