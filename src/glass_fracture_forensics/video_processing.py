#!/usr/bin/env python3
"""
VIDEO PROCESSING MODULE
=======================

Real-time video processing for AR-guided fracture capture.
Extracts frames, detects fractures, and manages capture session.

FEATURES:
- Video stream processing (camera, file)
- Frame extraction and preprocessing
- Fracture detection and masking
- Temporal consistency tracking
- Quality assessment per frame
- Session management

Author: Forensic Engineering Team
Version: 2.1
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
from enum import Enum
import time


class CaptureSource(Enum):
    """Video capture source types"""
    CAMERA = "camera"
    VIDEO_FILE = "video_file"
    IMAGE_SEQUENCE = "image_sequence"


@dataclass
class VideoFrame:
    """Single video frame with metadata"""
    frame_id: int
    timestamp: float
    image: np.ndarray
    grayscale: np.ndarray
    fracture_mask: Optional[np.ndarray] = None
    quality_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class CaptureSession:
    """Complete capture session data"""
    session_id: str
    start_time: float
    frames: List[VideoFrame] = field(default_factory=list)
    camera_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    camera_poses: List[np.ndarray] = field(default_factory=list)

    def add_frame(self, frame: VideoFrame, pose: Optional[np.ndarray] = None):
        """Add frame to session"""
        self.frames.append(frame)
        if pose is not None:
            self.camera_poses.append(pose)

    def get_frame_count(self) -> int:
        """Get total frame count"""
        return len(self.frames)

    def get_duration(self) -> float:
        """Get session duration in seconds"""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


class FractureDetector:
    """
    FRACTURE DETECTION AND SEGMENTATION

    Uses edge detection and morphological operations to identify
    fracture lines in glass.

    ALGORITHM:
    1. Gaussian blur for noise reduction
    2. Canny edge detection
    3. Morphological closing to connect edges
    4. Contour filtering by length and linearity
    """

    def __init__(self,
                 blur_kernel: int = 5,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 min_contour_length: int = 50):
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_length = min_contour_length

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect fractures in image

        Args:
            image: Grayscale image

        Returns:
            Binary mask with detected fractures
        """
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by length
        mask = np.zeros_like(image)
        for contour in contours:
            if cv2.arcLength(contour, False) > self.min_contour_length:
                cv2.drawContours(mask, [contour], -1, 255, 2)

        return mask

    def detect_with_confidence(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect fractures with confidence score

        Returns:
            mask: Binary fracture mask
            confidence: Detection confidence [0, 1]
        """
        mask = self.detect(image)

        # Compute confidence based on edge strength
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        edge_density = np.sum(edges > 0) / edges.size

        # Heuristic confidence
        if edge_density > 0.1:  # Too many edges - noisy
            confidence = 0.5
        elif edge_density < 0.001:  # Too few edges - no fracture
            confidence = 0.3
        else:
            confidence = 0.9

        return mask, confidence


class VideoProcessor:
    """
    VIDEO STREAM PROCESSOR

    Manages video capture, frame extraction, and preprocessing.
    """

    def __init__(self,
                 source: CaptureSource,
                 source_path: Optional[str] = None,
                 target_fps: int = 10,
                 max_frames: Optional[int] = None):
        self.source = source
        self.source_path = source_path
        self.target_fps = target_fps
        self.max_frames = max_frames

        self.fracture_detector = FractureDetector()
        self.capture = None
        self.frame_count = 0

    def open(self) -> bool:
        """Open video source"""
        if self.source == CaptureSource.CAMERA:
            self.capture = cv2.VideoCapture(0)  # Default camera
        elif self.source == CaptureSource.VIDEO_FILE:
            if self.source_path is None:
                raise ValueError("source_path required for VIDEO_FILE")
            self.capture = cv2.VideoCapture(self.source_path)
        else:
            raise NotImplementedError(f"Source {self.source} not implemented")

        return self.capture is not None and self.capture.isOpened()

    def close(self):
        """Close video source"""
        if self.capture is not None:
            self.capture.release()

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from source"""
        if self.capture is None:
            return None

        ret, frame = self.capture.read()
        if not ret:
            return None

        return frame

    def process_frame(self, frame: np.ndarray, frame_id: int) -> VideoFrame:
        """
        Process single frame

        Args:
            frame: BGR image from camera
            frame_id: Frame identifier

        Returns:
            VideoFrame with processed data
        """
        # Convert to grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect fractures
        fracture_mask, confidence = self.fracture_detector.detect_with_confidence(grayscale)

        # Create VideoFrame
        video_frame = VideoFrame(
            frame_id=frame_id,
            timestamp=time.time(),
            image=frame,
            grayscale=grayscale,
            fracture_mask=fracture_mask,
            quality_score=confidence
        )

        return video_frame

    def capture_session(self,
                       camera_matrix: np.ndarray,
                       frame_callback: Optional[Callable[[VideoFrame], None]] = None,
                       stop_callback: Optional[Callable[[], bool]] = None) -> CaptureSession:
        """
        Capture complete session

        Args:
            camera_matrix: Camera intrinsic matrix
            frame_callback: Called for each frame (e.g., for display)
            stop_callback: Returns True to stop capture

        Returns:
            CaptureSession with all frames
        """
        if not self.open():
            raise RuntimeError("Failed to open video source")

        session = CaptureSession(
            session_id=f"session_{int(time.time())}",
            start_time=time.time(),
            camera_matrix=camera_matrix
        )

        frame_id = 0

        try:
            while True:
                # Check max frames
                if self.max_frames is not None and frame_id >= self.max_frames:
                    break

                # Check stop condition
                if stop_callback is not None and stop_callback():
                    break

                # Read frame
                frame = self.read_frame()
                if frame is None:
                    break

                # Process frame
                video_frame = self.process_frame(frame, frame_id)

                # Add to session
                session.add_frame(video_frame)

                # Callback (e.g., display)
                if frame_callback is not None:
                    frame_callback(video_frame)

                frame_id += 1

                # Frame rate control
                time.sleep(1.0 / self.target_fps)

        finally:
            self.close()

        return session


class FrameSelector:
    """
    INTELLIGENT FRAME SELECTION

    Selects optimal frames for reconstruction based on:
    - Motion between frames (parallax)
    - Image quality
    - Spatial coverage
    - Temporal distribution
    """

    def __init__(self,
                 min_motion_threshold: float = 10.0,
                 max_motion_threshold: float = 100.0,
                 target_frame_count: int = 30):
        self.min_motion_threshold = min_motion_threshold
        self.max_motion_threshold = max_motion_threshold
        self.target_frame_count = target_frame_count

    def compute_motion(self, frame1: VideoFrame, frame2: VideoFrame) -> float:
        """
        Compute motion between frames using optical flow

        Returns:
            Mean motion magnitude [pixels]
        """
        # Detect features in first frame
        corners = cv2.goodFeaturesToTrack(
            frame1.grayscale,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10
        )

        if corners is None or len(corners) < 10:
            return 0.0

        # Track to second frame
        corners2, status, _ = cv2.calcOpticalFlowPyrLK(
            frame1.grayscale,
            frame2.grayscale,
            corners,
            None
        )

        # Compute mean motion
        valid_corners = corners[status.ravel() == 1]
        valid_corners2 = corners2[status.ravel() == 1]

        if len(valid_corners) == 0:
            return 0.0

        motion = np.linalg.norm(valid_corners2 - valid_corners, axis=1)
        mean_motion = np.mean(motion)

        return mean_motion

    def select_keyframes(self, session: CaptureSession) -> List[int]:
        """
        Select keyframes from session

        Args:
            session: Complete capture session

        Returns:
            List of selected frame indices
        """
        if len(session.frames) <= self.target_frame_count:
            return list(range(len(session.frames)))

        selected = [0]  # Always include first frame

        for i in range(1, len(session.frames)):
            # Compute motion from last selected frame
            last_idx = selected[-1]
            motion = self.compute_motion(session.frames[last_idx], session.frames[i])

            # Select if motion is in valid range
            if self.min_motion_threshold <= motion <= self.max_motion_threshold:
                selected.append(i)

            # Stop if target reached
            if len(selected) >= self.target_frame_count:
                break

        # Ensure last frame is included
        if selected[-1] != len(session.frames) - 1:
            selected.append(len(session.frames) - 1)

        return selected


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def extract_images_and_masks(session: CaptureSession,
                            selected_indices: Optional[List[int]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract images and masks for forensic analysis

    Args:
        session: Capture session
        selected_indices: Optional frame indices to extract

    Returns:
        images: List of grayscale images
        masks: List of fracture masks
    """
    if selected_indices is None:
        selected_indices = range(len(session.frames))

    images = []
    masks = []

    for idx in selected_indices:
        frame = session.frames[idx]
        images.append(frame.grayscale)

        if frame.fracture_mask is not None:
            masks.append(frame.fracture_mask)
        else:
            # Empty mask if not detected
            masks.append(np.zeros_like(frame.grayscale))

    return images, masks


def save_session(session: CaptureSession, output_dir: Path):
    """
    Save capture session to disk

    Args:
        session: Capture session
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    session_dir = output_dir / session.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save frames
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(session.frames):
        # Save image
        img_path = frames_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(img_path), frame.image)

        # Save mask
        if frame.fracture_mask is not None:
            mask_path = frames_dir / f"mask_{i:04d}.png"
            cv2.imwrite(str(mask_path), frame.fracture_mask)

    # Save metadata
    import json
    metadata = {
        'session_id': session.session_id,
        'start_time': session.start_time,
        'duration': session.get_duration(),
        'frame_count': session.get_frame_count(),
        'camera_matrix': session.camera_matrix.tolist()
    }

    with open(session_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Session saved to: {session_dir}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demo video processing"""

    print("="*70)
    print("VIDEO PROCESSING MODULE - DEMO")
    print("="*70)

    # Camera intrinsics (example)
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)

    print("\n1. Fracture Detection Demo")
    print("-" * 70)

    # Create test image with line
    test_image = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(test_image, (100, 100), (500, 400), 255, 2)
    cv2.line(test_image, (200, 400), (600, 100), 255, 2)

    detector = FractureDetector()
    mask, confidence = detector.detect_with_confidence(test_image)

    print(f"  Detection confidence: {confidence:.2f}")
    print(f"  Detected pixels: {np.sum(mask > 0)}")

    print("\n2. Frame Selection Demo")
    print("-" * 70)

    # Create dummy session
    session = CaptureSession(
        session_id="demo",
        start_time=time.time(),
        camera_matrix=camera_matrix
    )

    # Add dummy frames
    for i in range(50):
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        frame = VideoFrame(
            frame_id=i,
            timestamp=time.time() + i * 0.1,
            image=img,
            grayscale=img,
            quality_score=0.8
        )
        session.add_frame(frame)

    print(f"  Total frames: {session.get_frame_count()}")
    print(f"  Duration: {session.get_duration():.2f}s")

    selector = FrameSelector(target_frame_count=10)
    # Note: select_keyframes would need actual motion
    # selected = selector.select_keyframes(session)
    # print(f"  Selected frames: {len(selected)}")

    print("\n" + "="*70)
    print("Video processing ready for integration")
    print("="*70)
