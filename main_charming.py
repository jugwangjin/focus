import glob
import math
import os
import subprocess
import tkinter as tk
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageTk

import shutil
import time

CAMERA_WINDOW = "Camera"
VIDEOS_DIR = "videos"

HEAD_TURN_THRESHOLD_DEG = 20.0
GAZE_TURN_THRESHOLD_DEG = 20.0
FRAME_JITTER_THRESHOLD = 5


num_distracted = 0


FACE_LANDMARK_IDS = [33, 263, 1, 61, 291, 199]
FACE_3D_MODEL_POINTS = np.array(
    [
        (-30.0, 0.0, -30.0),
        (30.0, 0.0, -30.0),
        (0.0, 0.0, 0.0),
        (-20.0, -35.0, -20.0),
        (20.0, -35.0, -20.0),
        (0.0, -65.0, -5.0),
    ],
    dtype=np.float64,
)


LEFT_EYE_CORNER_IDS = [33, 133]
RIGHT_EYE_CORNER_IDS = [362, 263]
LEFT_IRIS_IDS = [468, 469, 470, 471]
RIGHT_IRIS_IDS = [473, 474, 475, 476]


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass
class GazeDirection:
    horizontal: float
    vertical: float


class FaceOrientationEstimator:
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_counter = 0

    def process(self, frame_bgr: np.ndarray) -> Optional[Any]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._frame_counter += 1
        timestamp_ms = self._frame_counter * 33
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.face_landmarks and len(result.face_landmarks) > 0:
            return result.face_landmarks[0]
        return None

    @staticmethod
    def estimate_head_pose(
        landmarks: Any,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[HeadPose]:
        height, width, _ = frame_shape
        image_points = []
        for idx in FACE_LANDMARK_IDS:
            lm = landmarks[idx]
            image_points.append((lm.x * width, lm.y * height))

        if len(image_points) != len(FACE_LANDMARK_IDS):
            return None

        image_points = np.array(image_points, dtype=np.float64)

        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            FACE_3D_MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return FaceOrientationEstimator._rotation_matrix_to_head_pose(rotation_matrix)

    @staticmethod
    def _rotation_matrix_to_head_pose(rotation_matrix: np.ndarray) -> HeadPose:
        correction = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        corrected = rotation_matrix @ correction

        sy = math.hypot(corrected[0, 0], corrected[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(corrected[2, 1], corrected[2, 2])
            yaw = math.atan2(-corrected[2, 0], sy)
            roll = math.atan2(corrected[1, 0], corrected[0, 0])
        else:
            pitch = math.atan2(-corrected[1, 2], corrected[1, 1])
            yaw = math.atan2(-corrected[2, 0], sy)
            roll = 0.0

        return HeadPose(
            yaw=FaceOrientationEstimator._normalize_angle(math.degrees(yaw)),
            pitch=FaceOrientationEstimator._normalize_angle(math.degrees(pitch)),
            roll=FaceOrientationEstimator._normalize_angle(math.degrees(roll)),
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        wrapped = (angle + 180.0) % 360.0 - 180.0
        if wrapped == -180.0:
            return 180.0
        return wrapped


class GazeEstimator:
    @staticmethod
    def _landmark_to_point(
        landmark: Any,
        frame_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        height, width, _ = frame_shape
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    @classmethod
    def estimate(
        cls,
        landmarks: Any,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[GazeDirection]:
        try:
            left_eye = [cls._landmark_to_point(landmarks[idx], frame_shape) for idx in LEFT_EYE_CORNER_IDS]
            right_eye = [cls._landmark_to_point(landmarks[idx], frame_shape) for idx in RIGHT_EYE_CORNER_IDS]
            left_iris = [cls._landmark_to_point(landmarks[idx], frame_shape) for idx in LEFT_IRIS_IDS]
            right_iris = [cls._landmark_to_point(landmarks[idx], frame_shape) for idx in RIGHT_IRIS_IDS]
        except (IndexError, KeyError):
            return None

        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2.0

        left_iris_center = np.mean(left_iris, axis=0)
        right_iris_center = np.mean(right_iris, axis=0)
        iris_center = (left_iris_center + right_iris_center) / 2.0

        eye_width = np.linalg.norm(right_eye_center - left_eye_center)
        eye_height = eye_width * 0.5 + 1e-6

        offset = iris_center - eye_center
        norm_horizontal = float(offset[0] / max(eye_width, 1e-6))
        norm_vertical = float(offset[1] / eye_height)

        horizontal_deg = np.degrees(np.arctan(norm_horizontal)) * 60.0 / 45.0
        vertical_deg = np.degrees(np.arctan(norm_vertical)) * 60.0 / 45.0

        return GazeDirection(horizontal=horizontal_deg, vertical=vertical_deg)


class AttentionState:
    def __init__(self) -> None:
        self.out_counter = 0
        self.in_counter = 0
        self.is_attentive = True

    def update(self, is_attentive: bool) -> Tuple[bool, bool]:
        if is_attentive:
            self.in_counter += 1
            self.out_counter = 0
        else:
            self.out_counter += 1
            self.in_counter = 0

        became_inattentive = False
        became_attentive = False

        if self.out_counter >= FRAME_JITTER_THRESHOLD and self.is_attentive:
            self.is_attentive = False
            became_inattentive = True
        elif self.in_counter >= FRAME_JITTER_THRESHOLD and not self.is_attentive:
            self.is_attentive = True
            became_attentive = True

        return became_inattentive, became_attentive


class TkinterVideoPlayer:
    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("Video Player")
        self._root.withdraw()
        self._label = tk.Label(self._root)
        self._label.pack()
        self._capture: Optional[cv2.VideoCapture] = None
        self._current_path: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._frame_interval_sec: float = 1.0 / 30.0
        self._after_id: Optional[str] = None
        self._audio_process: Optional[subprocess.Popen] = None
        self._next_frame_time: float = time.perf_counter()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        screen_width = self._root.winfo_screenwidth()
        video_x = int(screen_width * 0.05)
        video_y = 100
        self._root.geometry(f"+{video_x}+{video_y}")

    @property
    def is_playing(self) -> bool:
        return self._capture is not None

    def play(self, path: str, fps_hint: Optional[float] = None) -> bool:
        self.stop()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            print(f"[TkinterVideoPlayer] Failed to open video: {path}")
            return False

        fps = fps_hint
        if fps is None or not math.isfinite(fps) or fps <= 1e-3:
            fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 1e-3:
            fps = 30.0
            print("USING DEFAULT")
        print(fps)
        self._frame_interval_sec = max(1.0 / fps, 1.0 / 120.0)
        self._next_frame_time = time.perf_counter() + self._frame_interval_sec

        self._capture = capture
        self._current_path = path
        self._root.deiconify()
        self._root.lift()

        self._start_audio(path)
        self._schedule_next_frame()
        return True

    def stop(self) -> None:
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._stop_audio()

        self._current_path = None
        self._label.configure(image="")
        self._label.image = None
        self._photo = None
        self._root.withdraw()

    def process_events(self) -> None:
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self.stop()

    def destroy(self) -> None:
        self.stop()
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    def _schedule_next_frame(self) -> None:
        if self._capture is None:
            return
        delay_sec = max(0.0, self._next_frame_time - time.perf_counter())
        delay_ms = max(int(delay_sec * 1000), 1)
        self._after_id = self._root.after(delay_ms, self._update_frame)

    def _update_frame(self) -> None:
        if self._capture is None:
            return

        now = time.perf_counter()
        if now > self._next_frame_time + self._frame_interval_sec:
            behind = now - self._next_frame_time
            frames_to_skip = min(int(behind / self._frame_interval_sec), 30)
            for _ in range(frames_to_skip):
                if not self._capture.grab():
                    self.stop()
                    return
            self._next_frame_time = now

        ret, frame = self._capture.read()
        if not ret:
            self.stop()
            return

        photo = self._frame_to_photo_image(frame)
        if photo is None:
            self.stop()
            return

        self._photo = photo
        self._label.configure(image=photo)
        self._label.image = photo

        display_time = time.perf_counter()
        if self._next_frame_time <= display_time:
            self._next_frame_time = display_time + self._frame_interval_sec
        else:
            self._next_frame_time += self._frame_interval_sec
        self._schedule_next_frame()

    def _on_close(self) -> None:
        self.stop()

    @staticmethod
    def _frame_to_photo_image(frame: np.ndarray) -> Optional[ImageTk.PhotoImage]:
        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (width // 3, height // 3))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        try:
            return ImageTk.PhotoImage(image=image)
        except tk.TclError as exc:
            print(f"[TkinterVideoPlayer] Unable to display frame: {exc}")
            return None

    def _start_audio(self, path: str) -> None:
        self._stop_audio()
        if self._launch_ffplay(path):
            return
        self._launch_afplay(path)

    def _stop_audio(self) -> None:
        if self._audio_process is None:
            return
        if self._audio_process.poll() is None:
            self._audio_process.terminate()
            try:
                self._audio_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._audio_process.kill()
        self._audio_process = None

    def _launch_ffplay(self, path: str) -> bool:
        if shutil.which("ffplay") is None:
            return False
        try:
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._audio_process = proc
            return True
        except Exception as exc:
            print(f"[TkinterVideoPlayer] Failed to start ffplay audio: {exc}")
            return False

    def _launch_afplay(self, path: str) -> None:
        if shutil.which("afplay") is None:
            print("[TkinterVideoPlayer] No audio player available; audio disabled")
            self._audio_process = None
            return
        try:
            self._audio_process = subprocess.Popen(["afplay", path])
        except Exception as exc:
            print(f"[TkinterVideoPlayer] Failed to start afplay audio: {exc}")
            self._audio_process = None


class VideoManager:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.video_entries: List[Tuple[str, Optional[float]]] = self._load_video_entries(directory)
        self.current_path: Optional[str] = None
        self._player = TkinterVideoPlayer()

    @staticmethod
    def _load_video_entries(directory: str) -> List[Tuple[str, Optional[float]]]:
        patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
        videos: List[Tuple[str, Optional[float]]] = []
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                videos.append((path, VideoManager._read_fps(path)))
        return videos

    @staticmethod
    def _read_fps(path: str) -> Optional[float]:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if not fps or not math.isfinite(fps) or fps <= 1e-3:
            return None
        return float(fps)

    @property
    def is_playing(self) -> bool:
        return self._player.is_playing

    def start_random_video(self) -> None:
        global num_distracted
        if not self.video_entries:
            print("[VideoManager] No videos found in directory:", self.directory)
            return

        index = num_distracted % len(self.video_entries)
        path, fps = self.video_entries[index]
        self.stop_video()

        if self._player.play(path, fps_hint=fps):
            self.current_path = path
            num_distracted += 1
            print(f"[VideoManager] Playing video: {path}")
        else:
            self.current_path = None

    def stop_video(self) -> None:
        if self._player.is_playing:
            self._player.stop()
        self.current_path = None

    def update(self, inattentive: bool) -> None:
        was_playing = self.is_playing
        self._player.process_events()

        if was_playing and not self.is_playing and inattentive:
            self.start_random_video()

    def shutdown(self) -> None:
        self.stop_video()
        self._player.destroy()


def attention_check(head_pose: Optional[HeadPose], gaze: Optional[GazeDirection]) -> bool:
    if head_pose is None or gaze is None:
        return False

    if abs(head_pose.yaw) > HEAD_TURN_THRESHOLD_DEG:
        return False
    if abs(head_pose.pitch-30) > HEAD_TURN_THRESHOLD_DEG:
        return False
    if abs(gaze.horizontal) > GAZE_TURN_THRESHOLD_DEG:
        return False
    if abs(gaze.vertical) > GAZE_TURN_THRESHOLD_DEG:
        return False
    return True


def annotate_frame(frame: np.ndarray, landmarks: Any, head_pose: Optional[HeadPose], gaze: Optional[GazeDirection], attentive: bool) -> None:
    if landmarks is not None:
        height, width, _ = frame.shape
        for lm in landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    status_text = "ATTENTIVE" if attentive else "DISTRACTED"
    color = (0, 200, 0) if attentive else (0, 0, 255)
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    if head_pose:
        cv2.putText(
            frame,
            f"Head yaw: {head_pose.yaw:.1f} pitch: {head_pose.pitch:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    if gaze:
        cv2.putText(
            frame,
            f"Gaze h: {gaze.horizontal:.1f} v: {gaze.vertical:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )


def main() -> None:
    video_manager = VideoManager(VIDEOS_DIR)
    face_estimator = FaceOrientationEstimator()
    attention_state = AttentionState()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Cannot open default camera (index 0)")

    ret, test_frame = capture.read()
    if ret:
        original_height, original_width = test_frame.shape[:2]
        camera_width = original_width // 2
        camera_height = original_height // 2
    else:
        camera_width = 640
        camera_height = 480

    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW, camera_width, camera_height)
    
    temp_root = tk.Tk()
    screen_width = temp_root.winfo_screenwidth()
    temp_root.destroy()
    
    camera_x = int(screen_width * 0.55)
    camera_y = 100
    cv2.moveWindow(CAMERA_WINDOW, camera_x, camera_y)

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("[Main] Failed to grab frame from camera")
                break

            landmarks = face_estimator.process(frame)
            head_pose = None
            gaze_direction = None

            if landmarks is not None:
                head_pose = face_estimator.estimate_head_pose(landmarks, frame.shape)
                gaze_direction = GazeEstimator.estimate(landmarks, frame.shape)

            is_attentive_now = attention_check(head_pose, gaze_direction)
            became_inattentive, became_attentive = attention_state.update(is_attentive_now)

            if became_inattentive and not video_manager.is_playing:
                video_manager.start_random_video()
            if became_attentive and video_manager.is_playing:
                video_manager.stop_video()

            video_manager.update(not attention_state.is_attentive)

            height, width = frame.shape[:2]
            resized_frame = cv2.resize(frame, (width // 2, height // 2))
            
            annotate_frame(resized_frame, landmarks, head_pose, gaze_direction, attention_state.is_attentive)
            cv2.imshow(CAMERA_WINDOW, resized_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        capture.release()
        video_manager.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
