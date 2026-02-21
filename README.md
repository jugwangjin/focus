# Charming Focus (Minimal Release)

이 저장소에는 `/Users/jugwangjin/Desktop/charming_focus/main.py` 기반의 버전 2 구현만을 포함합니다. 비디오 자원, Mediapipe 모델, 기타 파생 스크립트는 배포 대상에 포함돼 있지 않습니다. 아래 내용을 참고해 환경을 구성하고 필요한 설정을 조정하세요.

## 준비물

- Python 3.9 이상 권장
- 필수 패키지
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pillow`
  - (표준 라이브러리) `tkinter`
- Mediapipe face landmarker 모델 파일: `face_landmarker.task`
  - Google Mediapipe 공식 리포지터리에서 다운로드 후, 프로젝트 루트(또는 원하는 경로)에 배치하고 `main.py`에서 로드되는 경로를 맞춰주세요.

## main.py 구성 (버전 2)

- 비디오 재생용 디렉터리는 파일 상단에서 정의합니다.

  ```python
  VIDEOS_DIR_V2 = "videos_2"
  EMPTY_VIDEO_BASENAME = "empty"
  ```

  - 기본 루프용 “empty” 영상은 `VIDEOS_DIR_V2/empty.*` 형태로 준비해야 하며, 음소거 상태로 무한 반복 재생됩니다.
  - 사용자가 10프레임 이상 시선을 벗어나면 동일 디렉터리 내 다른 영상을 무작위로 1회 재생하고, 종료 시 다시 empty 루프로 복귀합니다.
- Mediapipe 추정에서 pitch 축이 부정확하게 계산되던 문제를 해결하기 위해, 고정 오프셋(예: `pitch-30`, `pitch-40`)을 제거하고 실제 회전행렬에서 추출한 각도를 그대로 사용합니다.
- 오디오 재생은 `ffplay`(기본) 또는 macOS의 `afplay` 중 사용 가능한 실행 파일을 통해 진행됩니다.

## 실행 방법

1. 위 준비물을 설치하고 `face_landmarker.task` 모델 파일을 프로젝트에 배치합니다.
2. `videos_2` 폴더를 만들고 `empty` 영상과 재생하고 싶은 다른 영상을 추가합니다.
3. 다음 명령으로 실행합니다.

   ```bash
   python main.py
   ```

카메라가 정상적으로 열리고, empty 영상이 화면에 표시된 상태에서 시선을 10프레임 이상 벗어나면 임의의 컨텐츠 영상이 소리와 함께 재생됩니다.

## 기타 파일

- `main_charming.py`: 기본 “empty” 영상 없이 동작하는 초기 버전입니다. 시선이 벗어나면 지정된 비디오 목록을 순서대로 재생/중지하며, pitch 축 보정 역시 고정 오프셋 없이 회전행렬 기반 계산으로 반영돼 있습니다. 필요 시 참고용으로만 사용하세요.

## 주의사항

- 이 배포본에는 `main.py`만 포함돼 있으므로 비디오 자원과 Mediapipe 모델 파일은 직접 준비해야 합니다.
- 오디오 출력을 위해 `ffplay` 또는 `afplay` 중 하나는 시스템에 설치돼 있어야 합니다.
