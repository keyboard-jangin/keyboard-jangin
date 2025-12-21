# Glass Fracture Forensic System - 개선 영역 분석 보고서

## 현재 구현 상태 분석

### 🔴 우선순위 1: Feature Tracking (_extract_tracks)
**위치**: `src/glass_fracture_forensics/forensic_system.py:871-886`

**현재 상태**: 더미 데이터 반환
```python
def _extract_tracks(self, images: List[np.ndarray],
                   masks: List[np.ndarray]) -> List[Track2D]:
    """Extract 2D tracks from image sequence (placeholder)"""
    # This would use actual KLT tracking
    # For now, return dummy tracks
    tracks = []
    for i in range(10):
        points = np.random.rand(10, 2) * 100  # 랜덤 포인트
        fb_errors = np.random.rand(10) * 0.5   # 랜덤 에러
        # ...
```

**필요한 구현**:
- 실제 Good Features to Track 감지 (Shi-Tomasi corner detection)
- 프랙처 마스크 영역에서 특징점 추출
- KLT 광학 흐름을 사용한 프레임 간 추적
- Forward-Backward 검증 (이미 FeatureTracker 클래스에 구현됨)
- 긴 트랙 체인 생성 및 관리

**임팩트**: 높음 - 전체 파이프라인의 입력 데이터 품질 결정


---

### 🟡 우선순위 2: 3D Trajectory Reconstruction (_reconstruct_trajectories) ⭐
**위치**: `src/glass_fracture_forensics/forensic_system.py:888-902`

**현재 상태**: 더미 3D 궤적 반환
```python
def _reconstruct_trajectories(self, tracks: List[Track2D],
                             K: np.ndarray) -> List[Trajectory3D]:
    """Reconstruct 3D trajectories from tracks (placeholder)"""
    # This would use actual reconstruction
    # For now, return dummy trajectories
    trajectories = []
    for i in range(3):
        points_3d = np.random.rand(20, 3) * 10  # 랜덤 3D 포인트
```

**필요한 구현**:
1. **트랙 그룹화**: 각 프랙처 라인별로 2D 트랙 분류
2. **페어와이즈 재구성**:
   - 연속 프레임 쌍에서 Essential Matrix 계산
   - 상대 포즈 복구 (이미 RelativeReconstructor에 구현됨)
   - 삼각측량으로 3D 포인트 생성
3. **멀티뷰 통합**:
   - 여러 뷰의 3D 포인트 병합
   - 일관성 검사 및 아웃라이어 제거
4. **궤적 구성**:
   - 3D 포인트를 연속적인 궤적으로 정렬
   - 각 궤적에 대한 품질 메트릭 계산

**임팩트**: 매우 높음 - 원점 추정 및 분류의 핵심 입력

**복잡도**: 중간~높음


---

### 🟢 우선순위 3: Capture Quality Validation
**위치**: `src/glass_fracture_forensics/forensic_system.py:310-316`

**현재 상태**: 하드코딩된 플레이스홀더 값
```python
# Compute spatial coverage (simplified - would use actual grid)
coverage_fraction = min(1.0, n_valid / 20.0)  # Placeholder

# Estimate parallax (simplified - would compute from tracks)
mean_parallax = 10.0  # Placeholder [degrees]
```

**필요한 구현**:
1. **실제 Parallax 계산**:
   - 트랙의 시작과 끝 위치 차이 분석
   - 깊이 추정을 위한 기준선/깊이 비율 계산
   - 각 트랙의 parallax angle 계산

2. **공간 커버리지 계산**:
   - 이미지를 그리드로 분할 (예: 4x4)
   - 각 그리드 셀의 트랙 분포 확인
   - 커버된 셀의 비율 계산

**임팩트**: 중간 - 불확실성 정량화 개선

**복잡도**: 낮음


---

### 🟢 우선순위 4: Fracture Mechanics Analysis
**위치**: `src/glass_fracture_forensics/forensic_system.py:672-673`

**현재 상태**: 고정된 branching angle
```python
# Estimate branching angle (simplified - would need reference direction)
theta = np.pi / 6  # Placeholder: 30 degrees
```

**필요한 구현**:
1. **참조 방향 설정**:
   - 원점에서 각 궤적으로의 주 응력 방향 추정
   - Mode I (opening) 방향 결정

2. **실제 Branching Angle 계산**:
   - 궤적 방향과 참조 방향 간 각도 계산
   - 각 분기점의 각도 분석

**임팩트**: 중간 - 응력 강도 계산 정확도

**복잡도**: 중간


---

### 🟢 우선순위 5: Failure Mode Classification
**위치**: `src/glass_fracture_forensics/forensic_system.py:742-743`

**현재 상태**: 단순화된 branch density
```python
# Branch density (simplified - would compute actual density)
branch_density = len(trajectories)  # Placeholder
```

**필요한 구현**:
- 실제 공간 밀도 계산 (branches per unit area)
- 원점 주변 국부 밀도 분석
- 방사형 분포 패턴 분석

**임팩트**: 낮음 - 분류 정확도 개선

**복잡도**: 낮음


---

## 권장 개선 순서

### Phase 1 (핵심 기능)
1. ✅ **Feature Tracking 구현** (우선순위 1)
2. ⭐ **3D Reconstruction 구현** (우선순위 2) - **현재 타겟**

### Phase 2 (품질 향상)
3. **Capture Validation 구현** (우선순위 3)
4. **Fracture Mechanics 개선** (우선순위 4)

### Phase 3 (정확도 향상)
5. **Classification 개선** (우선순위 5)


---

## 두 번째 우선순위 세부 분석: 3D Reconstruction

### 현재 문제점
- 랜덤 3D 포인트 생성으로 의미 없는 결과
- 실제 프랙처 기하학 반영 안 됨
- Origin estimation이 무작위 데이터 기반

### 구현 계획

#### Step 1: Track Segmentation
각 프랙처 라인을 개별 트랙 그룹으로 분리
- DBSCAN 또는 Connected Components 사용
- 공간적 근접성 기반 클러스터링

#### Step 2: Pairwise Reconstruction
기존 `RelativeReconstructor` 활용
```python
reconstructor = RelativeReconstructor(self.thresholds)
for i in range(len(images)-1):
    points_3d, quality = reconstructor.reconstruct(
        points1, points2, K
    )
```

#### Step 3: Multi-view Integration
- 여러 프레임 쌍의 3D 포인트 병합
- Scale ambiguity 해결 (상대적 스케일 통일)
- RANSAC으로 아웃라이어 제거

#### Step 4: Trajectory Construction
- 3D 포인트를 시간/공간 순서로 정렬
- Trajectory3D 객체 생성
- 품질 메트릭 계산 (재투영 오차, 일관성 등)

### 예상 난이도
- 중간~높음
- 구현 시간: 2-3일 (경험 있는 개발자 기준)
- 테스트 시간: 1-2일

### 의존성
- ✅ RelativeReconstructor (이미 구현됨)
- ⚠️ Feature Tracking (우선순위 1 완료 권장)
- ✅ Trajectory3D (이미 구현됨)


---

## 결론

**두 번째로 강화할 부분**: `_reconstruct_trajectories` 메서드

이 부분을 구현하면:
- Origin estimation이 실제 데이터로 작동
- 불확실성 정량화가 의미 있어짐
- 전체 파이프라인이 실용적으로 변환됨

하지만 최적의 결과를 위해서는 **Feature Tracking (우선순위 1)을 먼저 구현**하는 것을 강력히 권장합니다.
