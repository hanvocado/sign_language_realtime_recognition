# Sign Language Recognition (MediaPipe Holistic + LSTM)

**Mục tiêu:** xây dựng pipeline chuyển video -> landmark (.npy) -> huấn luyện LSTM nhận diện ký hiệu (face + hands) -> inference realtime.

## Cấu trúc project

```
sign-language-recognition/
├── data/
├── models/
│   └── checkpoints/    # saved model checkpoints + label map
├── src/                # scripts & modules
├── requirements.txt
└── README.md
```

## Quick start (example)

1. Cài dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Thu thập video

- Đặt video gốc vào `data/wlvsl/raw_unprocessed/<label>/`.
- Mỗi `<label>` là một class (ví dụ: `hello`, `thanks`, ...).
- Mỗi clip chỉ nên chứa 1 ký hiệu, thời lượng 2–5 giây.

3. Chuẩn hóa video
   Dùng script `preprocess_videos.py` để:

- Đưa video về **30fps, 1280×720 (16:9)**.
- Tự động phát hiện chuyển động và cắt video.
- **Pixel value normalization** về range [0,1] sử dụng min-max normalization.
- Tùy chọn `--skip_existing` để bỏ qua các video đã được cắt (đã tồn tại file output trong `data/raw`).

  ```bash
  python -m src.preprocess.preprocess_video --input_dir data/wlvsl/raw_unprocessed --output_dir data/wlvsl/raw
  ```

4. Convert tất cả video sang npy:

   ```bash
   python -m src.preprocess.video2npy --input_dir data/wlvsl/raw --output_dir data/wlvsl/npy
   ```

5. Split dataset into train/val/test:

   ```bash
   python -m src.preprocess.split_dataset --data_dir data/wlvsl/npy --output_dir data/wlvsl/splits
   ```

6. Huấn luyện và đánh giá:

   ```bash
   python -m src.model.train --data_dir data/wlvsl/npy --source npy --ckpt_dir models/checkpoints/vsl_v1
   ```

7. Realtime inference:

   ```bash
   python -m src.infer_realtime --ckpt models/checkpoints/vsl_v1/best.pth --label_map models/checkpoints/vsl_v1/label_map.json
   ```

8. **Web Application (Webapp):**

   Ứng dụng web real-time nhận diện ngôn ngữ kí hiệu thông qua webcam.

   **Tính năng:**
   - Real-time video capture từ webcam
   - Phát hiện chuyển động (Motion Detection FSM)
   - Nhận diện ký hiệu tự động
   - Hiển thị kết quả với độ tin cậy (confidence)
   - Lịch sử dự đoán

   **Cách chạy:**

   a. Cài đặt dependencies cho webapp (nếu chưa có):

   ```bash
   pip install -r requirements.txt
   ```

   b. Khởi động Flask server:

   ```bash
   python .\src\webapp\server.py
   ```

   c. Mở browser và truy cập: `http://localhost:5000`

   d. Cho phép truy cập webcam khi popup yêu cầu.

   **Cấu hình Webapp:**

   File `webapp/config.py` chứa các tham số:
   - `MODEL_PATH`: Đường dẫn đến model checkpoint (mặc định: `models/checkpoints/vsl_v1/best.pth`)
   - `LABEL_MAP_PATH`: Đường dẫn đến label map (mặc định: `models/checkpoints/vsl_v1/label_map.json`)
   - `MIN_PREDICTION_CONFIDENCE`: Ngưỡng độ tin cậy tối thiểu (mặc định: 0.35)
   - `STILL_FRAMES_REQUIRED`: Số frame không chuyển động để kết thúc ghi nhận (mặc định: 8)

   **Cấu trúc Webapp:**

   ```
   webapp/
   ├── server.py              # Flask server chính
   ├── config.py              # Cấu hình
   ├── static/
   │   ├── app.js            # Logic capture & Socket.IO
   │   └── style.css         # CSS
   └── templates/
       └── index.html        # HTML giao diện
   ```

## Ghi chú kỹ thuật

- Feature vector hiện tại: **pose (33*3) + left hand (21*3) + right hand (21\*3) = 225** chiều.
- Tất cả file `.npy` được lưu dạng `(seq_len, 225)`.
- Coordinate normalization sử dụng wrist joints làm reference point theo công thức: L̂_t = (L_t - L_ref) / ||L_max - L_min||
- `train.py` lưu `label_map.json` (list labels theo thứ tự index) trong folder checkpoint để inference tải lại mapping.

## Cấu trúc lakehouse

Bucket chính: `data` — các prefix được tham số hóa theo `DATASET_NAME` (mặc định `wlasl`) và `SEQ_LEN` (mặc định `25`).

```
data/
└── lakehouse/
      ├── system/
      │   └── iceberg/                          # Iceberg catalog metadata (PostgreSQL-backed)
      ├── bronze/
      │   └── <dataset>/
      │       └── videos/
      │           ├── raw/                      # local dataset ingest
      │           │   └── <yyyymm>/<yyyymmdd>/<label>/<filename>.mp4
      │           └── user_uploads/             # webapp user uploads
      │               └── <yyyymm>/<yyyymmdd>/<label>/<filename>.mp4
      ├── silver/
      │   └── <dataset>/
      │       └── videos/
      │           └── preprocessed/             # normalized & segmented videos
      │               └── <run_month>/<run_stamp>/<label>/<segment>.mp4
      └── gold/
          └── <dataset>/
              └── npy/
                  └── <seq_len>/                # landmark arrays
                      └── <gold_version>/       # vNNNN (e.g. v0001)
                          └── <label>/<segment>.npy

mlflow/
└── ...   (artifact root nằm trong bucket `mlflow`)
```

Ý nghĩa:

- **Bronze** (`lakehouse/bronze/<dataset>/videos/`): dữ liệu thô gồm `raw` (local dataset ingest) và `user_uploads` (webapp).
  Partition theo `yyyymm/yyyymmdd/label` để cân bằng giữa hiệu năng và quản lý dữ liệu.
- **Silver** (`lakehouse/silver/<dataset>/videos/preprocessed/`): video đã normalize (resize, fps sync) và segment, tổ chức theo `run_month/run_stamp/label`.
- **Gold** (`lakehouse/gold/<dataset>/npy/<seq_len>/<gold_version>/`): full snapshot landmark `.npy` theo label.
  Mỗi version mới copy toàn bộ snapshot cũ + merge landmarks mới, đảm bảo mỗi version là self-contained.
- **System** (`lakehouse/system/iceberg/`): metadata + manifests của Iceberg tables (catalog: PostgreSQL), tách riêng khỏi Medallion business layers.

Versioning hiện tại:

- Iceberg tables theo medallion:
  - `bronze_user_upload_inventory` — theo dõi tất cả Bronze objects đã ingest (dùng làm checkpoint skip duplicate).
  - `silver_raw_inventory` — theo dõi Silver preprocessed videos.
  - `gold_training_landmarks_inventory` — theo dõi Gold landmark snapshots.
- Gold dataset version (`v0001`, `v0002`, ...): tăng tự động mỗi lần pipeline tạo thêm landmarks mới, được lưu trong `gold_version_state.json`.
- MLflow log kèm `dataset_version` và artifacts model/eval.


## Preprocessing Pipeline
Luồng xử lý (`preprocessing_pipeline` DAG — ref: `airflow/dags/preprocessing_pipeline.py`):

```
start
  └─► [Bronze] bronze_prepare_run_context
        └─► bronze_ingest_local_raw
              └─► bronze_collect_unprocessed_inputs
                    └─► [Silver] silver_preprocess_videos
                          └─► [Gold] gold_extract_landmarks
                                └─► gold_merge_snapshot
                                      └─► end
```

1. **`bronze_prepare_run_context`**: tạo run metadata (`run_id`, `run_month`, `run_stamp`, `run_dir`) dùng chung cho toàn pipeline.
2. **`bronze_ingest_local_raw`**: upload local raw videos vào `bronze/<dataset>/videos/raw/<yyyymm>/<yyyymmdd>/<label>/...`.
3. **`bronze_collect_unprocessed_inputs`**: load Iceberg checkpoint (via Spark) để xác định objects đã xử lý, scan tất cả Bronze sources (`raw` + `user_uploads`), chỉ download objects chưa xử lý (skip theo `etag` hoặc `object_path`) vào staging dir.
4. **`silver_preprocess_videos`**: chạy `preprocess_video.py` (normalize 30fps, 1280x720, motion detection, segment) trên staging dir, output vào Silver local dir.
5. **`gold_extract_landmarks`**: chạy `video2npy.py` (MediaPipe Holistic landmark extraction, seq_len padding/truncation) trên Silver videos, output `.npy`.
6. **`gold_merge_snapshot`**: upload Silver videos + Gold landmarks lên MinIO, nếu có landmarks mới thì tạo Gold snapshot version mới (copy previous + merge new), append inventory vào Iceberg tables (Bronze, Silver, Gold) via Spark, tạo preprocessing manifest JSON.
7. Training pipeline đọc bản Gold mới nhất, train model và log lên MLflow.