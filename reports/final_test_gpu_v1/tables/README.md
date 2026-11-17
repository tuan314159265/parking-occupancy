# Final-test results

Configuration selected on validation; no test parameter search.

The legacy notebook had already exposed the test split; this is not an unseen blind holdout.

| Method | F1 | Accuracy | Coverage | Vacant MAE | ms/image |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.9654 | 0.9126 | 0.9353 | 4.681 | 347.9 |
| YOLO full 640 | 0.3679 | 0.6767 | 0.9015 | 15.807 | 170.1 |
| Hybrid | 0.6517 | 0.7583 | 0.9026 | 11.400 | 210.5 |
| YOLO full pixel-matched | 0.6249 | 0.7487 | 0.8955 | 11.563 | 185.3 |

Timing is the original-image pipeline, including all tiles. The model stage includes Ultralytics internal preprocessing/NMS and hybrid mapping, not just neural forward time.

Read the CPU/GPU execution metadata before comparing these times with validation.
