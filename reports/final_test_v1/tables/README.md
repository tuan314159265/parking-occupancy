# Final-test results

Configuration selected on validation; no test parameter search.

| Method | F1 | Accuracy | Coverage | Vacant MAE | ms/image |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.9654 | 0.9126 | 0.9353 | 4.681 | 167.8 |
| YOLO full 640 | 0.3679 | 0.6767 | 0.9015 | 15.807 | 165.3 |
| Hybrid | 0.6517 | 0.7583 | 0.9026 | 11.400 | 386.5 |
| YOLO full pixel-matched | 0.6249 | 0.7487 | 0.8955 | 11.563 | 377.9 |

Timing is the original-image pipeline, including all tiles. The model stage includes Ultralytics internal preprocessing/NMS and hybrid mapping, not just neural forward time.

Read the CPU/GPU execution metadata before comparing these times with validation.
