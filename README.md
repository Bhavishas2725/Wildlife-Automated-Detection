## Automated Wildlife Detection & KDD System

An end-to-end wildlife monitoring pipeline that combines 
YOLOv8n for real-time animal detection with a custom-trained 
7-layer CNN classifier for 9-class species identification 
(bear, deer, donkey, giraffe, horse, lion, monkey, tiger, unknown).

### Features
- Real-time bounding-box detection using YOLOv8n
- Custom CNN trained from scratch — 92% validation accuracy
- Automated image preprocessing (grayscale, blur, edge detection)
- KDD database generation with structured metadata CSV
- Auto-sorts detected images into labelled species folders

### Tech Stack
Python | TensorFlow | YOLOv8 | OpenCV | Pandas | NumPy
