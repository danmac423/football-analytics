# ⚽ Football Analytics

## Overview 

This project provides a comprehensive analysis of football matches from video recordings, using computer vision and machine learning to extract tactical insights and player performance data. It was developed by Dominika Boguszewska, Daniel Machniak, and Natalia Pieczko.

## ✨ **Core Features**
- **Player and Ball Tracking**: Automatically identifies and tracks players, goalkeepers, referees, and the ball in real-time.
- **Performance Metrics**:
  - Calculates the speed of each player.
  - Measures the total distance covered by each player.
- **Advanced Visualizations**:
  - Overlays player and ball tracking directly onto the video.
  - Generates a 2D tactical radar for a bird's-eye view of player and ball positions.

## 🛠️ **Technology Stack**
- **Core**: Python 3.12
- **Computer Vision & Machine Learning**:
  - `PyTorch`
  - `YOLO (You Only Look Once)`
  - `SigLIP`
  - `scikit-learn`
  - `Supervision`
- **Backend**: `gRPC` for microservices
- **Development Tools**:
  - `Jupyter Notebooks`
  - `make`
  - `black & flake8`
  - `venv`

## 🚀 **Getting Started**

### Prerequisites
- Python 3.12
- `uv`

### Installation
1. Clone the repository:
    ```bash
    git clone https://gitlab-stud.elka.pw.edu.pl/dmachnia/football-analytics.git
    cd football-analytics
   ```
2. Create a virtual environment:
    ```bash
    make env
    source .venv/bin/activate
    ```
3. Download pre-trained models:
    ```bash
    make download-models
    ```

### Running the Services
The system uses a set of microservices that need to be running to process videos.

```bash
make full-stack
```

## 🎥 **Results**

![Football Analytics Demo](media/demo.gif)


