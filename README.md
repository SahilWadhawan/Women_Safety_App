# Woman Safety Analytics Project

## SIH_ROUND_2_PPT:
[SIH_ROUND_2_PPT_ALPHA.pptx](https://github.com/user-attachments/files/16936386/SIH_ROUND_2_PPT_ALPHA.pptx)

## SIH_ROUND_2_DEMO_VIDEO:
https://youtu.be/4Jc1AEVYBfg

---

## Steps to Run the Program Locally:

### 1. Clone this repository
Open your terminal and run the following command to clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Install the required dependencies
```bash
pip install opencv-python tensorflow torch torchvision matplotlib numpy scikit-learn Flask twilio
pip install jupyterlab
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
```
### 3. Install YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```
### 4. Install Flask
```bash
pip install Flask
```
### 5. Change UPLOAD_FOLDER in app.py to your 'uploads' folder path.
```bash
UPLOAD_FOLDER = 'your/local/uploads/folder/path'
```
### 6. Run the Flask app
```bash
python app.py
```
