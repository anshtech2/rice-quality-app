from flask import Flask, render_template, request
import os
import cv2
from rice_detector import detect_rice

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    output_path, count_broken, count_whole = detect_rice(filepath)

    return render_template('result.html', 
                           uploaded_image=output_path, 
                           broken=count_broken, 
                           whole=count_whole)

if __name__ == '__main__':
    app.run(debug=True)
