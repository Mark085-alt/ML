from flask import Flask, render_template
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-face')
def run_face():
    script_path = os.path.join('faceRecognition', 'Face_recognisation.py')
    subprocess.run(['python', script_path])
    return 'Face Recognition script executed!'

@app.route('/run-feature')
def run_object():
    script_path = os.path.join('featureExploration', 'main.py')
    subprocess.run(['python', script_path])
    return 'Object Detection script executed!'

@app.route('/run-imageR')
def run_image():
    script_path = os.path.join('Image_recognisation', 'train.py')
    subprocess.run(['python', script_path])
    return 'Image recognition script executed!'

@app.route('/run-dc')
def run_doc():
    script_path = os.path.join('document_characterization', 'demo.py')
    subprocess.run(['python', script_path])
    return 'Document Characterization script executed!'

@app.route('/run-DG')
def run_dg():
    script_path = os.path.join('dialogue_generation', 'main.py')
    subprocess.run(['python', script_path])
    return 'Dialogue generation script executed!'

if __name__ == '__main__':
    app.run(debug=True)
