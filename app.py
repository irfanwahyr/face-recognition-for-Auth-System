from flask import Flask, request, jsonify, send_from_directory
import face_recognition
import numpy as np
import os
import pickle

app = Flask(__name__, static_url_path='/', static_folder='static')

# Memuat model wajah yang sudah ada
if os.path.exists('face_encodings.pkl'):
    with open('face_encodings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/register', methods=['POST'])
def register():
    file = request.files['file']
    name = request.form['name']

    # Simpan gambar ke folder dataset dengan nama yang diberikan oleh user
    image_path = os.path.join('dataset', f'{name}.jpg')
    file.save(image_path)

    # Memproses gambar yang diunggah
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        encoding = face_encodings[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

        # Menyimpan kembali model yang diperbarui
        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)

        return jsonify({"status": "success", "message": "User registered successfully"})
    else:
        return jsonify({"status": "failure", "message": "No face found in the image"})

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['file']

    # Memproses gambar yang diunggah
    image = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"

        face_distance = face_recognition.face_distance(known_face_encodings, encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        return jsonify({"status": "success", "name": name})
    else:
        return jsonify({"status": "failure", "message": "No face found in the image"})

if __name__ == '__main__':
    app.run(debug=True)
