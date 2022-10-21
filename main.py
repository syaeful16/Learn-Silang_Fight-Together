from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model = load_model('learnsilang.h5')

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"),
        metrics=['accuracy']        
        )

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

def prediksi(image):
    image = cv2.resize(image, (64, 64))
    image = image.reshape(-1, 64, 64, 1).astype(float)
    prediction = np.argmax(model.predict(image))

    return CATEGORIES[prediction]

def detectHandPredict(frame):
    results = hands.process(frame)
    h, w, _ = frame.shape

    hands_status = {'Right': False, 'Left': False, 'Right_index' : None, 'Left_index': None}

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.append((int(landmark.x * w), int(landmark.y * h), (landmark.z * w)))

            x_coordinates = np.array(landmarks)[:,0]
            y_coordinates = np.array(landmarks)[:,1]
            x1  = int(np.min(x_coordinates) - 10)
            y1  = int(np.min(y_coordinates) - 10)
            x2  = int(np.max(x_coordinates) + 10)
            y2  = int(np.max(y_coordinates) + 10)

            for id_hand, hand_info in enumerate(results.multi_handedness):
                hand_type = hand_info.classification[0].label
                hands_status[hand_type] = True

                if hands_status[hand_type]:
                    # cv2.putText(frame, hand_type + ' Hand Detected', (10, (hand_index+1) * 30),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                    cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (155, 0, 255), 2, cv2.LINE_8)
                    crop = frame[(y1-40):(y2+40), (x1-40):(x2+40)]

                    # Prediksi
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    result = prediksi(crop)
                    return result

            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS, 
            mpDraw.DrawingSpec(color=(85, 255, 211), thickness=2, circle_radius=2))


camm = cv2.VideoCapture(1)

Aplhabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
def webImage():
    count = 0
    resultTrue = 0
    stat = ''
    while(True):
        status, frame = camm.read()
        frame = cv2.flip(frame, 1)

        y, x, _ = frame.shape

        # print(str(x) + ' ' + str(y))
        
        detectHandPredict(frame)

        if status:
            count += 1
        
        # print(count)
        if count%45 == 0:
            hasil = detectHandPredict(frame)
            if hasil == Aplhabet[resultTrue]:
                print(Aplhabet[resultTrue] + " : " + str(hasil) + " benar")
                stat = 'Benar'
                resultTrue += 1
            else:
                stat = 'Salah'
                print(Aplhabet[resultTrue] + " : " + str(hasil) + " salah")
        cv2.putText(frame, stat, (40, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.rectangle(frame, (x-90, 85), (x-20, 155),(238, 249, 252), -1, cv2.LINE_8)
        cv2.putText(frame, Aplhabet[resultTrue], (x-70, 135), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
        

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/menus.html', methods=['POST', 'GET'])
def menus():
    if request.method == 'POST':
        cat = request.form['alp']
        # angka = request.form['angka']
        return redirect(url_for('camera', category = cat))
    return render_template('menus.html')

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    cat = request.args.get('category')
    # resultTrue = 0
    # if cat == 'alphabet':
    #     while(True):
    #         if resultPredict() == Aplhabet[resultTrue]:
    #             # print(Aplhabet[resultTrue] + " : " + "benar")
    #             resultTrue += 1
                # print(resultTrue)
            # else: 
                # print(Aplhabet[resultTrue] + " : " + "salah")

    return render_template('camera.html', alph = cat)


@app.route('/webcam')
def webcam():
    return Response(webImage(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()