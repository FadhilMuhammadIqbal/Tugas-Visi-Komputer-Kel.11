from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pyrebase
import time
# from gpiozero import LED
from time import sleep


# import drivers

# gpioPin1 = LED(17)
# gpioPin2 = LED(27)
# gpioPin3 = LED(22)
# gpioPin4 = LED(10)
# gpioPin5 = LED(9)
# gpioPin6 = LED(11)
# # [[LED(17), LED(27), LED(22), LED(10), LED(9), LED(11)]]
# display = drivers.Lcd()
# #
config = {
    "apiKey": " AIzaSyDqz-OaYAOB8L-UN8UlILvAhVZNdhGc_Q0 ",
    "authDomain": "risetpkm-f1dfd",
    "databaseURL": "https://risetpkm-f1dfd-default-rtdb.firebaseio.com/",
    "storageBucket": "project-id.appspot.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
# Load the model
model = load_model('keras_model.h5')


def classify_image(img_path):
    data = np.ndarray(shape=(1, 224, 224, 3),
                      dtype=np.float32)  # membuat data dengan dimensi N (jumlah), H (tinggi), W(width), C(channel) --> RGB
    image = img_path
    image_array = np.asarray(image)  # turn the image into a numpy array
    normalized_image_array = (image_array.astype(
        np.float32) / 127.0) - 1  # Normalize the image --> 0 - 225 --> -1 sampai 1
    data[0] = normalized_image_array  # Load the image into the array
    img_class = {0: "Makan", 1: "Minum", 2: "No", 3: "Pindah", 4: "Toilet"}
    prediction = model.predict(data)  # run the inference
    return (img_class[np.argmax(prediction)])


import cv2

cap = cv2.VideoCapture(0)  # memperbolehkan cv2 mengambil gambar
print(cap.get(3), cap.get(4))
w = cap.get(3)
h = cap.get(4)

a_w = 224 / w
a_h = 224 / h

# def draw(text):
# 	'''
# 	function untuk menambahkan hasil klasifikasi ke realtime image footage
# 	'''
# 	font = cv2.FONT_ITALIC
# 	if text[1] == "Makan":
# 		text = str(text)
# 		cv2.putText(frame, text, (20, 40), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
# 	if text[0] == "Minum":
# 		text = str(text)
# 		cv2.putText(frame, text, (20, 40), font, 1, (0, 0, 238), 1, cv2.LINE_AA)
# 	if text[2] == "Pindah":
# 			text = str(text)
# 			cv2.putText(frame, text, (20, 40), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
# 	if text[4] == "Toilet":
# 			text = str(text)
# 			cv2.putText(frame, text, (20, 40), font, 1, (0, 0, 238), 1, cv2.LINE_AA)
# 	else:
# 		text = str(text)
# 		cv2.putText(frame, text, (20, 40), font, 1, (0, 0, 238), 1, cv2.LINE_AA)


while True:
    ret, frame = cap.read()  # return gambar yang diambil oleh opencv
    if ret == True:
        frame_to_clf = cv2.resize(frame, None, fx=a_w, fy=a_h)
        prediction = classify_image(frame_to_clf)
        print(prediction)
        if prediction == 'Minum':
            hasil = 'Minum'
            data = {
                "Minum": hasil
            }

            db.child("smartasssistive").set(data)
            db.child("smartasssistive").push(data)
        # gpioPin1.on()
        # sleep(0.1)
        # gpioPin1.off()
        # gpioPin6.on()
        # sleep(0.1)
        # gpioPin1.off()
        # display.lcd_display_string("PKM KOMPOR TEAM", 1)
        # display.lcd_display_string("Pasien Membutuhkan Minum", 2)
        # display.lcd_clear()

        if prediction == 'Makan':
            hasil = 'Makan'
            data = {
                "Makan": hasil
            }
            db.child("smartasssistive").set(data)
            db.child("smartasssistive").push(data)
        # gpioPin2.on()
        # sleep(0.1)
        # gpioPin2.off()
        # gpioPin6.on()
        # sleep(0.1)
        # gpioPin6.off()
        # display.lcd_display_string("PKM KOMPOR TEAM", 1)
        # display.lcd_display_string("Pasien Membutuhkan Makan", 2)
        # display.lcd_clear()

        if prediction == 'Pindah':
            hasil = 'Pindah'
            data = {
                "Pindah": hasil
            }
            db.child("smartasssistive").set(data)
            db.child("smartasssistive").push(data)
        # gpioPin3.on()
        # sleep(0.1)
        # gpioPin3.off()
        # gpioPin6.on()
        # sleep(0.1)
        # gpioPin6.off()
        # display.lcd_display_string("PKM KOMPOR TEAM", 1)
        # display.lcd_display_string("Pasien Ingin Berpindah", 2)
        # display.lcd_clear()

        if prediction == 'Toilet':
            hasil = 'Toilet'
            data = {
                "Toilet": hasil
            }
            db.child("smartasssistive").set(data)
            db.child("smartasssistive").push(data)
        # gpioPin4.on()
        # sleep(0.1)
        # gpioPin4.off()
        # gpioPin6.on()
        # sleep(0.1)
        # gpioPin6.off()
        # display.lcd_display_string("PKM KOMPOR TEAM", 1)
        # display.lcd_display_string("Pasien Ingin Ke Toilet", 2)
        # display.lcd_clear()
        else:
            print("oke")
        # gpioPin1.off()
        # gpioPin2.off()
        # gpioPin3.off()
        # gpioPin4.off()
        # gpioPin5.off()
        # gpioPin6.off()
    # draw(prediction)
    cv2.imshow("prediction", frame)  # menampilkan gambar
    if cv2.waitKey(1) & 0xFF == ord('q'):  # close window tekan "q"
        break

cap.release()
cv2.destroyAllWindows()