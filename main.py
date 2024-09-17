from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from pymongo import MongoClient
import gridfs
from PIL import Image, ExifTags
import face_recognition


app = FastAPI()

# Load your pre-trained CNN model
model = tf.keras.models.load_model("E:\Graduation Project\EmotionDetectScratch.h5")
labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# Face detection from image
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

CurrencyModel = tf.keras.models.load_model("E:\webcamtest\Currencydetector.h5")

Currencylabels = {0:'1.LE', 1:'10.LE', 2:'10.LE (new)', 3:'100.LE', 4:'20.LE', 5:'20.LE (new)', 6:'200.LE', 7:'5.LE', 8:'50.LE'}

YoloV8Model = YOLO("yolov8l.pt")

# Connect to MongoDB Atlas
client = MongoClient('mongodb+srv://Sha3ban:iDWtDyobzj4eQCrj@facerecognizecluster.u7o6neq.mongodb.net/')
db = client['face_recognition_db']
fs = gridfs.GridFS(db)
usersCollection = db['users']




def newimg(image):
    # Resize the image to match the expected input shape of the model
    image = cv2.resize(image, (48, 48))
    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    # Preprocess the image by normalizing pixel values
    image = image / 255.0
    return image

def preprocess(im):
    # Get the dimensions of the image
    height, width, _ = im.shape
    # Convert the image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (p, q, r, s) in faces:
        # Extract the face region
        face_image = gray[q:q + s, p:p + r]
        # Resize the face image to the required size
        face_image = cv2.resize(face_image, (48, 48))
        # Perform any additional preprocessing if needed
        img = newimg(face_image)
        # Return the preprocessed image
        return img

def get_dominant_color(img):
    detected_colors = []
    
    color_ranges = {
        'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
        'green': (np.array([40, 100, 100]), np.array([80, 255, 255])),
        'blue': (np.array([100, 100, 100]), np.array([140, 255, 255])),
        'white': (np.array([0, 0, 200]), np.array([180, 20, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
        'brown': (np.array([5,60,70]), np.array([15,200,200])),
        'orange': (np.array([10, 102, 153]), np.array([20, 255, 255])),
        'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
        'purple': (np.array([130, 100, 100]), np.array([160, 255, 255])),
        'pink': (np.array([150, 100, 100]), np.array([170, 255, 255])),
        'cyan': (np.array([80, 100, 100]), np.array([100, 255, 255])),
        'magenta': (np.array([160, 100, 100]), np.array([180, 255, 255])),
        'teal': (np.array([70, 100, 100]), np.array([90, 255, 255])),
        'lavender': (np.array([120, 100, 100]), np.array([140, 255, 255])),
        'olive': (np.array([30, 100, 100]), np.array([50, 255, 255])),
        'sky_blue': (np.array([100, 60, 60]), np.array([120, 255, 255])),
        'gold': (np.array([20, 100, 100]), np.array([30, 255, 255])),
        
    }
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    color_space = []

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        color_area = sum(cv2.contourArea(contour) for contour in contours)
        if color_name.lower() not in ("black", "white"):
            color_space.append(color_area)
            detected_colors.append(color_name)
        
    # Find the most dominant color
    max_index = find_max_index(color_space)
    
    dominant_color = detected_colors[max_index]
 
    return dominant_color

def find_max_index(arr):
    max_index = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[max_index]:
            max_index = i
    return max_index

def CurrencyPreprocess_image(image):
    # Resize the image to 256x256
    image = cv2.resize(image, (224, 224))
    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=0)
    # Preprocess the image by normalizing pixel values
    image = image / 255.0
    return image

def objectof(image):
    results = YoloV8Model.predict(image)
    predictions = results[0]
    name_of_class = predictions.verbose()
    return name_of_class


# Function to detect and extract faces from an image
def read_and_detect_faces(image, scale=0.5):
    if image is None:
        raise FileNotFoundError("Image file not found")

    # Check and correct the orientation of the image
    image = correct_image_orientation(image)
    # Resize the image for faster processing
    small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_image)
    if face_locations:
        face_images = []
        for (top, right, bottom, left) in face_locations:
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            face_image = image[top:bottom, left:right]
            face_images.append(face_image)
        return face_images
    else:
        return None

def correct_image_orientation(image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(pil_image._getexif().items())
        if exif[orientation] == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif exif[orientation] == 6:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif exif[orientation] == 8:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    return image

def insertUser(android_id):
    user = usersCollection.find_one({"Andriod_Id": android_id})
    if not user:
        user = {
            "Andriod_Id": android_id,
            "photos": []
        }
        usersCollection.insert_one(user)
        return f"New user inserted with ID: {android_id}"
    else:
        return "User already exists"

def addPersonToUserRelvants(android_id, person_name, image, filename):
    face_images = read_and_detect_faces(image)
    if not face_images:
        return "No faces detected."

    face_image = face_images[0]
    _, encoded_image = cv2.imencode('.jpg', face_image)
    image_bytes = encoded_image.tobytes()

    photo_id = fs.put(image_bytes, filename=filename)
    new_photo = {
        "person_name": person_name,
        "photo_id": photo_id
    }
    usersCollection.update_one(
        {"Andriod_Id": android_id},
        {"$push": {"photos": new_photo}}
    )
    return f"Added new photo for existing user with ID: {android_id}"

def get_relevant_persons(andriod_iD):
    user =usersCollection.find_one({'Andriod_Id': andriod_iD})
    if user:
        return user["photos"]
    return None

def recognize(image, android_id):
    photos = get_relevant_persons(android_id)
    known_face_encodings = []
    known_face_names = []

    if photos:
        for face in photos:
            file = fs.get(face['photo_id'])
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(face['person_name'])

        if not known_face_encodings:
            return "No known faces found in the user's photos."

        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            return "No faces found in the provided image."

        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                results.append(f"This is: {name}")
            else:
                results.append("This person isn't in our relative faces.")
        return results
    else:
        return "We have no relative faces known."


@app.post("/insert_user/")
async def insert_user_endpoint(android_id: str):
    try:
        result = insertUser(android_id)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize/")
async def recognize_endpoint(android_id: str, file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Call the recognize function
        result = recognize(image, android_id)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_person/")
async def add_person_endpoint(android_id: str, person_name: str, file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Call the addPersonToUserRelvants function
        result = addPersonToUserRelvants(android_id, person_name, image, file.filename)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/object_detection")
async def object_detection(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Make predictions using the model
        
        predictions = objectof(image)

        # Get the predicted label
        
        print(f" The object/s is/are : {predictions}")
        return JSONResponse(content={"The object/s is/are  ": predictions}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_currency")
async def predict_currency(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        preprocessed_image = CurrencyPreprocess_image(image_rgb)

        # Make predictions using the model
        predictions = CurrencyModel.predict(preprocessed_image)

        # Get the predicted label
        predicted_label = Currencylabels[np.argmax(predictions)]
        print(f" The BankNote is: {predicted_label}")
        return JSONResponse(content={"BankNote is ": predicted_label}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recogcolor")
async def recogcolor(file: UploadFile):
    try:
        # Receive image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        print("Received image upload")

        # Preprocess the uploaded image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Make color recognition
        dominant_color = get_dominant_color(image)
        print("Model predictions completed")

        # Get the Color name
        print(f"The dominant color is: {dominant_color}")
        return JSONResponse(content={"color": dominant_color})

    except Exception as e:
        # Handle errors, log them, and return an appropriate response
        print(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/predict")
async def predict(file: UploadFile):
    try:
        # Receive image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        print("Received image upload")

        # Preprocess the uploaded image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_image = preprocess(image)
        print("Image preprocessing completed")

        # Make predictions using the loaded model
        predictions = model.predict(processed_image)
        pred_label = labels[predictions.argmax()]
        print("Model predictions completed")

        # Get the predicted label
        print(f"Predicted emotion: {pred_label}")
        return JSONResponse(content={"emotion": pred_label})
    except Exception as e:
        # Handle errors, log them, and return an appropriate response
        print(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
