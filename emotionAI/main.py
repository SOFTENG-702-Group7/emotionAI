import cv2 as cv
import torch
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO


# Load models and transforms
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_recognizer = torch.load("emotion-recognition-model.pt", map_location=device)
    emotion_recognizer.eval()
    face_detector = YOLO("yolov8n-face.pt")
    emotion_labels = ["neutral", "disgust", "fear", "sad", "happy", "surprise" ]
    return emotion_recognizer, face_detector, emotion_labels, device

def prepare_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

emotion_recognizer, face_detector, emotion_labels, device = load_models()
transform = prepare_transform()

# Process frames
def process_frame(frame, models, transform, device):
    face_detector, emotion_recognizer, emotion_labels = models
    results = face_detector(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face_tensor = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                emotion_pred = emotion_recognizer(face_tensor)
                emotion_idx = torch.argmax(emotion_pred).item()
                emotion = emotion_labels[emotion_idx]
                emotion_confidence = torch.nn.functional.softmax(emotion_pred, dim=1)[0][emotion_idx].item()
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{emotion}: {emotion_confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Start camera
def start_camera():
    cap = cv.VideoCapture(0)
    cv.namedWindow("Emotion Recognition", cv.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, (face_detector, emotion_recognizer, emotion_labels), transform, device)
        cv.imshow('Emotion Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    start_camera()