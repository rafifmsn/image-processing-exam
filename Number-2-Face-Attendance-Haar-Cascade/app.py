import cv2
import numpy as np
import streamlit as st
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Face Attendance", layout="centered")
st.title("🎓 Simple Face Attendance System")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if FACE_CASCADE.empty():
    st.error("Haar Cascade file not loaded properly.")
    st.stop()

THRESHOLD = 60  # lower = stricter

# -----------------------------
# UTILITIES
# -----------------------------
def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))
    return face

def get_lbph():
    return cv2.face.LBPHFaceRecognizer_create()

# -----------------------------
# UI INPUTS
# -----------------------------
st.subheader("1️⃣ Upload Registered Face")
ref_file = st.file_uploader("Reference Image", type=["jpg", "png", "jpeg"])

st.subheader("2️⃣ Upload Attendance Media")
test_file = st.file_uploader(
    "Test Image or Video",
    type=["jpg", "png", "jpeg", "mp4", "avi"]
)

# -----------------------------
# PROCESS
# -----------------------------
if ref_file and test_file:

    # --- Load reference image
    ref_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
    ref_img = cv2.imdecode(ref_bytes, 1)

    ref_face = extract_face(ref_img)

    if ref_face is None:
        st.error("❌ No face detected in reference image")
        st.stop()

    recognizer = get_lbph()
    recognizer.train([ref_face], np.array([0]))

    st.image(ref_img, channels="BGR", caption="Registered Face")

    # --- IMAGE CHECK
    if test_file.name.lower().endswith(("jpg", "png", "jpeg")):
        test_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
        test_img = cv2.imdecode(test_bytes, 1)

        face = extract_face(test_img)

        if face is None:
            st.error("❌ No face detected in test image")
            st.stop()

        label, confidence = recognizer.predict(face)

        st.image(test_img, channels="BGR", caption="Attendance Image")

        if confidence < THRESHOLD:
            st.success(f"✅ PRESENT (Confidence Distance: {confidence:.2f})")
        else:
            st.error(f"❌ ABSENT (Distance: {confidence:.2f})")

    # --- VIDEO CHECK
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(test_file.read())

        cap = cv2.VideoCapture(tfile.name)
        found = False
        best_conf = 999

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            face = extract_face(frame)
            if face is not None:
                _, conf = recognizer.predict(face)
                best_conf = min(best_conf, conf)

                if conf < THRESHOLD:
                    found = True
                    cv2.putText(
                        frame, "MATCH FOUND",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )
                    stframe.image(frame, channels="BGR")
                    break

            stframe.image(frame, channels="BGR")

        cap.release()

        if found:
            st.success(f"✅ PRESENT (Best Distance: {best_conf:.2f})")
        else:
            st.error("❌ ABSENT (No matching face in video)")