import cv2

video_path = "videos/Jumping_over_barrier.mp4"
cap = cv2.VideoCapture(video_path)

# Mouse callback function
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: X={x}, Y={y}")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", get_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:   # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()