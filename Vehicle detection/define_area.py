import cv2

vid = cv2.VideoCapture('../Media/toll_gate.mp4')

success, frame = vid.read()

if not success:
    print("Failed to capture video frame.")
    exit()
points = []
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('Frame', frame)


cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", get_points)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Width: {width}")
print(f"Video Height: {height}")
cv2.waitKey(0)
cv2.destroyAllWindows()

print(points)
