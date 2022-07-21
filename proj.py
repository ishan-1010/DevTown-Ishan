import cv2
import numpy as np

video = cv2.VideoCapture(
    "mixkit-young-woman-giving-a-presentation-on-a-green-screen-28292(1)(1).mp4")
image = cv2.imread("pexels-aleksandar-pasaric-325185.jpg")

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (1300, 650))
    image = cv2.resize(image, (1300, 650))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_g = np.array([32, 94, 132])
    u_g = np.array([62, 255, 255])

    mask = cv2.inRange(hsv, l_g, u_g)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f = frame-res
    green_screen = np.where(f == 0, image, f)
   # cv2.imshow("Frame", frame)
   # cv2.imshow("Mask", mask)
   # cv2.imshow("RES", res)
   # cv2.imshow("f", f)
    cv2.imshow("final", green_screen)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
