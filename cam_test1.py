import cv2
import time

capture = cv2.VideoCapture(0)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == True:
        # フレームを表示

        
        cv2.imshow('Webcam Live', frame)
        
        # 'q'キーが押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    else:
        break

# キャプチャをリリースし、ウィンドウを閉じる
capture.release()
cv2.destroyAllWindows()
