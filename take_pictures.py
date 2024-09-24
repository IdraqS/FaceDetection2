import cv2
import os
import time

webcam = cv2.VideoCapture(0)
storage_folder = r'C:\Users\Idraq\Desktop\Project\Python Files\dataset_2'

frame_count = 0 
recording = False   

if not webcam.isOpened():
    print("Camera not open!!")
    exit()

#////////////////////////////////////////////////////////////////////////////////////////////////

while True:
    ret,frame = webcam.read()
    if not ret:
        print("Frames not being received, exiting")
        break
    
    cv2.imshow("Webcam feed", frame)

    if cv2.waitKey(1) == ord('p'):
        recording = True
        print('Recording started, frames gathering...')
        
        while recording:
            ret,frame = webcam.read()
            if not ret:
                print('Frames not being received, exiting')
                break

            file = os.path.join(storage_folder, f'photo_{int(cv2.getTickCount())}.jpg')
            cv2.imwrite(file,frame)
            print(f'Frame{frame_count} saved as {file}')
            frame_count += 1
            cv2.imshow("Webcam feed", frame)

    elif cv2.waitKey(1) == ord('q'):
        recording = False
        break
    

webcam.release()
cv2.destroyAllWindows()