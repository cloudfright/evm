import cv2
import numpy as np

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    face_detection_model = "face_detection_yunet_2023mar.onnx"
    score_threshold = 0.9
    nms_threshold = 0.3
    top_k = 5000
    
    detector = cv2.FaceDetectorYN.create(
        face_detection_model,
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )
    
    # read the image from the camera
    while True:
        ret, frame = cap.read()
        resized_frame = crop_square(frame, 320)

        faces = detector.detect(resized_frame)
    
        for face in (faces[1] if faces[1] is not None else []):

            box = face[0:4].astype(np.int32)
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(resized_frame, box, color, thickness, cv2.LINE_AA)
            conf = face[-1]
          
            """
                0-1: x, y of bbox top left corner
                2-3: width, height of bbox
                4-5: x, y of right eye 
                6-7: x, y of left eye 
                8-9: x, y of nose tip 
                10-11: x, y of right corner of mouth 
                12-13: x, y of left corner of mouth 
                14: face score
            """

        cv2.imshow('resized', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # close the camera
            cap.release()
            break
    

"""
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332
"""