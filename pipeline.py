from super_gradients.training import models
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as pl


class predictPipeline():
    def __init__(self) -> None:
        
        # Load model
        self.model = models.get('yolo_nas_m',
                                num_classes=1,
                                checkpoint_path='yolo_nas_m_model.pth')
    
    def detect(self, img_path):
        image = Image.open(img_path).convert('RGB')
        img_array = np.array(image)

        preds = self.model.predict(img_array, conf=0.5)[0].prediction
        bboxes_coordinates = []
        for idx, bbox in enumerate(preds.bboxes_xyxy):
            bboxes_coordinates.append([int(num) for num in bbox] + [round(preds.confidence[idx]*100, 2)])
        return bboxes_coordinates
    
    
    def drawDetections2Image(self, img_path, detections):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        for bbox in detections:
            x1, y1, x2, y2, score = bbox
            cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
            cv.putText(img, text=f'{score}%', org=(x1, y1-2), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), lineType=cv.LINE_AA)
        img_detections = np.array(img)
        return img_detections



