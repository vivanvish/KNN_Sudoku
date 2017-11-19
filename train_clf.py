import cv2
import os
import numpy as np

def data_prep():
    rootdir = r'.\data\Fnt'
    train_data = open(r'data/train.data', 'w')
    labels_data = open(r'data/label.data', 'w')

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            labels_data.write(str(float(subdir.split('\\')[-1]))+'\n')
            img = cv2.imread(os.path.join(subdir,file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh = cv2.bitwise_not(gray)

            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rotated = np.array(cv2.resize(rotated,(28,28)).reshape(1,784).flatten().astype(np.float32))
            ansstr = ' '.join(str(i) for i in rotated)
            train_data.write(ansstr+'\n')

if __name__ == '__main__':
    data_prep()