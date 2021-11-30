import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get("PYTHONHTTPSVERIFY","") and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C','D', 'E','F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
nclasses = len(classes) 
#Splitting the data and scaling it 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500) 
#scaling the features 
X_train_scaled = X_train/255.0 
X_test_scaled = X_test/255.0 
#Fitting the training data into the model 
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train) 
#Calculating the accuracy of the model 
y_pred = clf.predict(X_test_scaled) 
accuracy = accuracy_score(y_test, y_pred) 
print("The accuracy is :- ",accuracy)
cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        im_pil=Image.fromarray(roi)
        img_bw=im_pil.convert("L")
        img_bw_resize=img_bw.resize((28,28),Image.ANTIALIAS)
        img_bw_resize_inverted=PIL.ImageOps.invert(img_bw_resize)
        pixel_filter=20
        min_pixel=np.percentile(img_bw_resize_inverted,pixel_filter)
        img_bw_resize_scaled=np.clip(min_pixel,0,255)
        max_pixel=np.max(img_bw_resize_inverted)
        img_bw_resize_inverted=np.asarray(img_bw_resize_scaled)/max_pixel
        test_sample=np.array(img_bw_resize_inverted).reshape(1,784)
        test_pred=clf.predict(test_sample)
        print(test_pred)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except:
        pass
cap.release()
cv2.destroyAllWindows()