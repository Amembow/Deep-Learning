
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import datetime
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense



model = load_model('third_30.h5')

dt_now = datetime.datetime.now()


plog = 'data/result.txt'
f = open(plog,mode='a')
f.write(str(dt_now) + '\n')

def process_image(image):
    ret,bi_inv = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    #plt.imshow(bi_inv, 'gray')
    return bi_inv, image

def create_data(tdir_, type_, frm, to, t_data):
    colored_data=[]
    bi_inv_data=[]
    print('Working On '+t_data+' Data : '+type_ +'\n')
    rng = ["%05d" % x for x in range(frm, to+1)]
    for i in rng:
        dir_ =tdir_+type_+'/'+i+'.jpg'
        image = cv2.imread(dir_, 0)
        bi_inv, colored_img = process_image(image)
        colored_data.append(colored_img)
        bi_inv_data.append(bi_inv)   
        
    print('Images Processed from '+rng[0]+' to '+rng[len(rng)-1]+'\n')    
    return colored_data, bi_inv_data #6

def predict_image_util(final_pred_inv):
    img_test = (final_pred_inv[0].reshape((1, 227, 227, 1)))  
    raw_predicted_label = model.predict(img_test, batch_size=None, verbose=0, steps=None)[0][0]
    
    predicted_label=1;    
    if(raw_predicted_label<0.8):
        predicted_label=0
        
    predicted_label_str='Crack'    
    if(predicted_label==0):
        predicted_label_str='No Crack'

    f.write(',' + predicted_label_str + '\n')
        
    print('Raw Predicted Label(Numeric): '+str(raw_predicted_label))
    print('\nPredicted Label : '+predicted_label_str) #20

from_data_dir = 'original_data/'  #Directory

def predict_image2(type_, num):
    
    pred_data_colr_, pred_data_inv_ = create_data(from_data_dir, type_, num, num+1, 'Predictable')
    print(pred_data_colr_[0])
    plt.imshow(pred_data_colr_[0])
    pred_data_colr =[]
    pred_data_inv = []
    
    pred_data_inv.append(pred_data_inv_[0])
    pred_data_colr.append(pred_data_colr_[0])
    
    final_pred_colr = np.array(pred_data_colr).reshape(((len(pred_data_colr), 227, 227, 1)))  
    final_pred_inv = np.array(pred_data_inv).reshape(((len(pred_data_inv), 227, 227, 1)))
    predict_image_util(final_pred_inv) #21



numo = 1

while numo < 40001:
    print(numo)

    f.write(str(numo))

    predict_image2('positive', numo)
    numo += 1

dt_now = datetime.datetime.now()
f.write(str(dt_now) + '\n')

f.close()

#print('1')
#predict_image2('positive', 1) #22

#print('2')
#predict_image2('positive', 7645) #23

#print('3')
#predict_image2('negative', 1111) #24

#print('4')
#predict_image2('negative', 9991) #25


#model.save_weights('third_30_weights.h5')
#model.save('third_30.h5') #26

print('End')


