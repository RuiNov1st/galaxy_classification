import torch
import os
import sys
import h5py
import numpy as np
import pandas as pd
import zoobot
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
from zoobot.pytorch.predictions import predict_on_catalog

dataset_labels = ["Disturbed","Merging","Round Smooth","In-between Round Smooth","Cigar Shaped Smooth","Barred Spiral","Unbarred Tight Spiral","Unbarred Loose Spiral","Edge-on without Bulge","Edge-on with Bulge"]
# Disturbed - 扰乱型
# Merging - 合并型
# Round Smooth - 圆滑型
# In-between Round Smooth - 中间圆滑型
# Cigar Shaped Smooth - 雪茄形滑型
# Barred Spiral - 条纹螺旋型
# Unbarred Tight Spiral - 无条纹紧密螺旋型
# Unbarred Loose Spiral - 无条纹松散螺旋型
# Edge-on without Bulge - 无突起的边缘型
# Edge-on with Bulge - 有突起的边缘型


def galaxy_classify_zoobot(image_path='./dataset/Galaxy10_DECals_predict/Barred Spiral_1.png',model_path='./model/zoobot/finetune_model/FinetuneableZoobotClassifier.ckpt'):
    """
    classification process
    - params:
    image_path(str): galaxy image for classification. classify only one image at a time.
    model_path(str): pretrained machine learning model for classification.
    - return:
    predict_res(str): most likely type for input image
    predict_probability(float): confidence associated with predict_res
    result_dict(dict): confidences of all types .
    """

    # check path:
    if not os.path.exists(image_path):
        print(f"Image {image_path} not exists. Please check!")
        return None
        
    if not os.path.exists(model_path):
        print(f"Model weight {model_path} not exists. Please check!")
        return None

    # make data catalog: just one image
    id_list = ['test_a']
    file_list = [image_path]
    test_catalog = pd.DataFrame({'id_str':id_list,'file_loc':file_list})
    
    # load model:
    model = FinetuneableZoobotClassifier.load_from_checkpoint(model_path,map_location='cpu')
    
    # predict:
    prediction = predict_on_catalog.predict(
      test_catalog,
      model,
      n_samples=1,
      label_cols=dataset_labels,  # name the output columns
      save_loc='./outputs/finetuned_predictions.csv',
      trainer_kwargs={'accelerator': 'cpu'},
      datamodule_kwargs={'num_workers': 2, 'batch_size': 32, 'greyscale': False},
    )
    predict_idx = np.argmax(prediction[0,:,0])
    predict_res = dataset_labels[predict_idx]
    predict_probability = prediction[0,predict_idx,0]
    result_dict = dict(zip(dataset_labels,prediction[0,:,0]))

    return predict_res,predict_probability,result_dict

if __name__ == '__main__':
    image_path = './dataset/Galaxy10_DECals_predict/Barred Spiral_1.png'
    model_path = './model/zoobot/finetune_model/FinetuneableZoobotClassifier.ckpt'
    predict_res,predict_probability,result_dict = galaxy_classify_zoobot(image_path,model_path)
    print("==============================")
    print("galaxy type:",predict_res)
    print("classfication confidence:",str(predict_probability))
    print("all types results:")
    print(result_dict)