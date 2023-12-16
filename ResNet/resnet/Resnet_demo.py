import cv2 as cv
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
type_labels = ["Negative", "Positive",]


def test_vehicle_codes():
    if torch.cuda.is_available():
        resnet_model = torch.load("detect_model.pt")
    else:
        resnet_model = torch.load("detect_model.pt", map_location='cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((128, 128)),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ]
    )

    test_dir = "../concerete_crack_images/test/"
    test_files = os.listdir(test_dir)
    result=[]

    for f in test_files:
        if f.split(".")[-1] != "png":
            continue
        image_dir = test_dir + f
        img_ID=f
        image = cv.imread(image_dir)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img = transform(image)
        x_input = img.view(1, 3, 128, 128)
        if torch.cuda.is_available():
            probs_type = resnet_model(x_input.cuda())
        else:
            probs_type = resnet_model(x_input)
        type_index = probs_type.cpu().tolist()
        t1 = np.argmax(type_index)
        # print("cuda: ", torch.cuda.is_available())
        # print(img_ID, t1)
        result.append([img_ID,type_labels[t1]])

    column = ['Image', 'Prediction']  # 列表头名称
    test = pd.DataFrame(columns=column, data=result)  # 将数据放进表格
    test.to_csv('output.csv')  # 数据存入csv,存储位置及文件名称

if __name__ == "__main__":
    test_vehicle_codes()