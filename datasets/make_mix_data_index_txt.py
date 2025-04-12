import os
import random

# 随机种子
random.seed(0)

image_file_path = r"./mix_smoke/images"
save_path = r"./mix_smoke/images_index/"

train_val_percent = 1
train_percent = 0.9

image_list = os.listdir(image_file_path)
num = len(image_list)
index_list = range(num)
train_val_num = int(num * train_val_percent)
train_num = int(train_val_num * train_percent)
train_val_index_list = random.sample(index_list, train_val_num)
train_index_list = random.sample(train_val_index_list, train_num)

print("train and val size: ", train_val_num)
print("train size: ", train_num)

train_val_stream = open(os.path.join(save_path, "train_val.txt"), "w")
train_stream = open(os.path.join(save_path, "train.txt"), "w")
val_stream = open(os.path.join(save_path, "val.txt"), "w")


for i in index_list:
    name = image_list[i][:-4] + '\n'
    if i in train_val_index_list:
        train_val_stream.write(name)
        if i in train_index_list:
            train_stream.write(name)
        else:
            val_stream.write(name)


train_val_stream.close()
train_stream.close()
val_stream.close()
