(1) 指定embs = [
    "plane", "bicycle", "bird", "boat", "bottle", "buses", "car",
    "cat", "chair", "cow", "table", "dog", "horse", "motorbike",
    "people", "plant", "sheep", "sofa", "train", "monitor"
]

(2) 运行如下：
python  generate.py

(3) 会初步生成如下output_generated文件夹：
SD/output_generated
├── bicycle
├── bird
├── boat
├── bottle
├── buses
├── car
├── cat
├── chair
├── cow
├── dog
├── horse
├── monitor
├── motorbike
├── people
├── plane
├── plant
├── sheep
├── sofa
├── table
└── train


(4) 为了使得适用于我们的项目，使用reorganize.py文件进行文件重命名与重排列
在reorganize.py文件中修改成自己的文件地址
IMAGE_ROOT = "/home/songpf/DitBase/output_generated_50"
DATASET_ROOT = "/home/songpf/DitBase/diff_dataset"
其中IMAGE_ROOT代表上一步generate出来的output_generated文件夹地址
其中DATASET_ROOT代表重新命名后文件夹的期望地址

(5) 运行python reorganize.py