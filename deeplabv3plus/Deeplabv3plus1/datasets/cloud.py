import json
import os
from collections import namedtuple
import cv2

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Cloud(data.Dataset):
    """Cloud
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on SIA
    CloudClass = namedtuple('CloudClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [ #데이터 name #has_instances=True로 둬도 상민님은 별 문제 없었다고 하는데 체킹해보기
        CloudClass('unlabeled',             0, 254, 'void', 0, False, False, (0, 0, 0)), #배경은 검정색
        CloudClass('deep cloud',            1, 1, 'cloud', 1, True, False, (128, 0, 0)), #빨간색 : 짙은 구름
        CloudClass('cloud shadow',          2, 2, 'shadow', 2, True, False, (128, 128, 0)), #노란색 : 그림자
        CloudClass('light cloud',           3, 3, 'cloud', 1, True, False, (0, 128, 0)), #초록색 : 옅은 구름
        ]
#tain_id 로 변환해서 학습 하는 이유
#기존 이미지 데이터로 할때 보다 불필요한 이미지는 언라벨로 처리해서 데이터를 단순화/가볍게 만들 수 있다 
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    #target_type : 폴더 내 파일들 중에 target_type에 해당되는 애들만 가져오겠다
    def __init__(self, root, split='train', mode='fine', target_type='cloud_sia', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'label'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'data', split) #data 폴더 내> train 폴더 경로

        self.targets_dir = os.path.join(self.root, self.mode, split) #label 폴더 내> train 폴더 경로 
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        #우리 test 폴더 없음
        #학습을 시각화 하기때문에, test를 잘 안쓴다고함 -> 왜 인지 상민님께 여쭤보고
        #그럼 우린 이런 이유로 시각화를 해보려고 한다고 하면 좋을 것 같음
        #test 안쓸거고, 폴더 없으면 test는 수정하는게 맞음
        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++
        #이렇게 for문으로 파일을 하나씩 가져와서, 학습 진행하려고 함
        for cloud in os.listdir(self.images_dir): # 확실하지 않은 부분이라서, 상민님께 여쭤봐보기
            img_dir = os.path.join(self.images_dir, cloud) #data>train + list로 파일 하나씩 가져온 경로
            #print(img_dir)
            target_dir = os.path.join(self.targets_dir, cloud) #label>train+ data + list로 파일 하나씩 가져온 경로 
            #print(target_dir)
            #QQQ for 구문에서 data 파일들을 리스트로 가져오는데, target (라벨)경로랑 target 경로가 합쳐지지? 
            #여기서 가져오는 for 문의 i = 도시명 (폴더)


            #이미 위에서 이미지 파일 하나당 경로를 따로따로 만든것 같아서 아래 코드가 불필요 할 것 같은데, 아닌거 같기도
            #디렉토리 하위 폴더로 한번 더 해서 이중 FOR 문 돌릴 수 있는 구조로 만듦 ----> 해결 !
            for file_name in os.listdir(img_dir): #data>train 이미지 폴더에서 파일 하나씩 가져오기
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}{}'.format(file_name.split('.')[0], #라벨과 이미지 파일명 같으면 . , 다르면 붙여줌
                                             self._get_target_suffix(self.mode, self.target_type)) #png 파일만 가져오겠다
                                             #file not found 에러뜨면 못찾아오는거니까 _ 유무 확인해보기 :'{}_{}'.format
                #print(target_dir)
                #print(target_name)
                self.targets.append(os.path.join(target_dir, target_name)) #targets 빈 리스트에 저장한거 나중에 밑에 getitem 함수에서 target으로 하나씩 불러와줌
            #print(self.targets)
            #학습코드에서는 채널이 3개로 되어있는데, 데이터를 제대로 못불러와서 그런건지/ 아님 데이터 변환이 잘 안된건지
    
    #상민님 이부분 이해 안되심 -> 그냥 라벨 이미지를 불러오면 되어서 make_encode라는 함수를 따로 만드셨다고하심
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    #라벨 이미지를 불러와서, train_id로 바꿔주는 함수 코드
    def make_encode_target(cls, target):
        #target = np.array(target, np.uint8)
        #target = target*(1.0/255.0) #int-> uint 
        #target = np.uint8(target) #음수를 양수로 바꿔줌 (음수는 다 0으로 바꿔버리고, 버림 - 예상)
        #print(target)
        
        #target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) #grayscale 3채널을 1채널로 변경
        #deep cloud
        target[target == 76 ] = 1 #76: 빨강 = 짙은 구름
        #cloud shadow
        target[target == 226 ] = 2 #226:노랑 = 그림자
        #light cloud
        target[target == 150 ] =  3 #150: 초록 = 옅은 구름
        
        return target
        
    #cloud
        #바이너리 픽셀값이 255면, train_id 는 2로 바꿔주자 : target[target == 255] = 2
        #  +++++++++++++++++++++++++++++++++++++++
        # 근데 우리 라벨 이미지는 rgb 인데, 바이너리 형태의 흑백 형태 이미지로 변경해주고 그 다음에 진행해야하나?
        # 그럼 그걸 진행하는 코드는 어디쯤 위치하면 좋을까? -> 그냥 따로 코드로 json -> png 했던것처럼 흑백으로 바꿔줄까? 
        # 상민님이 json -> png 했던 영상 보면서 한번 봐봐야할듯 !
        # 그리고 rgb 라벨말고 흑백하면, 학습 진행 시각화가 좋다고 하지만 -- 그럼 반대로 단점은 없는건가?
        
        # 근데 그럼 짙은 구름, 옅은 구름, 그림자가 각각 어떤 값의 픽셀인지 알 수 있는거지?
        # 학습 시킬때 바이너리 값을 설정해서 학습한다고 하심 -> 이 부분이 뭔지 상민님께 더 물어보기, 어떻게 하는지랑 ㅠ 
        # 그리고 하나만? 학습한다는 식으로 말씀해주셨는데, 하나씩 학습해서, 멀티로는 못하는 느낌인건가?
        #이건 상민님께 물어보고, 분업해버리기 ! > 나는 논문찾고, 읽고 해보기

        #gray scale로 혜지님이 이미지 데이터 변환 해주기로 하심 !
        #grayscale 로 변환해서 학습 하는 이유   
        # 1) 학습 시각화 할 때 효과적으로 하기위해
        # 2) 채널이 3개이 rgb는 코드짤때 고려할게 더 많아서, 불필요한 에러가 발생 할 수도 있을 것이라고 생각해서
        # 편의상 수정해서 학습 진행



    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 1 #class 개수와 다르게, 무작위로 1 넣으니까 되셔서 -> 확인 필요 : 상민님께 문의
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        
        if self.transform:
            image, target = self.transform(image, target)
            # 여기부터 아래는 encode 함수를 이해했으면 필요 없는 부분인데, 이해하지 못해서 작성해줌 (make_encode 함수도 마찬가지_)
            # PIL라이브러리에서 image 로드해주는 걸 import 해줘서
            #png 파일 읽어오는게, 자료형이 안맞을 수 있어서 -> np.array로 넘파이형식으로 target 변경
        target = np.array(target)
        #그 다음에 위에서 만들어준 함수를 사용해서 train_id로 변경
        target = self.make_encode_target(target)
        #그 다음 혹시 몰라서 형식 변경이 될까봐 np.array한번 더 해주고,
        #변환 과정에서 또 자료형이 안맞을까봐 데이터 타입 int 형식으로 바꿔줌
        target = np.array(target, dtype ='int' ) #target = np.array(target, dtype ='uint8'): 실험해봤는데 동일 에러
        #print(f"target shape : {target.shape}")
        return image, target
        
        
    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'cloud_sia':
            return '_label.png'.format(mode) # label.png
        #elif target_type == 'semantic':
        #    return '{}_labelIds.png'.format(mode) 