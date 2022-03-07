# cloud_sia
### Welcome to GroomyRoom Cloud_SIA GITHUB!
![music-cloud-line-icon-linear-260nw-1658323474](https://user-images.githubusercontent.com/79895378/156993388-a1e8ff82-a524-4d7c-9837-ae9b9648aac5.jpeg)

# Introduction:

| **TEAM MEMBERS**   | **FACE**          | **DESCRIPTION** |**Characteristic**|
| ------------- |:-------------:| -----:|-----:|
| **안가영** **(팀장)**      |  | **👑우리 팀 지배자** |**모델 학습 학살자**, **리더 버프**,|
| **김태훈** **(팀원)**      | ![Tae Face Emoji](https://user-images.githubusercontent.com/79895378/157096500-7acce9e4-1c79-4185-9569-cd884baf3da8.png)     |   **🤡 우리 팀 광대** |**데이터 분석자**, **전처리 마스터**|
| **김혜지** **(팀원)**      |       |  **🗡️우리 팀 도적**  |**논문 학자**, **코드 수정 서포터**|



# Group Time Table:

| 구분           | 기간           |   활동   | 상세 내용| 
| ------------- |:-------------:| -------:|-------:|
| 1주차: 논문 분석  | 1/19(수요일) ~ 1/21(금요일) |*Deeplabv3plus,Unet, HRnet *기획안 작성하기| *베이스 라인 모델 논문 분석, *미니 해커톤 기획안 작성 및 제출 | 
| 2주차. 베이스 라인 모델 코드 확인 및 분석  | 1/24 (월요일)  ~  1/28 (금요일)  | *논문 리뷰 *베이스라인 모델 코드 분석, *기초 EDA |* 데이터로더, 모델 학습 부분의 코드 분석, *train, label 데이터 수, 크기, 파일 형태 들 분석  
|3주차. 코드 분석 및 전처리 | 2/03(목요일) ~ 2/11 (금요일) | *베이스라인 모델 코드 분석, *데이터 전처리      | *데이터로더, 모델 학습 부분의 코드 분석, * 데이터 전처리 |   
| 4주차. 코드 튜닝 | 2/10 (목요일) ~ 2/17 (목요일) |  *모델 튜닝 및 학습 진행, *성능 개선점 도출, 중간 보고 발표 | *구름 데이터셋에 맞게 코드 튜닝 (데이터로더 수정, HRNet으로 모델 변경), * 멀티 VS 단일 Class 성늘 비교, per epoch-augmentation |
| 5주차. 성능 개선| 2/18 (금요일) ~ 3/4 (목요일) | *심화 EDA, *성능 개선 및 싫험, *이미지 전처리|*Augmentation Each Epoch, *EDA PIXEL IMAGE 분석, *히스토그램 이미지 분석, *히스토그램 평활화, *감마 보정, *이미지 컬러 스페이스 변환 |   
|6주차. 최종 검토 | 3/4 (목요일) ~ 3/8 (화요일) | *발표 자료 준비, *발표 연습, *성능 개선 및 실험 |* 최종 발표 준비, *성능 개선 및 실험|   
|촞 개발기간 | 1/19 (수요일) ~ 3/8(화요일) | *모두가 열심히 했다    |*모두가 열심히 했다 |

# 📓 Assignment:

``` Satellite Cloud Image Semantic Segmentation ```

 ### **Level One ⭐:**  구름 검출 모델 구현:
 
 
#### 1️⃣DeeplabV3 + HRNet 모델 튜닝 하고 150 epoch 학습 진행하기. 


### **Level Two ⭐⭐:** EDA를 통해 데이터 분석, 검출 모델 성능 향상:
 

#### 1️⃣ Val 이미지 shape이 각기 다름 

#### ➡ zero-padding 제일 큰 이미지 사이즈에 맞추기

#### ➡️ 제일 작은 이미지에 맞춰서 이미지 crop 하기

#### ➡️ train data와 동일하게 random cropping (train epoch수에 맞게 적절한 학습 epoch수 고려 필)



### 2️⃣ 단일 class VS 멀티 class

#### ➡️ 잩은 구름, 옅은 구름, 그림자 3개의 class 동시검출과 단일 검출 시 성능 비교 실험

#### ➡️ 단일 검출이 더 좋은 성능을 보일것이라 예상되지만,
라벨과 mIoU 계산을 위해 단일 출력된 predict를 어떻게 합칠 것인가에 대한 접근 중요

# 💬 발표 자료:
| **발표:**   | **링크:** | 
| ------------- | ------------- |
| **중간 발표**  | [중간 발표 링크](https://github.com/grromyroom/Cloud_Sia_Overview/blob/main/%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD/%E1%84%8C%E1%85%AE%E1%86%BC%E1%84%80%E1%85%A1%E1%86%AB%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD%20(%E1%84%8B%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A1%E1%84%8B%E1%85%A7%E1%86%BC%2C%20%E1%84%80%E1%85%B5%E1%86%B7%E1%84%90%E1%85%A2%E1%84%92%E1%85%AE%E1%86%AB%2C%20%E1%84%80%E1%85%B5%E1%86%B7%E1%84%92%E1%85%A8%E1%84%8C%E1%85%B5).pdf) |
| **최종 발표**  | [최종 발표 링크](https://www.google.com) |

# 📋 논문 참고자료 (reference):
| **📋 논문**  | **📖 논문 링크** |**🎞️ 논문 영상** |
| ------------- | ------------- |------|
|**DeeplabV3+**| [Deeplab 논문 링크](https://arxiv.org/abs/1706.05587), [Deeplabv1,2,3,3+ 정리 블로그](https://kuklife.tistory.com/121) |[유튜브 영상 1](https://www.youtube.com/watch?v=i0lkmULXwe0&ab_channel=%EA%B3%A0%EB%A0%A4%EB%8C%80%ED%95%99%EA%B5%90%EC%82%B0%EC%97%85%EA%B2%BD%EC%98%81%EA%B3%B5%ED%95%99%EB%B6%80DSBA%EC%97%B0%EA%B5%AC%EC%8B%A4), [유튜브 영상 2](https://www.youtube.com/watch?v=TjHR9Z9iNLA) |
| **HRNET** | [HRNET 논문 링크](https://arxiv.org/abs/1908.07919), [HRNET 정리 블로그](https://paperswithcode.com/paper/190807919) |없음 |
| **U-NET** |[U-NET 논문 링크](https://arxiv.org/abs/1505.04597), [U-NET 정리 브로그](https://medium.com/@msmapark2/u-net-논문-리뷰-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a) |[유튜브 영상 1](https://www.youtube.com/watch?v=O_7mR4H9WLk&ab_channel=%EA%B3%A0%EB%A0%A4%EB%8C%80%ED%95%99%EA%B5%90%EC%82%B0%EC%97%85%EA%B2%BD%EC%98%81%EA%B3%B5%ED%95%99%EB%B6%80DSBA%EC%97%B0%EA%B5%AC%EC%8B%A4), [유튜브 영상 2](https://www.youtube.com/watch?v=oLvmLJkmXuc&ab_channel=AladdinPersson), [유튜브 영상 3](https://www.youtube.com/watch?v=evPZI9B2LvQ&ab_channel=%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%85%BC%EB%AC%B8%EC%9D%BD%EA%B8%B0%EB%AA%A8%EC%9E%84)|

# 📗 노션 페이지:
➡️ [노션 링크](https://www.notion.so/modulabs/GroomyRoom_-75ce8589a3a4499ea2913a1da83a45e6)
