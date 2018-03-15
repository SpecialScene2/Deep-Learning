## Deep Learning Repository
* Python Tensorflow라이브러리를 사용하여 실습한 내용을 정리하였습니다.

### 실습내용
* __Practice1__
  * 총 49,200개의 신용카드 거래내역 중 492 개의 “비정상적인 거래"를 잘 찾아   낼 수 있는 모델 만들기.
  * 주어진 데이터의 Column
    * Time : 상대적 시간(가장 먼저 일어난 Time은 0 임)
    * 임의속성(V1-V28) : 개인정보 때문에 밝힐 수 없는 개별 거래 속성.<br>
       V1이 가장 높은 분산, V28이 가장 낮은 분산값 갖음.
    * Class : 정상 거래 이면 0, 비정상 거래이면 1 값을 지님
* __Practice2__
  * CNN모델을 활용한 이미지 분류
  * 10개 Class(airplane, automobile, bird, cat, deer, dog, frog,
horse, ship and truck)를 갖는 data_batch의 이미지들을 학습시켜 test_batch에 있는 이미지의 Class 분류하기
* __Practice3__
  * 시계열(time-series)데이터 예측을 위한 RNN모델 만들기
  * 데이터 : 호주 멜버른에서 측정된 10 년 동안의 일별 기온.
    * raw.csv : 1981년 부터 1989년 까지의 일별 기온(시계열 데이터).
    * train.csv : 1981년 부터 1989년 까지의 매 6일동안의 기온(6개 feature)<br> AND 7일 째 기온.
    * test.csv : 1990년 기온 월~금까지의 6일간 기온이 주어져있고<br>
     일요일(7일째)는 공백
  * 목표 : 6 일 동안의기온이 주어졌을 때 7일 째의 최저 기온을 예측하는 것
* __Practice4__
  * 데이터로부터 noise를 제거한 데이터를 추출하는 Denoising Autoencoder 모델을 만들기.
  * 랜덤한 위치에 구멍이 난 그림을 구멍을 채운 그림으로 추출하는 것.
  * 데이터 : MNIST데이터
