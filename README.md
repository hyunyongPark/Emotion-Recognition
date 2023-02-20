# Emotion-Recognition
Emotion Recognition - Vision Multi Classification 

### Reference
##### DAN(Distract your Attention Network)
- https://arxiv.org/pdf/2109.07270.pdf
- https://github.com/yaoing/DAN
##### EfficientNet
- https://arxiv.org/pdf/1905.11946.pdf


### Model Description 
<table>
    <thead>
        <tr>
            <td>DAN(Distract your Attention Network) Architecture</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/Emotion-Recognition/blob/main/img/architecture.png"/></td>
        </tr>
    </tbody>
</table>



### Requirements
- python V  # python version : 3.8.13
- tqdm
- opencv-python
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
- timm
- dask
- partd
- pandas
- numpy
- scipy
- scikit-learn
- requests

### Download pretrained weight file
[DAN Model]
사전학습된 resnet weight파일을 먼저 "./Data" 경로에 넣어주세요.
- Downloads link : https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view
- The download link was described in reference to the original github.

### cmd running

The install cmd is:
```
conda create -n your_prjname python=3.8
conda activate your_prjname
cd {Repo Directory}
pip install -r requirements.txt
```
- your_prjname : Name of the virtual environment to create

##### ★★★Trained weight file & Data Download 
- DAN모델을 통해 미리 학습된 weight(파일명 DAN_best.pth)를 아래의 링크에서 다운받아서 "./Data" 경로에 위치시키세요.
- 추가로 Sample dataset(CK + RAF + AffectNet)의 각 압축파일을 "./Data" 경로에 위치시키세요.
- https://drive.google.com/drive/folders/1Zd1RrTjPvlV3Q3pS9nfn6MRHCCWyEF0e?usp=share_link

새롭게 학습 시킬 시 아래의 커맨드를 실행하세요.

The Training cmd is:
```

python3 train.py 

```

미리학습된 웨이트나 학습시킨 후의 웨이트로 test performance를 얻기 위해서는 아래의 커맨드를 실행하세요.

The testing cmd is: 
```

python3 test.py 

```

이미지 URL을 통해 감정을 예측할 시 아래의 커맨드를 참고하세요.
```
python3 predict.py --params https://img.freepik.com/free-photo/portrait-of-smiling-young-man-looking-at-camera_23-2148193854.jpg

```


### Result
- Our Performance


|Model|Dataset|Train|Validation|Train Accuracy|Validation Accuracy|
|---|---|---|---|---|---|
|EfficientNetb3|CK+RAF+AffectNet|28,601|15,206|99.5%|93.7%|
|DAN|CK+RAF+AffectNet|28,601|15,206|||



|Model|Dataset|Number|Accuracy|
|---|---|---|---|
|EfficientNet|CK+48|93|*99%*|
|EfficientNet|RAF|1,729|*88.9%*|
|EfficientNet|Affectnet|1,172|*77.8%*|
|DAN|CK+48|93|*98.9%*|
|DAN|RAF|1,729|**91.6%**|
|DAN|Affectnet|1,172|**92.1%**|

<table>
    </thead>
        <tr>
            <td>T-SNE Clustering Plot</td>
        </tr>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/Emotion-Recognition/blob/main/img/tsne.png"/></td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <td>DAN Train&vaidation Plot</td>
            <td>EfficientNet Train&vaidation Plot</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/Emotion-Recognition/blob/main/img/Training%20%26%20Validation%20train%20Plot.png"/></td>
            <td><img src="https://github.com/hyunyongPark/Emotion-Recognition/blob/main/img/efficientnet_b3%20Training%20%26%20Validation%20Plot.png"/></td>
        </tr>
    </tbody>
</table>
