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

### Download pretrained weight file
[DAN Model]
You must navigate to the "Data" folder and locate the "resnet18_msceleb.pth" file.
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

If you want to proceed with the new learning, adjust the parameters and set the directory and proceed with the command below.

The Training cmd is:
```

python3 train.py 

```

##### Trained weight file Download 
Download the trained weight file through the link below.
This file is a trained file that trained the Sample dataset(CK + RAF + AffectNet).
Ensure that the weight file is located at "./Data/".
- https://drive.google.com/file/d/11bt3BocyaukuP0GNVkAeM5Aue81A1kdX/view?usp=share_link


The testing cmd is: 
```

python3 test.py 

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
    <tbody>
        <tr>
            <td><img src=""/></td>
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
