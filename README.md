# Emotion-Recognition
Emotion Recognition - Vision Multi Classification 

### Reference
- https://arxiv.org/pdf/2109.07270.pdf
- https://github.com/yaoing/DAN


### Model Description 
<table>
    <thead>
        <tr>
            <td>DAN(Distract your Attention Network) Architecture</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>



### Requirements
- python V  # python version : 3.8.13
- tqdm
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
- dask
- partd
- pandas
- numpy
- scipy
- scikit-learn

### Download weight file
You must navigate to the "Data" folder and locate the "resnet18_msceleb.pth" file.
- Downloads link : https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view
- 해당 다운로드 링크는 원본 github에서 참고하여 기재함

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
This file is a trained file that learned the k-deep fashion dataset.
Ensure that the weight file is located at "./Data/".
- https://drive.google.com/file/d/11bt3BocyaukuP0GNVkAeM5Aue81A1kdX/view?usp=share_link




The testing cmd is: 
```

python3 test.py 

```

### Test Result
- Our Performance

|Dataset|Number|Accuracy|
|---|---|---|
|CK+48|93|*98.9%*|
|RAF|1,729|**91.6%**|
|Affectnet|1,172|*92.1%*|

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
            <td>Train / vaidation Loss And Accuracy Graph</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>
