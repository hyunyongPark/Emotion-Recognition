class CFG:
    ####### Path #######
    pretrained_path = './Data/resnet18_msceleb.pth'
    data_path = './Data'
    save_path = './Data'
    
    ####### Basic params #######
    model_name = "DAN" # DAN or efficientnet_b3
    img_size = 224 #224 #512
    Epoch = 150
    Batch_size = 64 #128 #256 #48
    pretrained = True#True #True
    n_classes = 6
    Num_worker = 12
    ####### LearningRate #######
    lr_start = 1e-4
    
    ####### Augmentation imagenet mean&std #######
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    ####### Device #######
    device = 'cpu'
    
    ####### Optimizer #######
    optimizer_type = 'adam'  # 'SGD', 'adam', 'ranger', 'adamw'
    weight_decay = 0.1 # adamw params
    
    ####### Scheduler #######
    scheduler_type = "ReduceLROnPlateau" # 'OneCycleLR' , 'ReduceLROnPlateau' , 'Shopee_Custom_scheduler' ,  None
    
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    ReduceLROnPlateau_PARAMS = {
        'factor' : 0.5, # lr이 감소될때 얼마만큼 조절할지 감소량을 정한다. gamma 인자와 같은 기능수행
        'patience' : 5, # patience=5라고 했을때, Loss가 5 Epoch이후에도 감소하지 않는다면 그때 lr값을 감소
        'min_lr' : 1e-07, # 학습률에 대한 하한
        'mode' : "min", # 새 lr과 이전 lr의 차이가 eps보다 작으면 업데이트가 무시됨
    }
    
    # https://www.kaggle.com/easter3163/solution
    OneCycleLR_PARAMS = {
        "max_lr" : 0.1,
        "anneal_strategy" : "cos", # cos, linear
        "pct_start" : 0.2, #  learning rate를 언제까지 증가시킬지 epoch에 대한 비율로 나타냄 default:0.3 ex)100epoch일 때, 30epoch까지 증가
    } 
    
    #OneCycleLR scheduler을 쓰면, 기본 하이퍼파라미터 상태에서는 max_lr의 1/25의 lr에서 시작하여
    #총 Epoch의 1/4만큼 max_lr까지 lr이 증가하다가 그 이후는 max_lr의 1/10000까지 lr이 감소합니다. 
    #증가하거나 감소할 때는 cosine함수의 모양을 따릅니다.
    
    
    