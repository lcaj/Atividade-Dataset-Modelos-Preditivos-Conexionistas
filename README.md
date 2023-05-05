# Projeto Final - Modelos Preditivos Conexionistas

### Luiz Carlos de Araújo Júnior

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
|Deteção de Objetos|YOLOv5|Tensorflow|

## Sobre o Dataset

Esse dataset tem como objetivo analisar a doença conhecida como Herpes Zoster.  **??%**.

## Performance

O modelo treinado possui performance de **47%**.

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
    wandb: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
wandb: Currently logged in as: lcaj. Use `wandb login --relogin` to force relogin
train: weights=yolov5s.pt, cfg=, data=/content/yolov5/Atividade-Dataset-Modelos-Preditivos-Conexionistas-5/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=300, batch_size=64, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
requirements: /content/requirements.txt not found, check failed.
YOLOv5 🚀 v7.0-162-gc3e4e94 Python-3.10.11 torch-2.0.0+cu118 CUDA:0 (Tesla T4, 15102MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.15.1
wandb: Run data is saved locally in /content/yolov5/wandb/run-20230505_132601-i45nf0nj
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run grievous-senate-5
wandb: ⭐️ View project at https://wandb.ai/lcaj/YOLOv5
wandb: 🚀 View run at https://wandb.ai/lcaj/YOLOv5/runs/i45nf0nj
Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     21576  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning /content/yolov5/Atividade-Dataset-Modelos-Preditivos-Conexionistas-5/train/labels.cache... 152 images, 0 backgrounds, 0 corrupt: 100% 152/152 [00:00<?, ?it/s]
train: Caching images (0.2GB ram): 100% 152/152 [00:01<00:00, 125.77it/s]
val: Scanning /content/yolov5/Atividade-Dataset-Modelos-Preditivos-Conexionistas-5/valid/labels.cache... 23 images, 0 backgrounds, 0 corrupt: 100% 23/23 [00:00<?, ?it/s]
val: Caching images (0.0GB ram): 100% 23/23 [00:00<00:00, 47.66it/s]

AutoAnchor: 5.23 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp3/labels.jpg... 
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp3
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      0/299      13.1G     0.1208    0.03183    0.04138         63        640: 100% 3/3 [00:05<00:00,  1.75s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:01<00:00,  1.82s/it]
                   all         23         35    0.00118     0.0571    0.00113   0.000144

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      1/299      13.1G     0.1191    0.03165    0.04092         70        640: 100% 3/3 [00:01<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:01<00:00,  1.23s/it]
                   all         23         35    0.00151     0.0857    0.00103    0.00013

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      2/299      13.1G      0.117    0.02935    0.04073         73        640: 100% 3/3 [00:01<00:00,  1.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.17it/s]
                   all         23         35    0.00137     0.0857   0.000843   0.000161

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      3/299      13.1G     0.1129    0.02822    0.03885         68        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.86it/s]
                   all         23         35    0.00115     0.0857   0.000684     0.0002

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      4/299      13.1G     0.1086    0.02731    0.03793         72        640: 100% 3/3 [00:01<00:00,  1.83it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.27it/s]
                   all         23         35    0.00102     0.0857   0.000741   0.000216

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      5/299      13.1G     0.1067    0.02553     0.0363         56        640: 100% 3/3 [00:01<00:00,  1.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.42it/s]
                   all         23         35    0.00263      0.229    0.00368   0.000639

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      6/299      13.1G      0.104    0.02627    0.03509         63        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.75it/s]
                   all         23         35    0.00348      0.286    0.00433    0.00081

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      7/299      13.1G     0.1013    0.02597    0.03312         66        640: 100% 3/3 [00:01<00:00,  1.57it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.15it/s]
                   all         23         35    0.00375      0.143    0.00594    0.00162

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      8/299      13.1G     0.1002    0.02627    0.03259         63        640: 100% 3/3 [00:01<00:00,  1.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.60it/s]
                   all         23         35    0.00598        0.2     0.0206    0.00416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      9/299      13.1G    0.09982    0.02611    0.03001         62        640: 100% 3/3 [00:01<00:00,  1.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.95it/s]
                   all         23         35    0.00484      0.143    0.00366   0.000584

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     10/299      13.1G    0.09483    0.02803    0.02838         74        640: 100% 3/3 [00:01<00:00,  1.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.26it/s]
                   all         23         35    0.00185     0.0571    0.00369   0.000636

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     11/299      13.1G    0.09137    0.02625    0.02616         61        640: 100% 3/3 [00:01<00:00,  1.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.66it/s]
                   all         23         35    0.00865      0.514     0.0225    0.00494

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     12/299      13.1G    0.08818     0.0273    0.02488         52        640: 100% 3/3 [00:01<00:00,  1.62it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.63it/s]
                   all         23         35    0.00876      0.457     0.0331    0.00773

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     13/299      13.1G    0.08699    0.02745    0.02268         54        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.07it/s]
                   all         23         35    0.00969      0.371     0.0108     0.0023

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     14/299      13.1G     0.0865    0.02827    0.02138         76        640: 100% 3/3 [00:01<00:00,  1.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.51it/s]
                   all         23         35    0.00908      0.486     0.0134    0.00248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     15/299      13.1G    0.08183    0.02954    0.01978         59        640: 100% 3/3 [00:01<00:00,  1.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.41it/s]
                   all         23         35    0.00885      0.657     0.0298     0.0078

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     16/299      13.1G    0.08175    0.02781    0.01853         77        640: 100% 3/3 [00:01<00:00,  1.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.55it/s]
                   all         23         35    0.00799      0.629     0.0186    0.00434

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     17/299      13.1G    0.07849    0.03107    0.01676         77        640: 100% 3/3 [00:01<00:00,  1.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.19it/s]
                   all         23         35     0.0139        0.2    0.00908    0.00212

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     18/299      13.1G    0.08075    0.02923    0.01651         58        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.00it/s]
                   all         23         35     0.0504      0.114     0.0237    0.00432

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     19/299      13.1G    0.07689    0.02772    0.01377         65        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.38it/s]
                   all         23         35     0.0877      0.314     0.0521     0.0122

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     20/299      13.1G    0.07689    0.02887    0.01361         60        640: 100% 3/3 [00:01<00:00,  1.74it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.37it/s]
                   all         23         35     0.0796      0.158     0.0472    0.00846

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     21/299      13.1G     0.0772    0.02366    0.01199         51        640: 100% 3/3 [00:01<00:00,  1.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.59it/s]
                   all         23         35       0.61     0.0286     0.0357     0.0181

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     22/299      13.1G    0.07738    0.02591    0.01104         61        640: 100% 3/3 [00:01<00:00,  1.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.40it/s]
                   all         23         35     0.0371      0.114     0.0111    0.00306

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     23/299      13.1G    0.07604    0.02704   0.009647         74        640: 100% 3/3 [00:01<00:00,  1.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.36it/s]
                   all         23         35     0.0283     0.0571     0.0204    0.00579

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     24/299      13.1G    0.07791    0.02565     0.0111         58        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.89it/s]
                   all         23         35      0.138     0.0571     0.0498     0.0163

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     25/299      13.1G    0.07573    0.02453   0.008116         48        640: 100% 3/3 [00:02<00:00,  1.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.17it/s]
                   all         23         35     0.0955      0.114     0.0365     0.0148

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     26/299      13.1G    0.07498     0.0253   0.007128         55        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.31it/s]
                   all         23         35      0.191      0.257      0.106     0.0316

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     27/299      13.1G    0.07498    0.02773   0.007312         66        640: 100% 3/3 [00:01<00:00,  1.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.58it/s]
                   all         23         35      0.147      0.192     0.0547     0.0169

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     28/299      13.1G    0.07291    0.02406    0.00794         54        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.23it/s]
                   all         23         35     0.0925        0.2     0.0576     0.0135

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     29/299      13.1G    0.07461    0.02202    0.00857         54        640: 100% 3/3 [00:01<00:00,  1.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.15it/s]
                   all         23         35     0.0596      0.171     0.0332    0.00889

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     30/299      13.1G    0.07215    0.02339   0.007651         54        640: 100% 3/3 [00:01<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.82it/s]
                   all         23         35      0.135      0.171     0.0736     0.0245

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     31/299      13.1G     0.0733    0.02444   0.006963         60        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.24it/s]
                   all         23         35     0.0578      0.171     0.0237    0.00576

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     32/299      13.1G    0.07117    0.02632   0.006928         67        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.64it/s]
                   all         23         35    0.00581      0.143    0.00453   0.000874

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     33/299      13.1G    0.07069    0.02423   0.008008         62        640: 100% 3/3 [00:01<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.74it/s]
                   all         23         35     0.0398     0.0857     0.0129    0.00408

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     34/299      13.1G    0.07034     0.0262    0.01322         70        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.57it/s]
                   all         23         35    0.00462      0.114    0.00318    0.00131

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     35/299      13.1G    0.07362    0.02272   0.009561         60        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.51it/s]
                   all         23         35     0.0608     0.0286     0.0074    0.00182

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     36/299      13.1G    0.06966    0.02314   0.009656         57        640: 100% 3/3 [00:02<00:00,  1.13it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.38it/s]
                   all         23         35    0.00257      0.143    0.00186   0.000367

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     37/299      13.1G    0.07149    0.02614    0.00709         82        640: 100% 3/3 [00:02<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.29it/s]
                   all         23         35     0.0839     0.0571      0.011    0.00249

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     38/299      13.1G    0.06925    0.02471   0.006201         67        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.06it/s]
                   all         23         35      0.108      0.114     0.0412    0.00984

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     39/299      13.1G    0.06958    0.02335   0.005932         59        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.40it/s]
                   all         23         35     0.0607     0.0857     0.0115    0.00321

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     40/299      13.1G    0.06847     0.0225   0.005172         63        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.58it/s]
                   all         23         35      0.163        0.2     0.0585     0.0131

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     41/299      13.1G     0.0681    0.02555   0.006392         81        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.54it/s]
                   all         23         35     0.0892      0.143     0.0286     0.0062

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     42/299      13.1G    0.06905    0.02585   0.005116         92        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.49it/s]
                   all         23         35     0.0388     0.0857     0.0396     0.0102

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     43/299      13.1G    0.06951    0.02265   0.005593         58        640: 100% 3/3 [00:02<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.11it/s]
                   all         23         35     0.0604      0.143     0.0392     0.0112

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     44/299      13.1G    0.06498      0.027   0.005093         71        640: 100% 3/3 [00:01<00:00,  1.62it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.27it/s]
                   all         23         35     0.0705      0.171     0.0392    0.00986

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     45/299      13.1G    0.06644     0.0225    0.00509         59        640: 100% 3/3 [00:01<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.19it/s]
                   all         23         35       0.21      0.257     0.0899     0.0197

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     46/299      13.1G    0.06669    0.02297   0.004686         65        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.40it/s]
                   all         23         35      0.203      0.182      0.102     0.0257

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     47/299      13.1G    0.06441    0.02526   0.003802         66        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.76it/s]
                   all         23         35      0.158        0.2     0.0754     0.0159

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     48/299      13.1G    0.06437    0.02128   0.003853         53        640: 100% 3/3 [00:01<00:00,  1.57it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.96it/s]
                   all         23         35     0.0655      0.171     0.0358    0.00743

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     49/299      13.1G    0.06503    0.02046   0.003286         48        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.03it/s]
                   all         23         35      0.131      0.114     0.0384    0.00764

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     50/299      13.1G    0.06277    0.02165   0.003908         55        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.98it/s]
                   all         23         35      0.128        0.2       0.08     0.0212

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     51/299      13.1G     0.0617     0.0214   0.003287         56        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.60it/s]
                   all         23         35      0.151        0.2     0.0989     0.0242

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     52/299      13.1G    0.06474    0.02132   0.004883         69        640: 100% 3/3 [00:01<00:00,  1.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.11it/s]
                   all         23         35      0.177      0.114     0.0534     0.0163

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     53/299      13.1G    0.06365    0.02226   0.003245         59        640: 100% 3/3 [00:01<00:00,  1.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.91it/s]
                   all         23         35      0.283      0.143      0.144     0.0346

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     54/299      13.1G    0.06099    0.02154   0.003198         60        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.58it/s]
                   all         23         35      0.154      0.229      0.103     0.0158

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     55/299      13.1G    0.06011    0.02179   0.003779         64        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.79it/s]
                   all         23         35       0.14      0.257      0.078     0.0171

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     56/299      13.1G    0.05834    0.01913   0.003558         56        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.63it/s]
                   all         23         35      0.162      0.229      0.108     0.0252

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     57/299      13.1G    0.06052    0.02399   0.003926         81        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.87it/s]
                   all         23         35     0.0693      0.114     0.0356    0.00957

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     58/299      13.1G    0.05844    0.01926    0.00304         58        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.59it/s]
                   all         23         35     0.0856        0.2     0.0745     0.0167

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     59/299      13.1G    0.06111    0.02026   0.003308         57        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.76it/s]
                   all         23         35      0.237      0.229      0.117     0.0292

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     60/299      13.1G    0.05656    0.02147   0.002813         52        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.30it/s]
                   all         23         35      0.196      0.229      0.148     0.0338

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     61/299      13.1G    0.05907      0.021    0.00439         60        640: 100% 3/3 [00:02<00:00,  1.41it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.23it/s]
                   all         23         35       0.46      0.341      0.239      0.072

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     62/299      13.1G    0.05906    0.02188   0.003091         61        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.78it/s]
                   all         23         35      0.285      0.286      0.155     0.0497

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     63/299      13.1G    0.05667    0.01993   0.002374         52        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.57it/s]
                   all         23         35      0.365      0.257      0.196     0.0611

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     64/299      13.1G    0.05798    0.01977   0.002906         53        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.64it/s]
                   all         23         35      0.307      0.314      0.166     0.0422

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     65/299      13.1G    0.05612    0.01899   0.002925         54        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.68it/s]
                   all         23         35      0.283      0.229      0.113     0.0381

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     66/299      13.1G    0.05689    0.02435   0.002127         84        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.07it/s]
                   all         23         35      0.343      0.314      0.176     0.0517

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     67/299      13.1G    0.05832    0.02128   0.002728         55        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.94it/s]
                   all         23         35      0.346      0.286      0.165     0.0602

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     68/299      13.1G    0.05666    0.02054   0.002294         64        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.78it/s]
                   all         23         35      0.264      0.229      0.116     0.0445

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     69/299      13.1G    0.05621    0.01928   0.003233         61        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.98it/s]
                   all         23         35      0.355        0.2      0.133     0.0421

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     70/299      13.1G      0.055    0.01894   0.002057         52        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.93it/s]
                   all         23         35      0.259      0.314      0.157     0.0525

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     71/299      13.1G    0.05697    0.01919   0.002074         53        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.34it/s]
                   all         23         35        0.5      0.286      0.278     0.0698

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     72/299      13.1G    0.05429    0.02092   0.002043         76        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.13it/s]
                   all         23         35      0.345      0.171      0.159     0.0482

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     73/299      13.1G    0.05403    0.01929   0.001418         46        640: 100% 3/3 [00:02<00:00,  1.44it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.58it/s]
                   all         23         35      0.434      0.143      0.151     0.0433

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     74/299      13.1G    0.05441    0.01837   0.002894         63        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.92it/s]
                   all         23         35      0.363      0.171      0.167     0.0488

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     75/299      13.1G    0.05149    0.02042   0.001492         66        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.15it/s]
                   all         23         35      0.284        0.2      0.131      0.037

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     76/299      13.1G    0.05147    0.01785    0.00204         45        640: 100% 3/3 [00:01<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.62it/s]
                   all         23         35      0.328      0.286      0.173      0.059

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     77/299      13.1G    0.05462    0.01795   0.001735         60        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.45it/s]
                   all         23         35       0.44      0.343      0.244     0.0775

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     78/299      13.1G    0.05009    0.02082   0.001851         59        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.95it/s]
                   all         23         35      0.255      0.343      0.162     0.0696

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     79/299      13.1G    0.05034    0.01817   0.002266         54        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.78it/s]
                   all         23         35      0.293      0.314      0.156     0.0716

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     80/299      13.1G    0.05194    0.02062   0.002258         82        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.76it/s]
                   all         23         35      0.171      0.229     0.0844     0.0349

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     81/299      13.1G    0.05129    0.01864   0.002297         71        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.45it/s]
                   all         23         35      0.462      0.229      0.161     0.0641

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     82/299      13.1G    0.05091    0.01697   0.002366         47        640: 100% 3/3 [00:01<00:00,  1.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.05it/s]
                   all         23         35      0.172        0.2      0.125     0.0342

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     83/299      13.1G    0.04826    0.01835   0.001885         51        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.75it/s]
                   all         23         35       0.27      0.286      0.182     0.0501

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     84/299      13.1G    0.05009    0.01512   0.001451         57        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.70it/s]
                   all         23         35      0.325        0.2      0.106     0.0388

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     85/299      13.1G     0.0495     0.0189   0.001716         70        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.44it/s]
                   all         23         35      0.239      0.269      0.109     0.0418

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     86/299      13.1G    0.04834    0.01724   0.001706         53        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.49it/s]
                   all         23         35      0.329      0.229      0.144     0.0413

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     87/299      13.1G    0.04833    0.01852   0.001837         57        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.13it/s]
                   all         23         35      0.197        0.2     0.0929     0.0305

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     88/299      13.1G    0.04834     0.0177   0.002804         56        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.56it/s]
                   all         23         35      0.277      0.143      0.101     0.0325

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     89/299      13.1G    0.04918    0.01989   0.001929         82        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.21it/s]
                   all         23         35      0.433      0.114      0.105     0.0379

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     90/299      13.1G    0.04743    0.01808   0.002171         87        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.59it/s]
                   all         23         35      0.266      0.229      0.124     0.0419

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     91/299      13.1G    0.04582    0.01696   0.001893         53        640: 100% 3/3 [00:01<00:00,  1.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.64it/s]
                   all         23         35      0.314        0.2      0.139     0.0511

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     92/299      13.1G    0.04865    0.01727   0.001795         54        640: 100% 3/3 [00:01<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.02it/s]
                   all         23         35      0.268      0.143     0.0991      0.032

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     93/299      13.1G    0.04664     0.0181   0.001912         58        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.70it/s]
                   all         23         35      0.228      0.229      0.124     0.0401

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     94/299      13.1G     0.0445    0.01752   0.002287         59        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.76it/s]
                   all         23         35      0.306      0.171      0.138     0.0474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     95/299      13.1G    0.04592    0.01788   0.002417         59        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.88it/s]
                   all         23         35      0.489      0.286      0.208     0.0801

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     96/299      13.1G    0.04394    0.01723   0.001727         62        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.99it/s]
                   all         23         35        0.5      0.229      0.202     0.0836

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     97/299      13.1G     0.0424    0.01541   0.002248         52        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.14it/s]
                   all         23         35      0.383        0.2       0.18      0.053

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     98/299      13.1G    0.04565    0.01661   0.002253         63        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.23it/s]
                   all         23         35      0.421        0.2      0.159     0.0472

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     99/299      13.1G     0.0457    0.01624   0.002256         43        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.80it/s]
                   all         23         35      0.501        0.2      0.162     0.0692

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    100/299      13.1G    0.04669    0.01549   0.001704         53        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.52it/s]
                   all         23         35      0.418      0.171      0.139     0.0503

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    101/299      13.1G    0.04429    0.01736   0.001416         67        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.03it/s]
                   all         23         35      0.377      0.286      0.153     0.0571

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    102/299      13.1G    0.04352    0.01476   0.001716         43        640: 100% 3/3 [00:02<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.45it/s]
                   all         23         35      0.258      0.114      0.102      0.041

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    103/299      13.1G    0.04248    0.01663   0.001594         60        640: 100% 3/3 [00:02<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.66it/s]
                   all         23         35      0.325      0.257      0.134     0.0457

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    104/299      13.1G    0.04273    0.01552   0.001216         62        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.57it/s]
                   all         23         35      0.208      0.229      0.117     0.0416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    105/299      13.1G    0.04412     0.0169   0.001562         62        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.96it/s]
                   all         23         35      0.218      0.229      0.124     0.0389

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    106/299      13.1G    0.04336    0.01686    0.00159         53        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.75it/s]
                   all         23         35      0.192      0.286      0.118     0.0415

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    107/299      13.1G    0.04528    0.01697   0.001338         67        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.36it/s]
                   all         23         35      0.192      0.314      0.168     0.0427

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    108/299      13.1G    0.04532    0.01555   0.001367         63        640: 100% 3/3 [00:01<00:00,  1.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.26it/s]
                   all         23         35      0.326        0.2      0.107       0.04

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    109/299      13.1G    0.04112    0.01593    0.00148         71        640: 100% 3/3 [00:02<00:00,  1.42it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.04it/s]
                   all         23         35      0.164      0.229     0.0976     0.0302

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    110/299      13.1G    0.04428    0.01552   0.001322         53        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.81it/s]
                   all         23         35      0.172      0.171     0.0907      0.022

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    111/299      13.1G    0.04132    0.01588  0.0007935         66        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.90it/s]
                   all         23         35      0.198      0.289      0.136     0.0427

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    112/299      13.1G    0.04206    0.01621   0.001162         60        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.87it/s]
                   all         23         35      0.283      0.257      0.127     0.0324

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    113/299      13.1G    0.04079    0.01576    0.00143         68        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.21it/s]
                   all         23         35       0.22      0.229      0.107     0.0326

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    114/299      13.1G     0.0421    0.01687   0.001697         69        640: 100% 3/3 [00:01<00:00,  1.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.75it/s]
                   all         23         35       0.22      0.314      0.125     0.0425

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    115/299      13.1G    0.04058      0.015   0.002001         55        640: 100% 3/3 [00:01<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.70it/s]
                   all         23         35       0.28      0.229      0.157     0.0489

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    116/299      13.1G    0.04291    0.01591   0.001277         69        640: 100% 3/3 [00:01<00:00,  1.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.66it/s]
                   all         23         35      0.248      0.229      0.125     0.0438

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    117/299      13.1G    0.03933    0.01565   0.001743         61        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.71it/s]
                   all         23         35      0.266      0.229      0.149     0.0582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    118/299      13.1G    0.03884    0.01369   0.001236         55        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.95it/s]
                   all         23         35      0.312      0.257      0.164     0.0582

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    119/299      13.1G    0.04344    0.01466   0.001262         51        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.52it/s]
                   all         23         35      0.308      0.257      0.145      0.055

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    120/299      13.1G    0.04062    0.01451  0.0009113         68        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.32it/s]
                   all         23         35      0.393      0.257      0.174     0.0565

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    121/299      13.1G    0.04042    0.01589   0.000949         58        640: 100% 3/3 [00:01<00:00,  1.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.11it/s]
                   all         23         35       0.33      0.171      0.113     0.0449

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    122/299      13.1G    0.03863    0.01518   0.001447         80        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.68it/s]
                   all         23         35       0.29      0.257      0.144     0.0483

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    123/299      13.1G    0.04067     0.0157   0.001981         67        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.84it/s]
                   all         23         35      0.351      0.314      0.171     0.0528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    124/299      13.1G    0.03931    0.01607   0.001399         56        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.42it/s]
                   all         23         35      0.231      0.257      0.119      0.039

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    125/299      13.1G     0.0402    0.01519  0.0009916         70        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.85it/s]
                   all         23         35      0.442      0.257      0.198     0.0531

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    126/299      13.1G    0.04015    0.01501   0.001274         65        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.07it/s]
                   all         23         35      0.321      0.229      0.146     0.0414

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    127/299      13.1G    0.03945    0.01507    0.00125         59        640: 100% 3/3 [00:02<00:00,  1.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.42it/s]
                   all         23         35      0.291      0.229      0.167     0.0549

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    128/299      13.1G    0.03903    0.01492    0.00105         72        640: 100% 3/3 [00:01<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.12it/s]
                   all         23         35      0.319      0.171      0.141     0.0443

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    129/299      13.1G    0.03954    0.01482   0.001091         74        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.74it/s]
                   all         23         35      0.249      0.171      0.146     0.0413

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    130/299      13.1G    0.03651    0.01321   0.001092         48        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.80it/s]
                   all         23         35      0.365      0.143      0.146     0.0657

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    131/299      13.1G    0.03757    0.01437   0.001082         61        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.01it/s]
                   all         23         35       0.47      0.257      0.257     0.0914

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    132/299      13.1G     0.0381    0.01356   0.001413         64        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.58it/s]
                   all         23         35      0.479        0.2      0.175     0.0672

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    133/299      13.1G    0.03839    0.01415   0.001311         55        640: 100% 3/3 [00:02<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.29it/s]
                   all         23         35      0.348        0.2      0.187     0.0654

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    134/299      13.1G    0.03802    0.01416   0.001701         67        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.95it/s]
                   all         23         35      0.305      0.226      0.213     0.0591

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    135/299      13.1G    0.04012    0.01389   0.001028         59        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.05it/s]
                   all         23         35      0.287      0.173      0.174     0.0656

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    136/299      13.1G    0.03626    0.01399   0.001666         60        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.57it/s]
                   all         23         35      0.186      0.143      0.126     0.0478

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    137/299      13.1G    0.03762    0.01508   0.001032         70        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.03it/s]
                   all         23         35      0.432      0.229      0.214     0.0723

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    138/299      13.1G     0.0368    0.01582   0.001331         69        640: 100% 3/3 [00:02<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.37it/s]
                   all         23         35      0.345      0.241      0.188     0.0732

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    139/299      13.1G    0.03714    0.01397   0.001214         67        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.76it/s]
                   all         23         35      0.319      0.257      0.223     0.0717

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    140/299      13.1G    0.03612     0.0146   0.001007         54        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.57it/s]
                   all         23         35      0.332      0.286      0.211     0.0779

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    141/299      13.1G    0.03727    0.01468   0.001209         71        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.97it/s]
                   all         23         35      0.428        0.2      0.195     0.0657

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    142/299      13.1G    0.03459    0.01476  0.0009005         71        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.72it/s]
                   all         23         35      0.315      0.257      0.173     0.0716

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    143/299      13.1G    0.03591    0.01446  0.0006505         83        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.45it/s]
                   all         23         35      0.314      0.339      0.228     0.0691

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    144/299      13.1G    0.03649    0.01338  0.0008317         53        640: 100% 3/3 [00:01<00:00,  1.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.31it/s]
                   all         23         35      0.308      0.257      0.184     0.0635

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    145/299      13.1G    0.03559     0.0136   0.001091         65        640: 100% 3/3 [00:01<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.04it/s]
                   all         23         35      0.281      0.257       0.19     0.0516

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    146/299      13.1G    0.03539    0.01369  0.0009857         64        640: 100% 3/3 [00:01<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.53it/s]
                   all         23         35      0.275      0.286      0.185      0.054

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    147/299      13.1G     0.0364    0.01206  0.0009329         53        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.95it/s]
                   all         23         35      0.323        0.2      0.157     0.0488

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    148/299      13.1G    0.03725    0.01503  0.0009158         77        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.91it/s]
                   all         23         35      0.291      0.229      0.167      0.051

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    149/299      13.1G    0.03667    0.01184   0.001281         54        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.61it/s]
                   all         23         35      0.231      0.257      0.144      0.045

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    150/299      13.1G    0.03538    0.01292   0.001105         50        640: 100% 3/3 [00:01<00:00,  1.53it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.53it/s]
                   all         23         35        0.3      0.171      0.139     0.0557

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    151/299      13.1G     0.0368    0.01278   0.001824         53        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.61it/s]
                   all         23         35      0.235      0.286      0.188     0.0547

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    152/299      13.1G    0.03501    0.01369   0.001064         67        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.11it/s]
                   all         23         35      0.215      0.235       0.14     0.0544

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    153/299      13.1G     0.0361    0.01283   0.000888         47        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.05it/s]
                   all         23         35      0.206      0.208      0.143     0.0516

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    154/299      13.1G    0.03558    0.01315  0.0008414         60        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.89it/s]
                   all         23         35      0.326      0.262      0.215     0.0661

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    155/299      13.1G    0.03328    0.01197  0.0009656         42        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.68it/s]
                   all         23         35      0.478      0.229      0.231     0.0647

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    156/299      13.1G    0.03408    0.01426    0.00088         82        640: 100% 3/3 [00:01<00:00,  1.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.17it/s]
                   all         23         35       0.31      0.244      0.217     0.0704

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    157/299      13.1G    0.03664    0.01147  0.0006593         50        640: 100% 3/3 [00:02<00:00,  1.47it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.43it/s]
                   all         23         35       0.32      0.257      0.182     0.0617

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    158/299      13.1G    0.03495    0.01286  0.0008446         57        640: 100% 3/3 [00:01<00:00,  1.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.15it/s]
                   all         23         35      0.366        0.2      0.158     0.0512

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    159/299      13.1G     0.0342    0.01378  0.0007265         80        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.79it/s]
                   all         23         35      0.308        0.2      0.155     0.0501

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    160/299      13.1G    0.03247    0.01224  0.0006644         52        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.61it/s]
                   all         23         35      0.376        0.2      0.168     0.0516

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    161/299      13.1G    0.03184    0.01257  0.0007168         56        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.76it/s]
                   all         23         35       0.39      0.171      0.138     0.0489

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    162/299      13.1G    0.03348    0.01357  0.0007212         73        640: 100% 3/3 [00:01<00:00,  1.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.40it/s]
                   all         23         35      0.384      0.171       0.15     0.0533

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    163/299      13.1G    0.03351    0.01419  0.0008412         74        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  1.45it/s]
                   all         23         35      0.263      0.257      0.146     0.0511

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    164/299      13.1G    0.03502    0.01325  0.0008022         50        640: 100% 3/3 [00:01<00:00,  1.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.09it/s]
                   all         23         35      0.412      0.171      0.177     0.0578

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    165/299      13.1G     0.0338    0.01358   0.001037         63        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.73it/s]
                   all         23         35      0.331      0.229      0.151     0.0547

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    166/299      13.1G    0.03318     0.0115  0.0007002         56        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.39it/s]
                   all         23         35      0.251      0.171      0.146     0.0592

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    167/299      13.1G    0.03229    0.01265  0.0007686         61        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.83it/s]
                   all         23         35      0.372      0.229       0.16     0.0583

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    168/299      13.1G    0.03074    0.01194  0.0006952         50        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.18it/s]
                   all         23         35      0.348      0.143      0.139     0.0514

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    169/299      13.1G    0.03248     0.0116   0.001007         60        640: 100% 3/3 [00:02<00:00,  1.47it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.15it/s]
                   all         23         35      0.381      0.211      0.174     0.0584

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    170/299      13.1G    0.03218    0.01212  0.0007265         49        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.26it/s]
                   all         23         35       0.22      0.143       0.12     0.0402

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    171/299      13.1G    0.03224    0.01216   0.000939         56        640: 100% 3/3 [00:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.65it/s]
                   all         23         35       0.53      0.229      0.207     0.0539

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    172/299      13.1G    0.03004    0.01215  0.0007543         74        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.33it/s]
                   all         23         35      0.363      0.143      0.133     0.0403

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    173/299      13.1G    0.03158    0.01221  0.0006913         59        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.73it/s]
                   all         23         35      0.314        0.2       0.13     0.0431

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    174/299      13.1G    0.03225    0.01178   0.001102         62        640: 100% 3/3 [00:01<00:00,  1.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.19it/s]
                   all         23         35      0.312      0.229      0.138      0.047

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    175/299      13.1G    0.03178    0.01146   0.001016         58        640: 100% 3/3 [00:02<00:00,  1.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.21it/s]
                   all         23         35      0.209      0.114      0.114      0.043

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    176/299      13.1G    0.03174    0.01235  0.0006439         63        640: 100% 3/3 [00:01<00:00,  1.62it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.30it/s]
                   all         23         35      0.196      0.171      0.134      0.053

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    177/299      13.1G     0.0327    0.01301    0.00113         66        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.65it/s]
                   all         23         35      0.269      0.171      0.151       0.06

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    178/299      13.1G    0.03372    0.01217  0.0007628         45        640: 100% 3/3 [00:01<00:00,  1.73it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.02it/s]
                   all         23         35      0.284      0.229      0.176     0.0616

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    179/299      13.1G    0.03015    0.01196  0.0007248         55        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.90it/s]
                   all         23         35      0.175      0.143      0.129     0.0481

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    180/299      13.1G    0.03076    0.01303   0.001185         82        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.79it/s]
                   all         23         35      0.236      0.171      0.144     0.0478

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    181/299      13.1G    0.03253    0.01283  0.0005568         83        640: 100% 3/3 [00:02<00:00,  1.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.10it/s]
                   all         23         35      0.106      0.371      0.135     0.0428

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    182/299      13.1G     0.0319     0.0111  0.0008896         46        640: 100% 3/3 [00:01<00:00,  1.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.98it/s]
                   all         23         35     0.0847      0.429      0.129     0.0449

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    183/299      13.1G    0.03011    0.01343  0.0007559         79        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.39it/s]
                   all         23         35      0.316      0.237      0.183     0.0706

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    184/299      13.1G    0.03067    0.01156  0.0006565         62        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.68it/s]
                   all         23         35      0.286      0.195      0.144     0.0588

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    185/299      13.1G    0.02919     0.0113   0.001041         60        640: 100% 3/3 [00:01<00:00,  1.71it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.78it/s]
                   all         23         35      0.484      0.171       0.15      0.055

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    186/299      13.1G    0.03074    0.01104  0.0006913         61        640: 100% 3/3 [00:01<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.27it/s]
                   all         23         35      0.509        0.2      0.172     0.0677

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    187/299      13.1G    0.02913     0.0124  0.0005435         84        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.19it/s]
                   all         23         35      0.469      0.227      0.204     0.0758

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    188/299      13.1G    0.02971    0.01278  0.0008152         80        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.99it/s]
                   all         23         35      0.491        0.2      0.192     0.0673

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    189/299      13.1G    0.03105    0.01143  0.0008241         55        640: 100% 3/3 [00:01<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.08it/s]
                   all         23         35      0.392      0.257      0.173     0.0563

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    190/299      13.1G    0.02774    0.01105  0.0007956         67        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.81it/s]
                   all         23         35      0.233      0.171      0.127     0.0402

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    191/299      13.1G    0.02891    0.01094  0.0006249         53        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.62it/s]
                   all         23         35      0.218      0.207      0.129     0.0432

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    192/299      13.1G    0.02871    0.01109  0.0005945         61        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.43it/s]
                   all         23         35      0.217      0.229      0.127     0.0441

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    193/299      13.1G     0.0296    0.01139   0.000595         67        640: 100% 3/3 [00:02<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.34it/s]
                   all         23         35      0.162        0.2      0.112     0.0372

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    194/299      13.1G     0.0308     0.0102  0.0005698         47        640: 100% 3/3 [00:01<00:00,  1.58it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.35it/s]
                   all         23         35      0.264      0.257      0.157     0.0432

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    195/299      13.1G    0.02838    0.01243  0.0007134         59        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.31it/s]
                   all         23         35       0.31      0.314      0.175     0.0566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    196/299      13.1G    0.02982    0.01393  0.0006908         86        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.59it/s]
                   all         23         35      0.235      0.257      0.148     0.0533

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    197/299      13.1G    0.02634    0.01172  0.0006257         65        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.79it/s]
                   all         23         35      0.279      0.286      0.167     0.0637

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    198/299      13.1G    0.02792   0.009931  0.0007521         53        640: 100% 3/3 [00:01<00:00,  1.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.43it/s]
                   all         23         35      0.298      0.314      0.188     0.0747

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    199/299      13.1G    0.03061    0.01154  0.0006983         64        640: 100% 3/3 [00:01<00:00,  1.51it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.35it/s]
                   all         23         35      0.352      0.286      0.179     0.0645

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    200/299      13.1G    0.02715    0.01074  0.0005978         53        640: 100% 3/3 [00:01<00:00,  1.59it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.44it/s]
                   all         23         35      0.318      0.314       0.16     0.0517

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    201/299      13.1G     0.0269    0.01218  0.0007613         64        640: 100% 3/3 [00:01<00:00,  1.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.75it/s]
                   all         23         35      0.282      0.314      0.163     0.0577

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    202/299      13.1G    0.02837     0.0121  0.0007135         84        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.71it/s]
                   all         23         35      0.283      0.343      0.176     0.0669

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    203/299      13.1G    0.02893    0.01175  0.0004989         67        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.66it/s]
                   all         23         35      0.313      0.314      0.182     0.0658

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    204/299      13.1G    0.02986    0.01142  0.0005454         64        640: 100% 3/3 [00:01<00:00,  1.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.51it/s]
                   all         23         35      0.322      0.257      0.163     0.0645

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    205/299      13.1G    0.02734    0.01212  0.0006071         74        640: 100% 3/3 [00:02<00:00,  1.47it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.51it/s]
                   all         23         35      0.201        0.2      0.131     0.0415

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    206/299      13.1G    0.02874    0.01074  0.0005902         52        640: 100% 3/3 [00:02<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.18it/s]
                   all         23         35      0.203      0.171      0.123     0.0601

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    207/299      13.1G    0.02762    0.01026  0.0005192         49        640: 100% 3/3 [00:01<00:00,  1.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.64it/s]
                   all         23         35      0.245        0.2      0.113     0.0428

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    208/299      13.1G    0.02871    0.01072   0.000679         71        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.03it/s]
                   all         23         35      0.324      0.143      0.121     0.0435

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    209/299      13.1G    0.02691    0.01087   0.000443         56        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.65it/s]
                   all         23         35      0.361      0.143      0.122     0.0558

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    210/299      13.1G    0.02971    0.01248  0.0004769         73        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.56it/s]
                   all         23         35      0.175        0.2      0.136     0.0438

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    211/299      13.1G    0.02717    0.01083  0.0004865         77        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.44it/s]
                   all         23         35      0.263      0.229      0.154      0.058

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    212/299      13.1G     0.0267    0.01065  0.0008388         70        640: 100% 3/3 [00:02<00:00,  1.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.35it/s]
                   all         23         35      0.225      0.171      0.129     0.0522

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    213/299      13.1G    0.02825    0.01125   0.001815         68        640: 100% 3/3 [00:01<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.89it/s]
                   all         23         35      0.191        0.2       0.13     0.0476

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    214/299      13.1G    0.02951    0.01034  0.0006829         54        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.26it/s]
                   all         23         35      0.233        0.2      0.125     0.0491

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    215/299      13.1G    0.02635    0.01076  0.0005198         58        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.86it/s]
                   all         23         35      0.198      0.114      0.106     0.0514

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    216/299      13.1G    0.02858    0.01244  0.0005755         85        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.58it/s]
                   all         23         35      0.225      0.171      0.115     0.0479

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    217/299      13.1G    0.02819   0.009686  0.0005199         44        640: 100% 3/3 [00:02<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.52it/s]
                   all         23         35      0.197      0.155      0.127     0.0513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    218/299      13.1G    0.02717   0.009147  0.0006572         50        640: 100% 3/3 [00:01<00:00,  1.52it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.09it/s]
                   all         23         35      0.268      0.143     0.0956     0.0401

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    219/299      13.1G    0.02647    0.01054  0.0005517         53        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.68it/s]
                   all         23         35      0.346      0.182      0.145     0.0454

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    220/299      13.1G    0.02559   0.009875  0.0005255         65        640: 100% 3/3 [00:01<00:00,  1.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.81it/s]
                   all         23         35      0.441        0.2      0.154     0.0444

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    221/299      13.1G    0.02782    0.01077  0.0004629         66        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.57it/s]
                   all         23         35      0.355      0.171      0.127      0.043

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    222/299      13.1G    0.02542    0.01115  0.0005562         78        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.82it/s]
                   all         23         35      0.235      0.171      0.127     0.0527

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    223/299      13.1G    0.02567    0.01034   0.000559         70        640: 100% 3/3 [00:01<00:00,  1.54it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.10it/s]
                   all         23         35      0.203      0.171      0.127     0.0472

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    224/299      13.1G    0.02553    0.01034  0.0005251         65        640: 100% 3/3 [00:02<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.25it/s]
                   all         23         35      0.183      0.171      0.119     0.0512

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    225/299      13.1G    0.02566   0.009924  0.0004183         63        640: 100% 3/3 [00:01<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.80it/s]
                   all         23         35      0.165      0.143      0.121      0.043

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    226/299      13.1G    0.02572    0.01141   0.001323         70        640: 100% 3/3 [00:01<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.02it/s]
                   all         23         35       0.18      0.143      0.113     0.0442

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    227/299      13.1G    0.02653    0.01073  0.0005514         72        640: 100% 3/3 [00:01<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.11it/s]
                   all         23         35      0.164      0.143      0.108     0.0385

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    228/299      13.1G    0.02535    0.01051  0.0004808         52        640: 100% 3/3 [00:01<00:00,  1.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.15it/s]
                   all         23         35      0.169      0.171      0.125     0.0434

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    229/299      13.1G    0.02471    0.01006  0.0004995         60        640: 100% 3/3 [00:02<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.37it/s]
                   all         23         35      0.176        0.2      0.131     0.0474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    230/299      13.1G    0.02429   0.009876  0.0006464         48        640: 100% 3/3 [00:01<00:00,  1.50it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  2.61it/s]
                   all         23         35      0.186      0.196      0.117     0.0446

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    231/299      13.1G    0.02596    0.01043  0.0004465         61        640: 100% 3/3 [00:01<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  3.82it/s]
                   all         23         35      0.187      0.229       0.14     0.0539
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 131, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

232 epochs completed in 0.189 hours.
Optimizer stripped from runs/train/exp3/weights/last.pt, 14.5MB
Optimizer stripped from runs/train/exp3/weights/best.pt, 14.5MB

Validating runs/train/exp3/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 1/1 [00:00<00:00,  4.13it/s]
                   all         23         35      0.454      0.257      0.257     0.0872
                herpes         23         35      0.454      0.257      0.257     0.0872
Results saved to runs/train/exp3
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▁▂▁▁▂▁▂▃▅▄▄█▇▆▄▆▅▅▄▅▅▅▅▆▅▅▅▅▆▅▆▅▅▅▄▄▅▄▅
wandb: metrics/mAP_0.5:0.95 ▁▁▁▁▁▂▁▂▂▄▄▄▇█▅▄█▆▅▄▆▆▅▇▇▅▆▅▆▆▆▇▆▆▆▅▅▅▅▆
wandb:    metrics/precision ▁▁▁▁▁▂▂▂▃▅▄▅█▇▅▇▇▆▄▄▄▆▄▅▇▅▄▅▅█▅▅▆▅▅▄▄▆▃▃
wandb:       metrics/recall ▂▄█▃▁▃▁▃▃▃▄▄▅▆▅▂▅▅▅▄▄▅▃▃▃▃▄▃▄▄▃▄▄▅▅▃▃▃▃▄
wandb:       train/box_loss █▇▆▅▅▅▅▄▄▄▄▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁
wandb:       train/cls_loss █▇▅▄▃▂▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/obj_loss █▆▆█▇▅▅▆▆▅▄▄▄▄▄▄▄▃▃▃▃▃▃▂▃▂▂▂▂▂▂▂▁▂▂▁▁▁▁▁
wandb:         val/box_loss █▅▄▄▄▃▄▃▃▂▂▂▂▁▁▁▂▁▂▁▂▁▂▁▂▁▁▁▂▂▁▁▁▁▁▂▁▂▂▂
wandb:         val/cls_loss ▅▄▄▃▃▂█▃▂▃▅▂▂▁▁▁▃▂▃▂▁▂▂▂▄▂▂▂▃▃▃▂▁▁▁▂▂▂▁▂
wandb:         val/obj_loss ▇▂▂▁▂▁▃█▄▃▃▃▂▂▃▄▃▃▃▄▃▄▅▅▅▅▅▅▅▅▇▅▆▅▆▆▇▇▇▆
wandb:                x/lr0 █▇▆▅▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                x/lr1 ▁▂▃▅▆▇████▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▃
wandb:                x/lr2 ▁▂▃▅▆▇████▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▃
wandb: 
wandb: Run summary:
wandb:           best/epoch 131
wandb:         best/mAP_0.5 0.25724
wandb:    best/mAP_0.5:0.95 0.09136
wandb:       best/precision 0.47043
wandb:          best/recall 0.25714
wandb:      metrics/mAP_0.5 0.25746
wandb: metrics/mAP_0.5:0.95 0.08725
wandb:    metrics/precision 0.45352
wandb:       metrics/recall 0.25714
wandb:       train/box_loss 0.02596
wandb:       train/cls_loss 0.00045
wandb:       train/obj_loss 0.01043
wandb:         val/box_loss 0.06627
wandb:         val/cls_loss 0.03423
wandb:         val/obj_loss 0.01708
wandb:                x/lr0 0.00241
wandb:                x/lr1 0.00241
wandb:                x/lr2 0.00241
wandb: 
wandb: 🚀 View run grievous-senate-5 at: https://wandb.ai/lcaj/YOLOv5/runs/i45nf0nj
wandb: Synced 5 W&B file(s), 13 media file(s), 1 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230505_132601-i45nf0nj/logs
wandb: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
  ```
</details>

### Evidências do treinamento

Nessa seção você deve colocar qualquer evidência do treinamento, como por exemplo gráficos de perda, performance, matriz de confusão etc.

Exemplo de adição de imagem:

https://raw.githubusercontent.com/lcaj/Atividade-Dataset-Modelos-Preditivos-Conexionistas/main/dataset/herpes/herpes%20(10).jpg

## Roboflow

https://universe.roboflow.com/cesarschool-ryrkk/atividade-dataset-modelos-preditivos-conexionistas

## HuggingFace

https://huggingface.co/spaces/lcaj/doenca/tree/main
