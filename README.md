This is an improve code  for Highly Accurate Dichotomous Image Segmentation (ECCV 2022).
Official code: https://github.com/xuebinqin/DIS.

U2Net + ISNet GT encoder， training base on ssim loss, iou loss and bce loss. Using weighted binary cross-entropy (BCE) loss enhances the capability to  extract foreground pixels. 

# Mutil loss
isnet.py
```
ssim_loss = SSIM(window_size=11,size_average=True)
iou_loss = IOU(size_average=True)
def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0
    
    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
            # ssim iou loss
            ssim_out = 1 - ssim_loss(preds[i],target)
            iou_out = iou_loss(preds[i],target)
            loss = loss + ssim_out + iou_out
        else:
            loss = loss + bce_loss(preds[i],target)
            # ssim iou loss
            ssim_out = 1 - ssim_loss(preds[i],target)
            iou_out = iou_loss(preds[i],target)
            loss = loss + ssim_out + iou_out
        if(i==0):
            loss0 = loss
    return loss0, loss
```
# Weighted BCE
isnet.py
```
def bce_loss_w(input, target):
    weight=torch.zeros_like(target)
    weight=torch.fill_(weight,0.3)
    weight[target>0]=0.7
    loss = nn.BCELoss(weight=weight, size_average=True)(input,target.float())
    return loss
```
# Data preparation
train_valid_inference_main.py
```
dataset_tr = {"name": "",
                 "im_dir": "../dataset/2dteeth/train/image/",
                 "gt_dir": "../dataset/2dteeth/train/mask/",
                 "im_ext": ".png",
                 "gt_ext": ".png",
                 "cache_dir":"../dataset/2dteeth/data_cache/"
                 }
dataset_vd = {"name": "",
             "im_dir": "../dataset/2dteeth/val/image/",
             "gt_dir": "../dataset/2dteeth/val/mask/",
             "im_ext": ".png",
             "gt_ext": ".png",
             "cache_dir":"../dataset/2dteeth/data_cache/"
             }
```
# train
Download pre-train model isnet-general-use.pth from https://github.com/xuebinqin/DIS.
```
python train_valid_inference_main.py
```

# Experimented on tooth segmentation on panoramic X-ray images.


# export onnx and test by onnxruntime
```
python torch2onnx.py

python demo_onnx.py
```

# tensorrt and c++
https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo
