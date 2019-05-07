VOC_ROOT=/media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012/
WEIGHTS=ilsvrc-cls_rna-a1_cls1000_ep-0001.params
CVD=0,1,2,3
BS=16
SN=austins_run_0506

echo "BEGIN"
echo "UNIQUESTRING 1"
date
: '
# 1. Train a classification network to get CAMs.
CUDA_VISIBLE_DEVICES=${CVD} python3 train_cls.py --lr 0.1 --num_workers 14 --batch_size ${BS} --max_epoches 15 --session_name ${SN}_cls --crop_size 448 --network network.resnet38_cls --voc12_root ${VOC_ROOT} --weights ${WEIGHTS} --wt_dec 5e-4

echo "UNIQUESTRING 2"
date
# 2. Generate labels for AffinityNet by applying dCRF on CAMs.
rm out_cam/* 
rm out_la_crf/* 
rm out_hat_crf/*
CUDA_VISIBLE_DEVICES=${CVD} python3 infer_cls.py --infer_list voc12/train_aug.txt --voc12_root ${VOC_ROOT} --network network.resnet38_cls --weights ${SN}_cls.pth --out_cam out_cam --out_la_crf out_la_crf --out_ha_crf out_ha_crf
'

echo "UNIQUESTRING Opt"
date
# (Optional) Check the accuracy of CAMs.
rm out_cam_pred/*
CUDA_VISIBLE_DEVICES=${CVD} python3 infer_cls.py --infer_list voc12/val.txt --voc12_root ${VOC_ROOT} --network network.resnet38_cls --weights ${SN}_cls.pth --out_cam_pred out_cam_pred
date
python3 compute_miou.py --pred_path out_cam_pred

:'
echo "UNIQUESTRING 3"
date
# 3. Train AffinityNet with the labels
CUDA_VISIBLE_DEVICES=${CVD} python3 train_aff.py --lr 0.1 --batch_size ${BS} --session_name ${SN}_aff  --max_epoches 8 --crop_size 448 --voc12_root ${VOC_ROOT} --network network.resnet38_aff --weights ${WEIGHTS} --wt_dec 5e-4 --la_crf_dir out_la_crf --ha_crf_dir out_ha_crf
date

echo "UNIQUESTRING 4 -- train, not val this time"
# 4. Perform Random Walks on CAMs
CUDA_VISIBLE_DEVICES=${CVD} python3 infer_aff.py --infer_list voc12/train.txt --voc12_root ${VOC_ROOT} --network network.resnet38_aff --weights  ${SN}_aff.pth --cam_dir out_cam --out_rw out_rw
date

# 5. Evaluate RW CAMs
python3 compute_miou.py --pred_path out_rw
date

'
echo "END"
