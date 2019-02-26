VOC_ROOT=/home/austin/datasets/VOC/VOCdevkit/VOC2012/
WEIGHTS=ilsvrc-cls_rna-a1_cls1000_ep-0001.params
echo "Not confident where each weights should be used"


echo "BEGIN"
# python3 train_cls.py --lr 0.1 --num_workers=14 --batch_size 16 --max_epoches 15 --crop_size 448 --network network.resnet38_cls --voc12_root ${VOC_ROOT} --weights ${WEIGHTS} --wt_dec 5e-4

echo "UNIQUESTRING 1"
python3 infer_cls.py --infer_list voc12/train_aug.txt --voc12_root ${VOC_ROOT} --network network.resnet38_cls --weights vgg_cls.pth --out_cam out_cam --out_la_crf out_la_crf --out_ha_crf out_ha_crf

echo "END"

#echo "UNIQUESTRING 2"
#python3 infer_cls.py --infer_list voc12/val.txt --voc12_root ${VOC_ROOT} --network network.resnet38_cls --weights res38_cls.pth --out_cam_pred out_cam_pred
#
#echo "UNIQUESTRING 3"
#python3 train_aff.py --lr 0.1 --batch_size 8 --max_epoches 8 --crop_size 448 --voc12_root ${VOC_ROOT} --network network.resnet38_aff --weights ${WEIGHTS} --wt_dec 5e-4 --la_crf_dir la_crf_dir --ha_crf_dir ha_crf_dir
#
#
#echo "UNIQUESTRING 4 -- train, not val this time"
#python3 infer_aff.py --infer_list voc12/train.txt --voc12_root ${VOC_ROOT} --network network.resnet38_aff --weights  ${WEIGHTS}--cam_dir cam_dir --out_rw out_rw
#
#echo "END"
