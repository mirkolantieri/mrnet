DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="exp_auc"

EPOCHS=20
PREFIX="mrnet_auc"

python train_auc.py -t acl -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_auc.py -t acl -p coronal --prefix_name $PREFIX --epochs=$EPOCHS  --experiment $EXPERIMENT
python train_auc.py -t acl -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT

python train_auc.py -t meniscus -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_auc.py -t meniscus -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_auc.py -t meniscus -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT

python train_auc.py -t abnormal -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_auc.py -t abnormal -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_auc.py -t abnormal -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT


