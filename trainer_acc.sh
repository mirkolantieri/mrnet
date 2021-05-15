DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="exp_acc"

EPOCHS=20
PREFIX="mrnet_acc"

python train_acc.py -t acl -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -t acl -p coronal --prefix_name $PREFIX --epochs=$EPOCHS  --experiment $EXPERIMENT
python train_acc.py -t acl -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT

python train_acc.py -t meniscus -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -t meniscus -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -t meniscus -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT

python train_acc.py -t abnormal -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -t abnormal -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -t abnormal -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT


