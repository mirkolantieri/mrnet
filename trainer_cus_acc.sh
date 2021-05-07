DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="exp_cus_acc"

EPOCHS=20
PREFIX="mrnet_cus_acc"

python train_acc.py -m=2 -t acl -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -m=2 -t acl -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -m=2 -t acl -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT

python train_acc.py -m=2 -t meniscus -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -m=2 -t meniscus -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -m=2 -t meniscus -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT

python train_acc.py -m=2 -t abnormal -p sagittal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -m=2 -t abnormal -p coronal --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
python train_acc.py -m=2 -t abnormal -p axial --prefix_name $PREFIX --epochs=$EPOCHS --experiment $EXPERIMENT
