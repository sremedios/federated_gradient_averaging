MODE=${1}
SITE=${2}
GPU=${3}

if [ ${SITE} = "A" ]
then
    DATADIR="/nfs/share5/remedis/data/CQS_TBI/manually_segmented/train_a/preprocessed"
elif [ ${SITE} = "B" ]
then
    DATADIR="/nfs/share5/remedis/data/CQS_TBI/manually_segmented/train_b/preprocessed"
else
    DATADIR="/nfs/share5/remedis/data/CQS_TBI/manually_segmented/train/preprocessed"
fi

if [ ${MODE} = "local" ]
then
    PORT="0"
else
    PORT=${4}
fi
 
python train_ct.py ${SITE} ${MODE} ${GPU} ${DATADIR} ${PORT}
