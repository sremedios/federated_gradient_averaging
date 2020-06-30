MODE=${1}
SITE=${2}
GPU=${3}

DATADIR="/nfs/share5/remedis/data/CQS_TBI/manually_segmented/train/preprocessed"

if [ ${MODE} = "local" ]
then
    PORT="0"
else
    PORT=${4}
fi
 
python train_ct.py ${SITE} ${MODE} ${GPU} ${DATADIR} ${PORT}
