#!/bin/bash

#EXPATH=$1 	# ./models/pt-en
EXPID=$1	# pt-en
DEVICES=$2	# -1 for cpu 
export CUDA_VISIBLE_DEVICES=$DEVICES
EXPDIR=$PWD

ONMT=$EXPDIR/OpenNMT/opennmt
CONFIG=$EXPDIR/config.yml
MODEL_DEFN=$EXPDIR/model_defn.py	# update model capacity, based on single pair or multilingual model training

RUNDIR=$EXPDIR/models/$EXPID 		# dir where exp is run 
DATADIR=$RUNDIR/data/spdata 		# preprocessed data from ./scripts/preprocess.sh [exp-id] [subword-size]
LOG=$RUNDIR/log.train


if [ ! -d $RUNDIR/model ]; then

  mkdir -p $RUNDIR
  echo -e "\nTRAINING LOG: [ $LOG ]"
  
  python $ONMT/bin/main.py train_and_eval \
			--config $CONFIG \
			--model $MODEL_DEFN \
			--run_dir $RUNDIR 
			--data_dir $DATADIR \
			--seed 1234 \
			--num_gpus 2 > $LOG 2> $LOG

			#--auto_config \
			#--checkpoint_path parent_model \
else
  echo -e "\nModel exists: $RUNDIR/model "
fi
