#!/bin/bash

PARENT_EXPID=$1 # 'pt-en'
EXPID=$2	# 'gl-en_progadapt'
DEVICES=$3	# -1 for cpu 
export CUDA_VISIBLE_DEVICES=$DEVICES
EXPDIR=$PWD

ONMT=$EXPDIR/OpenNMT/opennmt
CONFIG=$EXPDIR/config_adapt.yml		# update accordingly for different params
MODEL_DEFN=$EXPDIR/model_defn.py	# update model capacity, if training single pair or multilingual

RUNDIR=$EXPDIR/models/$EXPID 		# dir where exp is run 
DATADIR=$RUNDIR/data/spdata 		# preprocessed data 
LOG=$RUNDIR/log.train.progadapt

# parent model 
PARENTDIR=$EXPDIR/models/$PARENT_EXPID
PARENTMODEL=$PARENTDIR/model
PSRCVOCAB=$PARENTDIR/data/spdata/vocab.src
PTGTVOCAB=$PARENTDIR/data/spdata/vocab.tgt
#SHRDVOCAB=$PARENTDIR/data/spdata/vocab		# only for shared vocab

# child model
CHILDDIR=$RUNDIR/model				# progadapt model dir
SRCVOCAB=$RUNDIR/data/spdata/vocab.src
TGTVOCAB=$RUNDIR/data/spdata/vocab.tgt
#VOCAB=$RUNDIR/data/spdata/vocab		# only for shared vocab


# update vocab setting 
MODE="replace"	# repalce, merge
INIT="zeros"	# zeros, random


if [ -d $PARENTMODEL -a ! -d $CHILDDIR  ]; then

  echo -e "\nTRAINING PROGADAPT - TRANSFER PARENT TO CHILD MODEL: [$LOG]" 
  python $ONMT/bin/main.py update_vocab --config $CONFIG --model $MODEL_DEFN \
					--checkpoint_path $PARENTMODEL  \
					--output_dir $CHILDDIR \
					--src_vocab $PSRCVOCAB \
					--new_src_vocab $SRCVOCAB \
					--tgt_vocab $PTGTVOCAB \
					--new_tgt_vocab $TGTVOCAB \
					--mode $MODE --init $INIT > $LOG 2> $LOG
fi


if [ -d $CHILDDIR ]; then

  echo -e "\nTRAINING PROGADAPT - TRAIN CHILD MODEL: [$LOG]" 
  python $ONMT/bin/main.py train --model $MODEL_DEFN \
			--run_dir $RUNDIR --seed 1234 \
			--data_dir $DATADIR \
			--checkpoint_path $CHILDDIR \
			--config $CONFIG \
			--num_gpus 1 >> $LOG 2>> $LOG

			#--auto_config \
else
  echo -e "\nCHILD MODEL MISSING : $CHILDDIR "
fi
