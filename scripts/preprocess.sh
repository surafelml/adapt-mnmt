#!/bin/bash

#
# learn and apply sentencepiece subword segmentation - generate vocabulary (src & tgt merge)
# ././scripts/preprocess.sh [exp-id]
#

EXPID=$1	# pt-en, gl-en, gl-en_progadapt, ptgl-en_proggrow	 
SPMSIZE=$2 	# 4000/single-pair/low-resource 8000/single-pair 16000/single-pair 32000/mnmt based on the model type
EXPDIR=$PWD
DATADIR=$EXPDIR/models/$EXPID/data
SRC='src'
TGT='tgt'

ONMT=$EXPDIR/OpenNMT/opennmt
GEN_VOCAB=$ONMT/bin/build_vocab.py
SPTED=$EXPDIR/scripts/sentencepiece.py
SHARED_VOCAB=false	# true

SPMDIR=$DATADIR/spmodel
SPDATA=$DATADIR/spdata


if [ -d $DATADIR ] && [ ! -d $SPMDIR ]; then
    mkdir $SPMDIR
    echo -e "\nLEARNING SP MODEL ..."

    python $SPTED --run "train" \
            --spm_dir $SPMDIR \
	    --in_file $DATADIR/train \
            --src $SRC --tgt $TGT \
            --spm_size $SPMSIZE
    wait $!
    echo "SP MODEL: [$SPMDIR]"
fi



if [ -d $SPMDIR ] && [ ! -d $SPDATA ]; then
    mkdir -p $SPDATA && cd $SPDATA    

    for SET in train dev test; do
        echo -e "\nAPPLYING SP MODEL ON [$SET] ..."
        python $SPTED --run "encode" \
                    --spm_dir $SPMDIR \
                    --in_file $DATADIR/${SET} \
		    --src $SRC --tgt $TGT \
		    --op_file $SPDATA/${SET}  
     done 
     echo "SP DATA: [ $SPMDIR ]"


    # generate vocab using opennmt
    if [ -f train.$SRC ] && [ -f train.$TGT ]; then
       echo -e "\nGENERATING VOCABULARY ..."

       if $SHARED_VOCAB; then
         cat train.$SRC train.$TGT > train.${SRC}${TGT}
         python $GEN_VOCAB --save_vocab vocab train.${SRC}${TGT}
         rm -rf train.${SRC}${TGT}
       else
         python $GEN_VOCAB --size $VOCABSIZE --save_vocab vocab.$SRC train.$SRC
         python $GEN_VOCAB --size $VOCABSIZE --save_vocab vocab.$TGT train.$TGT
       fi
    fi
fi
