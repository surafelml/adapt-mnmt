#!/bin/bash

#
# scrip builds many-to-many direction by avoiding specific pairs as specified
#
# ./build-training-data.sh ['src1-en en-src1 src2-en en-src2'] [flag] [exp-id]
#

set -e


PAIRS=$1	# pairs for building training data
FLAG=$2		# if specified adds lang-id/flag on src - for multilingaul settings
EXPID=$3	# exp path in models/EXPID


EXPDIR=$PWD
DATADIR=$EXPDIR/data/ted-data
DSTDIR=$EXPDIR/models/$EXPID/data

# Merge all except pairs in SKIP - use for building multilingual data avoiding certain pair/s
#SKIP=' az tr be ru gl pt pt-br sk cs ' 	#9 langs skipped for mnmt_98
#SKIP=' az be gl sk ' 				#4 langs skipped for mnmt_108


MOSES=$PWD/mosesdecoder/scripts
NORM=$MOSES/tokenizer/normalize-punctuation.perl
TOK=$MOSES/tokenizer/tokenizer.perl
DETOK=$MOSES/tokenizer/detokenizer.perl
DEES=$MOSES//tokenizer/deescape-special-chars.perl


#--------------------------------------------------------------------------


COUNT=0
COUNTSKIP=0
if [ ! -d $DSTDIR -a -d $DATADIR ]; then 
  mkdir -p $DSTDIR
  pushd $DSTDIR
   mkdir ./test-sets
   echo -e "DSTDIR: $DSTDIR \nDATADIR: $DATADIR"

  for PAIR in $PAIRS; do #*; do
    SRC=`cut -d'-' -f1 <<< ${PAIR}`
    TGT=`cut -d'-' -f2 <<< ${PAIR}`

   #BUIILD FOR ALL PAIRS EXCEPT SKIP
   if [[ $SRC != $TGT ]] && [[ ! $SKIP =~ " $SRC " ]] && [[ ! $SKIP =~ " $TGT " ]]; then

    echo "BUILDING: $SRC>$TGT"
 
    if [ $SRC = 'en' ]; then
      DATA=$DATADIR/${TGT}_${SRC}
    else	
      DATA=$DATADIR/${SRC}_${TGT}
    fi


    COUNT=$((COUNT+1))
    for SET in train dev test; do

	RAWDATA=$DATA/ted-${SET}.orig

	if [ -n $FLAG ]; then 
          $NORM < ${RAWDATA}.$SRC | $DEES | $DETOK -l $SRC -q | awk -vtgt_tag="<2${TGT}>" '{ print tgt_tag" "$0 }' >> ${SET}.src	#$SRC 

          $NORM < ${RAWDATA}.$SRC | $DEES | $DETOK -l $SRC -q  >> ${SET}.tgt	

	else
          $NORM < ${RAWDATA}.$SRC | $DEES | $DETOK -l $SRC -q  >> ${SET}.src
          $NORM < ${RAWDATA}.$TGT | $DEES | $DETOK -l $TGT -q  >> ${SET}.tgt
	fi



	if [ $SET = 'test' ]; then # get test-sets for post training evaluation 
          if [ ! -f ./test-sets/${SET}.$SRC ]; then
            $NORM < ${RAWDATA}.$SRC | $DEES | $DETOK -l $SRC -q  > ./test-sets/${SET}.$SRC
	  fi

          if [ ! -f ./test-sets/${SET}.$TGT ]; then
            $NORM < ${RAWDATA}.$TGT | $DEES | $DETOK -l $TGT -q  > ./test-sets/${SET}.$TGT
	  fi
	fi
    done #PAIR

  else
    echo -e ">SKIPPING: ${SRC}-${TGT}"
    COUNTSKIP=$((COUNTSKIP+1))
  fi
done #PAIRS
popd

echo "Done: $DSTDIR"
wc -l $DSTDIR/*
echo "TOTAL SKIP: $COUNTSKIP"
echo "TOTAL ADDED: $COUNT"


else
  echo ">CHECK $DSTDIR OR $DATADIR"; exit 1
fi
