#!/bin/bash

# 
# for library specific requiremtns, see README of each repo
#

EXPDIR=$PWD

# libraries
MOSES=https://github.com/moses-smt/mosesdecoder.git
SENT_PIECE='sentencepiece==0.1.8'
TENSORFLOW='tensorflow-gpu=1.4.1'
#OPENNMT=https://github.com/OpenNMT/OpenNMT-tf/tree/v1.15.0 # install updated version ./OpenNMT


# Data, Processing
if [ ! -d $EXPDIR/mosesdecoder ]; 
  echo "Cloning Mosesdecoder ..."
  git clone $MOSES
fi

echo "Installing SentencePiece ..."
pip install $SENT_PIECE


# NMT
echo "Install tensorflow.."
pip install $TENSORFLOW

if [ -d $EXPDIR/OpenNMT ]
  cd ./OpenNMT
  pip install -e ./ 
fi
