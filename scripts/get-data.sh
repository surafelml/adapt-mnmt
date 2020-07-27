#!/bin/bash


EXPDIR=$PWD # experimental path  
READER=$EXPDIR/scripts/ted_reader.py

DATA=$EXPDIR/data
TEDDATA=$DATA/ted-data

if [ ! -d $TEDDATA ]; then
  echo "Loading and reading ted data ..."

  mkdir -p $DATA/ted-data
  pushd $DATA/ted-data

	  wget http://phontron.com/data/ted_talks.tar.gz
	  tar -xzvf ted_talks.tar.gz
	  rm -rf ted_talks.tar.gz

	  # for all ted-lang-ids [./scripts/ted_talks_langs.txt] extract [all langs-en] pairs
	  # https://github.com/neulab/word-embeddings-for-nmt/blob/master/ted_reader.py
	  python $READER

	  rm -rf *.tsv
  popd
fi
