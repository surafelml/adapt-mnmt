# Dynamic Transfer Learning for Low-Resource Neural Machine Translation

__Updates__

[July, 2020] Updated repo with scripts and notes on experimental settings

---


This repo implements the following papers and associated features based on [OpenNMT-tf1.15](https://github.com/OpenNMT/OpenNMT-tf/tree/v1.15.0): 


[Transfer Learning in Multilingual Neural Machine Translation](https://arxiv.org/pdf/1811.01137.pdf)

[Adapting Multilingual Neural Machine Translation to Unseen Languages](https://arxiv.org/pdf/1910.13998.pdf)



## Experimental Settings 

--- 

#### Requirements 

- [Mosesdecoder](https://github.com/moses-smt/mosesdecoder)

- [SentenciePiece](https://github.com/google/sentencepiece)

- See/run: `./setup-env.sh`



### Data

Experiments utilize the Ted Talks data, for its low-resource nature (ranging from ~5k to ~200k parallel examples) for more than 50 languages paired with English, from [Qi et al](https://www.aclweb.org/anthology/N18-2084/).

`./scripts/get-data.sh` 


### Preprocessing 
Prepare data for `src/s - tgt/s` pair/s (if flag is specified, tgt-lang-id is appended on the src side): 

`./scripts/build-training-data.sh ['src1-en en-src1 src2-en en-src2'] [flag] [exp-id]`


Preprocess (clean, detokenize, and subword segmentation with sentencepiece):

`./scripts/preprocess.sh [exp-id] [subword-size]`



### Pre-Training Parent Model

Train a parent model, that exhibits a relatively high-resource data (e.g. Portuguese-English / `Pt-En`).

`./train.sh [exp-id] [gpu-device]`


<!--As noted in the summary above, the dynamic transfer-learning differs from previous approaches by tailoring to the target low-resource languages, specifically the vocabulary and embeddings, with two proposals: 
-->


### Progressive Adaptation (ProgAdapt) to New Translation Directions

---

Steps for ProgAdapt of the parent model `Pt-En` to child low-resource pair Galician-English / `Gl-En`.


__Data__

`./scripts/build-training-data.sh 'gl-en' [child-model_exp-id]`

<!-- Optionally to preprocess multiple directions
`./scripts/build-training-data.sh 'gl-en en-gl' flag [child-model_exp-id]`
-->


__Data Preprocessing__

`./scripts/preprocess.sh [child-model_exp-id] [subword-size]`

<!--
Tip: since the dynamic transfer-learning allows to vary the `subword-size` of the child from the parent, it is adviced to try different/optimal subword sizes. 
-->


__ProgAdapt Training__

Training first customizes the parent model by taking in to consideration the child model (`Gl-En`) newly generated vocabulary:


`./train-dynamic-tl.sh [parent-model_exp-id] [child-model_exp-id] [gpu-device]`



### Progressive Growth (ProgGrow) with New Translation Directions

---

ProgGrow differs from progAdapt by incorporating the `Pt-En` parent model translation direction, while learning the new low-resource pair `Gl-En` (child model) direction.


__Data__

`./scripts/build-training-data.sh 'pt-en gl-en' flag [child-model_exp-id]`


__Data Preprocessing__

`./scripts/preprocess.sh [child-model_exp-id] [subword-size]` 


__ProgGrow Training__

`./train-dynamic-tl.sh [parent-model_exp-id] [child-model_exp-id] [gpu-device]`



### More Options 

---

At time of transfer-learning you can optionally: 

- Load specific components of the parent model. See `load_weights` in config_adapt.yml for more options: 

`['encoder', 'decoder', 'shared_embeddings', 'src_embs', 'tgt_embs', 'optim', 'projection']`. 


- Freeze sub-networks (i.e. selectively optimize the encoder or decoder). See `freeze` in config_adapt.yml for options. 


- In addition to `encoder` and/or `decoder` only customization, you can pre-train a parent model with an `encoder-decoder` shared vocab and customize for the child model. See `--shared_vocab` and `--new_shared_vocab` options in ./train-dynamic-tl.sh. 


__Note__: to replicate the experiments reported in our work, please see further details in the experimental section of each paper.



### References 
---

```bibtext 
@article{lakew2018transfer,
  title={Transfer learning in multilingual neural machine translation with dynamic vocabulary},
  author={Lakew, Surafel M and Erofeeva, Aliia and Negri, Matteo and Federico, Marcello and Turchi, Marco},
  journal={arXiv preprint arXiv:1811.01137},
  year={2018}
}

@article{lakew2019adapting,
  title={Adapting Multilingual Neural Machine Translation to Unseen Languages},
  author={Lakew, Surafel M and Karakanta, Alina and Federico, Marcello and Negri, Matteo and Turchi, Marco},
  journal={arXiv preprint arXiv:1910.13998},
  year={2019}
}
```