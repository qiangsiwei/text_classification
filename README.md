# Text Classification

[![Build Status](https://travis-ci.org/qiangsiwei/text_classification.svg?branch=master)](https://travis-ci.org/qiangsiwei/text_classification)

keras implementation of text classification algorithms

Models:
=====

1. MLP

2. CNN

3. RNN

4. BiRNN

5. RCNN

6. HAN

7. CLSTM (series)

8. CLSTM (parallel)

9. TextCNN

10. FastText

Install:
=====

```python
python setup.py install
```

Usage:
=====

```python
from text_classification import *

clf = TextClassifierFastText()
clf.fit(x,y,epochs=epochs,validation_split=validation_split)
clf.predict(x)
```

Limitations:
----

1. sentence length is limit by 'maxlen', words beyond will be truncated

2. out-off-bag words in prediction will be ignored by keras Tokenizer 
