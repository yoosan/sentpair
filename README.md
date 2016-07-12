#Modeling Sentence Pairs with Tree-structured Attentive Encoder

The implementation of this paper. It runs both training and evaluation. Note that we just test the projects on Mac OS X and Ubuntu 14.

##Preparing

You can download the preprocessed data (recommend) by running

``./download_data``

alternatively you can process them by youself.
##Requirement
Software requirement.

+ ruby
+ torch7
+ python

The torch package shoud be installed are the following. For example, you can install the nn package by ``luarock install nn``. 

+ nn
+ nngraph
+ optim
+ xlua
+ sys
+ lfs

##Running

To run our models, you can tap the command 

``
th main.lua -<opt_name> opt_value -<opt_name> opt_val ...
``

For example, to run the SICK dataset, you should tap

``
th main.lua -task MSRP -structure atreelstm -lr 0.05 -n_epoches 10
``

more details in the file ``main.lua``.

##License
MIT