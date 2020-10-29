These are compressed models that we have pre-trained, for quick use in validation, and to demonstrate our results.

Note that the weights of these checkpoints might be larger in disk than the numbers we report in the paper. There are
two main reasons for this:

1. Overhead from storing dictionaries
2. Overhead from storing codes with values higher than 255 using `short` Tensors, which take 16 bits each. This happens
   when, for example, we store the codes for the fully connected layer of ResNet18/50 for classification since, by
   default, the codebook sizes are 1024 and 2048 by default. This could be improved by bit packing the weights of these
   layers, but would make pytorch loading more complicated.
