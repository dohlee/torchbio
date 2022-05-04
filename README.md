# torchbio

PyTorch utilities for AI application in biology :dna:

## Examples

Get a sequence at genomic region as a one-hot encoded tensor.
```python
from torchbio import seq

encoder = seq.GenomeSeqEncoder('hg38.fa') # Give any fasta file.
x = encoder.encode_region('chr1', 12000000, 120001000) # x is a tensor with shape (4, 1000).
# Encoding scheme is as follows:
# A = [1, 0, 0, 0]
# C = [0, 1, 0, 0]
# G = [0, 0, 1, 0]
# T = [0, 0, 0, 1]
```

## Modules

**torchbio.seq**

