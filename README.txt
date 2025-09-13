Thinking


alignments/* - raw msa alignments for each sample

coupling_models - Potts model parameters

processed_data/* \
- wt.fasta (original) \
- seqs.fasta (mutated) \
- seqs.txt (mutated w/ less formatting) \
- Data.csv (mutations) \
   - Sequences, fitness \

unirep_weights - Unirep model weights



TODO - embeddings
- Try differnet embedding models
- Try different kernels


TODO - Direct
- Kmers


# Note - (most gaussian) dataset:  UBE4B_MOUSE_Klevit2013-nscor_log2_ratio
print()
print("best pair: ", min(score_collection, key=lambda x: x[0]))
print("array: ", np.array([x[0] for x in score_collection]))
print("std: ", np.array([x[0] for x in score_collection]).std())


# Todo
# opt
# - ADkernel
# - Kmer Kernel
# - Hamming Kernel
# - Recreate Coupling model
# - Recreate Unirep
# - Some weights of EsmModel were not initialized from the model checkpoint at facebook/esm2_t6_8M_UR50D and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
