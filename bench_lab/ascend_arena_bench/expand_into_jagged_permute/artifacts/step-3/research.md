# Step 3 Research: expand_into_jagged_permute

## Sources
- FBGEMM GPU sparse_ops docs describe expand_into_jagged_permute for sparse/jagged feature permutation.
- FBGEMM JaggedTensorOps docs describe jagged tensor offsets as core sparse feature representation.
- TorchRec docs/GitHub describe KeyedJaggedTensor and sparse recommendation model features.

## Mapped Shapes
- permute length maps to sparse feature/table count P.
- input_offsets/output_offsets length is P+1.
- output_size maps to expanded jagged positions / batch-feature bag length.
- Common recommendation anchors: 26, 32, 64, 80, 96, 128 feature tables and output_size around 4096 for this benchmark contract.

## Notes
Current proto enum caps output_size at 4101, so profile cases use table-count diversity and legal output_size values rather than unbounded production-scale bag lengths.
