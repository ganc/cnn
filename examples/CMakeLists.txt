CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET poisson-regression tag-bilstm embed-cl encdec get_monolingual_dict xor xor-xent rnnlm-aevb rnnlm-args rnnlm nlm n_best_words parallel-encdec score_parallel_encdec score_rnnlm-args get_char_embeddings textcat rnnlm2)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  target_link_libraries(${TARGET} cnn ${LIBS})
  if (WITH_CUDA_BACKEND)
    add_dependencies(${TARGET} cnncuda)
    target_link_libraries(${TARGET} cnncuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  endif (WITH_CUDA_BACKEND)
endforeach()

