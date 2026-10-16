#pragma once
#include "quantized_inputs.h"

struct ProjectionInputs {
  size_t M=WB_M,N=WB_N,K=WB_K,output_size=M*N;
  ggml_type weight_type=WB_WEIGHT_TYPE,activation_type=WB_ACTIVATION_TYPE;
  size_t weight_bytes,activation_bytes;
  std::vector<uint8_t> weights;
  std::vector<float> activation,expected;
  QuantizedInputs tables;
  ProjectionInputs(wb::InputSet &inputs) {
    ggml_cpu_init();
    if(K%ggml_blck_size(weight_type)||K%ggml_blck_size(activation_type))throw std::runtime_error("projection K must contain whole quantization blocks");
    weight_bytes=ggml_row_size(weight_type,K);activation_bytes=ggml_row_size(activation_type,K);
    weights.resize(N*weight_bytes);activation.resize(M*K);expected.resize(output_size);
    inputs.fill(activation);inputs.load("X",activation);
    std::vector<float> w(K),importance(K,1.0f);
    for(size_t n=0;n<N;++n){inputs.fill(w);ggml_quantize_chunk(weight_type,w.data(),weights.data()+n*weight_bytes,0,1,K,importance.data());}
    inputs.load("W",weights);inputs.finish();tables.prepare_tables();
    std::vector<uint8_t> quantized(M*activation_bytes);std::vector<float> decoded(M*K);
    for(size_t m=0;m<M;++m){wb_quantize_reference(activation_type,activation.data()+m*K,quantized.data()+m*activation_bytes,K);
      wb_dequantize_reference(activation_type,quantized.data()+m*activation_bytes,decoded.data()+m*K,K);}
    // Independent complete-column checks for decode; a bounded sample for prefill.
    size_t samples=M==1?N:std::min<size_t>(64,output_size);
    for(size_t s=0;s<samples;++s){size_t i=samples==1?0:s*(output_size-1)/(samples-1),m=i/N,n=i%N;
      wb_dequantize_reference(weight_type,weights.data()+n*weight_bytes,w.data(),K);double sum=0;
      for(size_t k=0;k<K;++k)sum+=double(w[k])*decoded[m*K+k];expected[i]=float(sum);}
  }
  void check(const std::vector<float> &y) const {
    for(auto value:y)if(!std::isfinite(value))throw std::runtime_error("numeric nonfinite projection");
    size_t samples=M==1?N:std::min<size_t>(64,output_size);
    for(size_t s=0;s<samples;++s){size_t i=samples==1?0:s*(output_size-1)/(samples-1);wb::close(y[i],expected[i],i);}
  }
};
