#pragma once
#include "worker.h"
#include "ime_contract.h"
#include "spacemit/ime_kernels.h"

struct IMEInputs {
  size_t M=WB_M,N=WB_N,K=WB_K,output_size=M*N;
  std::vector<uint8_t> weights,activation;
  std::vector<float> expected;
  IMEInputs(wb::InputSet &inputs) {
    using namespace ime1_repro;
    kN=N;kK=K;
    if((WB_TILE_ROWS!=1&&WB_TILE_ROWS!=4)||M%WB_TILE_ROWS||N%16||K%256)throw std::runtime_error("IME input requires complete row tiles/N16/K256 multiples");
    ggml_context *ctx=ggml_init({1024*1024,nullptr,true});
    auto selected=*find_case(WB_AFFINE?"q4_K_prefill":"q4_0_prefill");selected.m=M;
    size_t bytes=WB_AFFINE?304:288,offset=WB_AFFINE?48:32;
    weights.resize((N/16)*(K/32)*bytes);activation.resize(M*(K/32)*36);expected.resize(M*N);
    std::vector<float> w(N*K),x(M*K),row(K);
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%17)-8)/9.0f;
    for(size_t n=0;n<N;++n)for(size_t k=0;k<K;++k)w[n*K+k]=float(int((n*7+k*13)%61)-30)/float(8+n%3);
    if(inputs.seed!=410){inputs.fill(x);inputs.fill(w);}
    inputs.load("W_f32",w);inputs.load("X_f32",x);inputs.finish();
    std::vector<uint8_t> plain(ggml_row_size(selected.weight_type,K));
    for(size_t n=0;n<N;++n){
      selected.quantize_weight(w.data()+n*K,plain.data(),K);auto blocks=reference_weights(selected,plain);
      for(size_t b=0;b<K/32;++b){auto *record=weights.data()+((n/16)*(K/32)+b)*bytes;
        auto scale=ggml_fp32_to_fp16(blocks[b].scale);std::memcpy(record+(n%16)*2,&scale,2);
        if(WB_AFFINE)record[32+n%16]=blocks[b].zero;
        for(size_t k=0;k<32;++k)record[offset+(k/16)*128+(n%16)*8+k%8]|=uint8_t(blocks[b].q[k]+blocks[b].zero)<<(((k/8)%2)*4);
      }
      auto values=reference_output(selected,blocks,x);for(size_t m=0;m<M;++m)expected[m*N+n]=values[m];
    }
    for(size_t m=0;m<M;m+=WB_TILE_ROWS){
      if(WB_TILE_ROWS==1)spacemit_kernels::ime1::quantize_a_row_i8(32,x.data()+m*K,K,activation.data()+m*(K/32)*36);
      else spacemit_kernels::ime1::quantize_a_4row_i8(32,x.data()+m*K,K,activation.data()+m*(K/32)*36);
    }
    ggml_free(ctx);
  }
  void check(const std::vector<float>& y) const {for(size_t i=0;i<y.size();++i)wb::close(y[i],expected[i],i);}
};
