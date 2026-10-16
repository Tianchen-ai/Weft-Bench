#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

struct ProjectionReference {
  ggml_context *context;
  ggml_backend_t backend;
  ggml_backend_buffer_t buffer;
  ggml_cgraph *graph;
  ggml_backend_graph_plan_t plan;
  ggml_threadpool_t pool;
  ggml_tensor *result;
  std::vector<float> &output;
  ProjectionReference(ggml_type type,size_t m,size_t n,size_t k,const void *weights,const float *activation,std::vector<float> &out):output(out) {
    ggml_cpu_init();context=ggml_init({1024*1024,nullptr,true});
    auto *w=ggml_new_tensor_2d(context,type,k,n);
    auto *x=ggml_new_tensor_2d(context,GGML_TYPE_F32,k,m);
    result=ggml_mul_mat(context,w,x);graph=ggml_new_graph(context);ggml_build_forward_expand(graph,result);
    backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,1);
    auto pool_params=ggml_threadpool_params_default(1);
    pool=ggml_threadpool_new(&pool_params);ggml_backend_cpu_set_threadpool(backend,pool);
    buffer=ggml_backend_alloc_ctx_tensors(context,backend);
    if(!buffer)throw std::runtime_error("GGML tensor allocation failed");
    ggml_backend_tensor_set(w,weights,0,ggml_nbytes(w));
    ggml_backend_tensor_set(x,activation,0,ggml_nbytes(x));
    plan=ggml_backend_graph_plan_create(backend,graph);
    if(!plan)throw std::runtime_error("GGML graph plan creation failed");
  }
  void run(){
    if(ggml_backend_graph_plan_compute(backend,plan)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML projection failed");
  }
  void collect(){ggml_backend_tensor_get(result,output.data(),0,ggml_nbytes(result));}
  ~ProjectionReference(){ggml_backend_graph_plan_free(backend,plan);ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_threadpool_free(pool);ggml_free(context);}
};
