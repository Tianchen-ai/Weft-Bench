#pragma once
#include "timer.h"
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <functional>

namespace wb {
inline std::string json(const std::string &s) {
  std::string result="\"";
  for(unsigned char c:s) {
    if(c=='"'||c=='\\') {result+='\\';result+=c;}
    else if(c=='\n')result+="\\n";
    else if(c<32){char buf[7];std::snprintf(buf,sizeof(buf),"\\u%04x",c);result+=buf;}
    else result+=c;
  }
  return result+'"';
}
struct InputSet {
  uint32_t seed;
  std::map<std::string,std::string> files;
  std::map<std::string,double> parameters;
  std::set<std::string> consumed;
  explicit InputSet(uint32_t s):seed(s){}
  float random() {seed=seed*1664525U+1013904223U;return (float(seed>>8)/8388608.0f-1)*.5f;}
  template<class T> void fill(std::vector<T>& v) {for(auto &x:v)x=T(random());}
  template<class T> bool load(const std::string &name,std::vector<T>& v) {
    auto item=files.find(name);if(item==files.end())return false;
    std::ifstream f(item->second,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streampos(v.size()*sizeof(T)))throw std::runtime_error("input byte size mismatch: "+name);
    f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));
    if(!f)throw std::runtime_error("cannot read input: "+name);
    consumed.insert(name);return true;
  }
  float parameter(const std::string &name,float fallback) const {
    auto it=parameters.find(name);return it==parameters.end()?fallback:float(it->second);
  }
  void finish() const {for(auto &p:files)if(!consumed.count(p.first))throw std::runtime_error("unknown input role: "+p.first);}
};
inline void close(float actual,double expected,size_t index,double atol=1e-4,double rtol=2e-3) {
  if(!std::isfinite(actual)||!std::isfinite(expected)||std::abs(actual-expected)>atol+rtol*std::abs(expected))
    throw std::runtime_error("numeric mismatch at output "+std::to_string(index)+": "+std::to_string(actual)+" vs "+std::to_string(expected));
}
template<class Run,class Collect,class Reset,class Check,class T>
void execute(const char* name,Run run,Collect collect,Reset reset,Check check,std::vector<T>& output,bool stateful,bool save,const char* correctness) {
  bool exported=false;
  try {
    reset();run();collect();
    if(save) {
      std::ofstream f(std::string(name)+"-Y.bin",std::ios::binary);
      f.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(T));
      if(!f)throw std::runtime_error("cannot save output");
      exported=true;
    }
    check();
    measure(run,reset,correctness,stateful,name);
  } catch(const std::exception &e) {
    std::printf("WB_RESULT {\"implementation\":%s,\"status\":\"check_failed\",\"error\":%s,\"outputs_available\":%s,\"samples_ms\":[]}\n",json(name).c_str(),json(e.what()).c_str(),exported?"true":"false");
  }
}
}
