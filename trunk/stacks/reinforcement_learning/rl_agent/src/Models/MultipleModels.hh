#ifndef _MULTIPLEMODELS_HH_
#define _MULTIPLEMODELS_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>



/** Multiple Models */
class MultipleModels: public Model {

public:

  MultipleModels(int id, int nInput, int nOutput, int modelType, Random rng);

  virtual ~MultipleModels();





  virtual bool trainInstance(std::vector<float> input, 
			     std::vector<float> output);
  virtual std::vector<float> testInstance(std::vector<float> input);
  
  // helper functions
  void createModels();

private:

  const int id;
  const int nInput; 
  const int nOutput;
  const int type;
  
  Random rng;

  // MODELS
  std::vector<Model*> models;

};


#endif
  
