#ifndef _STUMP_HH_
#define _STUMP_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>
#include <set>
#include <map>

#define N_STUMP_EXP 250000

/** C4.5 decision stump class */
class Stump: public Classifier {

public:

  // mode - re-build stump every step?  
  // re-build only on misclassifications? or rebuild every 'trainFreq' steps
  Stump(int id, int trainMode, int trainFreq, int m, float featPct, Random rng);

  Stump(const Stump&);
  virtual Stump* getCopy();

  ~Stump();

  // structs to be defined
  struct stump_experience;
      
  struct stump_experience {
    std::vector<float> input;
    float output;
    int id;
  };
  
  enum splitTypes{
    ONLY,
    CUT
  };

  bool trainInstances(std::vector<classPair> &instances);
  bool trainInstance(classPair &instance);
  void testInstance(const std::vector<float> &input, std::map<float, float>* retval);
  float getConf(const std::vector<float> &input);

  void buildStump();

  // helper functions
  void initStump();
  bool passTest(int dim, float val, int type, const std::vector<float> &input);
  float calcGainRatio(int dim, float val, int type,float I);
  float* sortOnDim(int dim);
  float calcIofP(float* P, int size);
  float calcIforSet(const std::vector<stump_experience*> &instances);
  void printStump();
  void testPossibleSplits(float *bestGainRatio, int *bestDim, 
			  float *bestVal, int *bestType);
  void implementSplit(float bestGainRatio, int bestDim,
		      float bestVal, int bestType);
  void compareSplits(float gainRatio, int dim, float val, int type, 
		     int *nties, float *bestGainRatio, int *bestDim, 
		     float *bestVal, int *bestType);
  //std::vector<float> calcChiSquare(std::vector<stump_experience*> instances);
  void outputProbabilities(std::multiset<float> outputs, std::map<float, float>* retval);
  int findMatching(const std::vector<stump_experience*> &instances, int dim, 
		   int val, int minConf);

  void setParams(float margin, float forestPct, float minRatio);

  bool ALLOW_ONLY_SPLITS;

  bool STDEBUG;
  bool SPLITDEBUG;
  int nExperiences;

  float SPLIT_MARGIN;
  float MIN_GAIN_RATIO; 
  float REBUILD_RATIO;
  float LOSS_MARGIN;

private:

  const int id;
  
  const int mode;
  const int freq;
  const int M;
  float featPct;

  Random rng;

  int nOutput;
  int nnodes;

  // INSTANCES
  std::vector<stump_experience*> experiences;
  stump_experience allExp[N_STUMP_EXP];

  // split criterion
  int dim;
  float val;
  int type;
  float gainRatio;
  
  // set of all outputs seen at this leaf/node
  std::multiset<float> leftOutputs;
  std::multiset<float> rightOutputs;



};


#endif
  
