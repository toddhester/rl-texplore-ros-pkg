/** \file FactoredModel.hh
    Defines the FactoredModel class
    Please cite: Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#ifndef _FACTOREDMODEL_HH_
#define _FACTOREDMODEL_HH_

#include "../Models/C45Tree.hh"
#include "../Models/M5Tree.hh"
#include "../Models/LinearSplitsTree.hh"
#include "../Models/Stump.hh"
#include "../Models/MultipleClassifiers.hh"
#include "../Models/SepPlanExplore.hh"

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>


/** Builds an mdp model consisting of a tree (or ensemble of trees) to predict each feature, reward, and termination probability. Thus forming a complete model of the MDP. */
class FactoredModel: public MDPModel {
public:

  /** Default constructor
      \param id identify the model
      \param numactions # of actions in the domain
      \param M # of visits for a given state-action to be considered known
      \param modelType identifies which type of model to use
      \param predType identifies how to combine multiple models
      \param nModels # of models to use for ensemble models (i.e. random forests)
      \param treeThreshold determines the amount of error to be tolerated in the tree (prevents over-fitting with larger and larger trees)
      \param featRange range of each feature in the domain
      \param rRange range of reward values in domain
      \param needConf do we need confidence measures?
      \param dep assume dependent transitions between features (or indep)
      \param relTrans model relative transitions of features (or absolute)
      \param featPct pct of features to remove from set used for each tree split
      \param stoch if the domain is stochastic or deterministic
      \param episodic if the domain is episodic
      \param rng Random Number Generator 
  */
  FactoredModel(int id, int numactions, int M, int modelType, 
          int predType, int nModels, float treeThreshold,
          const std::vector<float> &featRange, float rRange,
          bool needConf, bool dep, bool relTrans, float featPct, 
	  bool stoch, bool episodic, Random rng = Random());

  /** Copy Constructor for MDP Tree */
  FactoredModel(const FactoredModel &);

  virtual ~FactoredModel();

  virtual bool updateWithExperiences(std::vector<experience> &instances);
  virtual bool updateWithExperience(experience &e);

  /** Initialize the MDP model with the given # of state features */
  bool initMDPModel(int nfactors);
  virtual float getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval);
  virtual FactoredModel* getCopy();

  /** Method to get a single sample of the predicted next state for the given state-action, rather than the full distribution given by getStateActionInfo */
  float getSingleSAInfo(const std::vector<float> &state, int act, StateActionInfo* retval);

  /** Combines predictions for each separate state feature into probabilities of the overall state vector */
  void addFactorProb(float* probs, std::vector<float>* next, std::vector<float> x, StateActionInfo* retval, int index, std::vector< std::map<float,float> > predictions, float* confSum);

  /** Set some parameters of the subtrees */
  void setTreeParams(float margin, float forestPct, float minRatio);

  /** Helper function to add two vectors together */
  std::vector<float> addVec(const std::vector<float> &a, const std::vector<float> &b);

  /** Helper function to subtract two vectors */
  std::vector<float> subVec(const std::vector<float> &a, const std::vector<float> &b);
  
private:
  
  /** Classifier to predict each feature */
  std::vector<Classifier*> outputModels;

  /** Classifier to predict reward */
  Classifier* rewardModel;

  /** Classifier to prediction termination probability */
  Classifier* terminalModel;

  int id;
  int nfactors;
  const int nact;
  const int M;
  const int modelType;
  const int predType;
  const int nModels;
  const int treeBuildType;
  const float treeThresh;

  const std::vector<float> featRange;
  const float rRange;

  const bool needConf;
  const bool dep;
  const bool relTrans;
  const float FEAT_PCT;
  const bool stoch;
  const bool episodic;
  Random rng;
  
  float EXP_PCT;

  bool MODEL_DEBUG;
  bool COPYDEBUG;

};



#endif
