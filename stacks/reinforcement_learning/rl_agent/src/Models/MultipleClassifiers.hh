/** \file MultipleClassifiers.hh
    Defines the Multiple Classifiers class, which uses an ensemble of classifiers, such as a set of decision trees in a random forest.
    \author Todd Hester
*/

#ifndef _MULTCLASS_HH_
#define _MULTCLASS_HH_

#include "../Models/C45Tree.hh"
#include "../Models/M5Tree.hh"
#include "../Models/LinearSplitsTree.hh"

#include "../Models/Stump.hh"

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>
#include <set>
#include <map>

/** Multiple Classifiers class: uses an ensemble of classifiers, such as a set of decision trees in a random forest */
class MultipleClassifiers: public Classifier {

public:

  /** Default Constructor
      \param id identify the model
      \param modelType identifies which type of model to use
      \param predType identifies how to combine multiple models
      \param nModels # of models to use for ensemble models (i.e. random forests)
      \param trainMode build every step? only on errors? every freq steps?
      \param trainFreq frequency of model building if using latter mode
      \param featPct pct of features to remove from set used for each tree split
      \param expPct pct of experiences to give to each model
      \param treeThreshold determines the amount of error to be tolerated in the tree (prevents over-fitting with larger and larger trees)
      \param stoch if the domain is stochastic or deterministic
      \param rng Random Number Generator 
  */
  MultipleClassifiers(int id, int modelType, int predType, int nModels, 
                      int trainMode, int trainFreq,
                      float featPct, float expPct, float treeThreshold,
                      bool stoch, Random rng);

  /** Copy constructor */
  MultipleClassifiers(const MultipleClassifiers&);
  virtual MultipleClassifiers* getCopy();

  ~MultipleClassifiers();

  virtual bool trainInstances(std::vector<classPair> &instances);
  virtual bool trainInstance(classPair &instance);
  virtual void testInstance(const std::vector<float> &input, std::map<float, float>* retval);
  virtual float getConf(const std::vector<float> &s);
  
  /** Update measure of accuracy for model if we're using best model only */
  void updateModelAccuracy(int i, const std::vector<float> &input, float out);

  /** Initialize models */
  void initModels();

  /** Calculate kl divergence of the model's predicted probability distributions */
  float klDivergence(const std::vector<float> &input);

  /** Calculate the variance of the model's predictions of continuous values */
  float variance(const std::vector<float> &input);

  bool STDEBUG;
  bool PRED_DEBUG;
  bool ACC_DEBUG;
  bool CONF_DEBUG;
  bool COPYDEBUG;

  /** The ensemble of Classifier Models used. */
  std::vector<Classifier*> models;

private:

  const int id;
  const int modelType;
  const int predType;
  const int nModels;
  const int mode;
  const int freq;
  float featPct;
  float expPct;
  const float treeThresh;
  const bool stoch;
  const bool addNoise;

  Random rng;

  std::vector<float> accuracy;
  int nsteps;
  std::vector<std::map<float, float> >infos;

};


#endif
  
