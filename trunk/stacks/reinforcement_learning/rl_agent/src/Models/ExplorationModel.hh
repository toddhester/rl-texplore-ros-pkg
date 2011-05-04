/** \file ExplorationModel.hh
    Defines the ExplorationModel class.
    Reward bonuses based on the variance in model predictions are described in: Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#ifndef _EXPLOREMODEL_HH_
#define _EXPLOREMODEL_HH_

#include "../Common/Random.h"
#include "../Common/core.hh"
#include <vector>
#include <map>
#include <set>


/** This model wraps an another model and adds reward
 bonuses based on model confidence, # of visits, or other metrics. */
class ExplorationModel: public MDPModel {

public:

  /** Default contstructor
      \param model The underlying MDP Model being used.
      \param modelType the type of model being used.
      \param exploreType type of reward bonuses to be added on top of model
      \param predType the way in which ensemble models combine their models
      \param nModels # of models to use for ensemble models (i.e. random forests)
      \param m # of visits for a given state-action to be considered known
      \param numactions # of actions in the domain
      \param rmax maximum one-step reward in the domain
      \param qmax maximum possible q-value in a domain
      \param rrange range of one-step rewards in the domain
      \param nfactors # of state features in the domain
      \param b coefficient used to determine how much reward to add to models
      \param featmax the maximum value of each state feature
      \param featmin the minimum value of each state feature
      \param rng Random Number Generator
  */
  ExplorationModel(MDPModel * model, int modelType, 
                   int exploreType, int predType, int nModels,
                   float m, int numactions, float rmax, 
                   float qmax, float rrange, int nfactors, float b, 
                   const std::vector<float> &featmax, 
                   const std::vector<float> &featmin, 
                   Random rng); 

  /** Copy constructor */
  ExplorationModel(const ExplorationModel&);

  virtual ~ExplorationModel();
  virtual ExplorationModel* getCopy();

  virtual bool updateWithExperiences(std::vector<experience> &instances);
  virtual bool updateWithExperience(experience &e);
  virtual bool getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval);

  /** Add state to a set of visited states */
  void addStateToSet(const std::vector<float> &s);

  /** Check if the given state is in the set of visited states */
  bool checkForState(const std::vector<float> &s);


  bool MODEL_DEBUG;

protected:



private:
  
  /** Set of all distinct sensations seen. 
      This way we can know what we've visited. */
  std::set<std::vector<float> > statespace;

  /** Underlying MDP model that we've wrapped and that we add bonus rewards onto. */
  MDPModel* model;

  std::vector<float> featmax;
  std::vector<float> featmin;

  int modelType; 
  int exploreType; 
  int predType;
  int nModels;
  float M; 
  int numactions; float rmax; float qmax; float rrange;
  int nfactors; 
  const float b;

  Random rng;
  
};



#endif
