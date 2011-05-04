/** \file ExplorationModel.cc
    Implements the ExplorationModel class.
    Reward bonuses based on the variance in model predictions are described in: Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#include "ExplorationModel.hh"




ExplorationModel::ExplorationModel(MDPModel* innermodel, int modelType, int exploreType,
                                   int predType, int nModels,
                                   float m, int numactions,
                                   float rmax, float qmax, float rrange,
                                   int nfactors, float b,
                                   const std::vector<float> &fmax,
                                   const std::vector<float> &fmin, Random rng):
  modelType(modelType), exploreType(exploreType), predType(predType),
  nModels(nModels),
  M(m), numactions(numactions), rmax(rmax), qmax(qmax), rrange(rrange),
  nfactors(nfactors), b(b), rng(rng)
{

  model = innermodel;

  MODEL_DEBUG = false; //true;

  cout << "Exploration Model " << exploreType << ", b: " << b << endl;

  featmax = fmax;
  featmin = fmin;

}

ExplorationModel::ExplorationModel(const ExplorationModel &em):
modelType(em.modelType), exploreType(em.exploreType), predType(em.predType),
  nModels(em.nModels),
  M(em.M), numactions(em.numactions), rmax(em.rmax), qmax(em.qmax), rrange(em.rrange),
  nfactors(em.nfactors), b(em.b), rng(em.rng)
{
  model = em.model->getCopy();
  MODEL_DEBUG = em.MODEL_DEBUG;
  featmax = em.featmax;
  featmin = em.featmin;
  statespace = em.statespace;
}

ExplorationModel* ExplorationModel::getCopy(){
  ExplorationModel* copy = new ExplorationModel(*this);
  return copy;
}


ExplorationModel::~ExplorationModel() {
  delete model;
}



bool ExplorationModel::updateWithExperiences(std::vector<experience> &instances){
  bool changed = model->updateWithExperiences(instances);
  
  // keep track of which states we've been to for this mode
  for (unsigned i = 0; i < instances.size(); i++){
    if (exploreType == UNVISITED_BONUS)
      addStateToSet(instances[i].s);

    if (exploreType == UNVISITED_ACT_BONUS){
      std::vector<float> last2 = instances[i].s;
      last2.push_back(instances[i].act);
      addStateToSet(last2);
    }
  }

  return changed;
}

// update all the counts, check if model has changed
// stop counting at M
bool ExplorationModel::updateWithExperience(experience &e){
  //if (MODEL_DEBUG) cout << "updateWithExperience " << &last << ", " << act
  //        << ", " << &curr << ", " << reward << endl;

  bool changed = model->updateWithExperience(e);

  // keep track of which states we've been to for this mode
  if (exploreType == UNVISITED_BONUS)
    addStateToSet(e.s);

  if (exploreType == UNVISITED_ACT_BONUS){
    std::vector<float> last2 = e.s;
    last2.push_back(e.act);
    addStateToSet(last2);
  }

  // anything that got past the 'return false' above is a change in conf or predictions
  return changed; //modelChanged;

}


// calculate state info such as transition probs, known/unknown, reward prediction
bool ExplorationModel::getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval){
  //if (MODEL_DEBUG) cout << "getStateActionInfo, " << &state <<  ", " << act << endl;

  retval->transitionProbs.clear();

  model->getStateActionInfo(state, act, retval);


  //cout << "state: " << state[0] << " act: " << act;

  if (MODEL_DEBUG)// || (retval->conf > 0.0 && retval->conf < 1.0))
    cout << "reward: " << retval->reward << " conf: " << retval->conf << endl;

  // check exploration bonuses
  switch(exploreType){

    // use qmax if state is unknown
  case(EXPLORE_UNKNOWN):
    if (!retval->known){
      if (MODEL_DEBUG){
        cout << "State-Action Unknown in model: conf: " << retval->conf << " ";
        for (unsigned si = 0; si < state.size(); si++){
          cout << (state)[si] << ",";
        }
        cout << " Action: " << act << endl;
      }
      retval->reward = qmax;
      retval->termProb = 1.0;
      if (MODEL_DEBUG || MODEL_DEBUG)
        cout << "   State-Action Unknown in model, using qmax "
             << qmax << endl;
    }
    break;

    // small bonus for unvisited states
  case(UNVISITED_BONUS):
    if (!checkForState(state)){
      // modify reward with a bonus of 10% of rmax
      float newQ = 0.75*retval->reward + 0.25*rmax;
      if (MODEL_DEBUG){
        cout << "   State unvisited bonus, orig R: "
             << retval->reward
             << " adding 25% rmax: " << rmax
             << " new value : " << newQ
             << endl;
      }
      retval->reward = newQ;
    }
    break;

    // small bonus for unvisited state-actions
  case(UNVISITED_ACT_BONUS):
    {
      std::vector<float> state2 = state;
      state2.push_back(act);
      if (!checkForState(state2)){
        // modify reward with a bonus of 10% of rmax
        float newQ = 0.75*retval->reward + 0.25*rmax;
        if (MODEL_DEBUG){
          cout << "   State-Action unvisited bonus, orig R: "
               << retval->reward
               << " adding 25% rmax: " << rmax
               << " new value : " << newQ
               << endl;
        }
        retval->reward = newQ;
      }
      break;
    }

    // use some % of qmax if we're doing continuous terminal bonus
  case(CONTINUOUS_BONUS):
    if (retval->conf < 1.0){
      // percent of conf
      float bonus = (1.0-retval->conf)*b;
      if (MODEL_DEBUG){
        cout << "   State-Action continuous bonus conf: "
             << retval->conf
             << ", using qmax*(1-pct): "
             << bonus << endl;
      }
      retval->reward = bonus;
      retval->termProb = 1.0;
    }
    break;

    // use some % of rmax if we're doing continuous bonus
  case(CONTINUOUS_BONUS_R):
    if (retval->conf < 1.0){
      // percent of conf
      float bonus = (1.0-retval->conf)*b;
      retval->reward += bonus;
      if (MODEL_DEBUG){
        cout << "   State-Action continuous bonus conf: "
             << retval->conf
             << ", using rmax*(1-pct): "
             << bonus << endl;
      }
    }
    break;

    // use qmax if we're doing threshold terminal bonus and conf under threshold
  case(THRESHOLD_BONUS):
    if (retval->conf < 0.5){
      float bonus = b;
      if (MODEL_DEBUG){
        cout << "   State-Action conf< thresh: "
             << retval->conf
             << " M: " << M
             << ", using qmax "
             << qmax << endl;
      }
      retval->reward = bonus;
      retval->termProb = 1.0;
    }
    break;

    // use rmax for additional thresh bonus and conf under thresh
  case(THRESHOLD_BONUS_R):
    if (retval->conf < 0.9){
      float bonus = b;
      retval->reward += bonus;
      if (MODEL_DEBUG){
        cout << "   State-Action conf< thresh: "
             << retval->conf
             << " M: " << M
             << ", using rrange "
             << rmax << endl;
      }
    }
    break;

    // visits conf
  case(VISITS_CONF):
    if (retval->conf < 0.5){
      float bonus = qmax;
      retval->reward += bonus;
      if (MODEL_DEBUG){
        cout << "   State-Action conf< thresh or 0 visits: "
             << retval->conf
             << " M: " << M
             << ", using qmax "
             << qmax << endl;
      }
      retval->reward = bonus;
      retval->termProb = 1.0;
    }
    break;

  default:
    break;
  }

  if (MODEL_DEBUG)
    cout << "   Conf: " << retval->conf << "   Avg reward: " << retval->reward << endl;
  if (isnan(retval->reward))
    cout << "ERROR: Model returned reward of NaN" << endl;

  return true;

}

// add state to set (if its not already in it)
void ExplorationModel::addStateToSet(const std::vector<float> &s){
  statespace.insert(s);
}

// check if state is in set (so we know if we've visited it)
bool ExplorationModel::checkForState(const std::vector<float> &s){
  return (statespace.count(s) == 1);
}
