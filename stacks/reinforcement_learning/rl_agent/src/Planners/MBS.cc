#include "MBS.hh"
#include <algorithm>

//#include <time.h>
#include <sys/time.h>


MBS::MBS(int numactions, float gamma,
         int MAX_LOOPS, float MAX_TIME, int modelType,
         const std::vector<float> &fmax, 
         const std::vector<float> &fmin, 
         const std::vector<int> &n, 
         const int k, Random newRng):
  k(k)
{
  
  vi = new ValueIteration(numactions, gamma, MAX_LOOPS, MAX_TIME, modelType,
                          fmax, fmin, n, newRng);
  DELAYDEBUG = false; //true;
  seedMode = false;

  if (DELAYDEBUG) cout << "MBS delay planner with k = " << k << endl;
}

MBS::~MBS() {
  delete vi;
}

void MBS::setModel(MDPModel* m){
  vi->setModel(m);
  model = m;
}


/** Use the latest experience to update state info and the model. */
bool MBS::updateModelWithExperience(const std::vector<float> &laststate,
                                    int lastact,
                                    const std::vector<float> &currstate,
                                    float reward, bool term){

  if (seedMode) return false;
  
  // add this action to our history list
  if (DELAYDEBUG) cout << "add new action " << lastact << " to list" << endl;
  actHistory.push_back(lastact);
  
  if (actHistory.size() > k){
    int effectiveAction = actHistory.front();
    actHistory.pop_front();
    
    // if history size is >= k
    // then we can add this experience
    if (DELAYDEBUG){
      cout << "update with old act: " << effectiveAction << endl;
      cout << "from: " << laststate[0] << ", " << laststate[1];
      cout << " to: " << currstate[0] << ", " << currstate[1];
      cout << " reward: " << reward << " term: " << term << endl;
    }
    

    return vi->updateModelWithExperience(laststate, effectiveAction,
                                         currstate, reward, term);
  }

  return false;

}


/** Choose the next action */
int MBS::getBestAction(const std::vector<float> &state){
  std::vector<float> statePred = state;

  // figure out what state we think we're in
  for (unsigned i = 0; i < actHistory.size(); i++){
    if (DELAYDEBUG) cout << i << " prediction: " 
                         << statePred[0] << ", " << statePred[1] 
                         << " pred for act: " << actHistory[i] << endl;
    StateActionInfo prediction;
    model->getStateActionInfo(statePred, actHistory[i], &prediction);

    // find most likely next state
    std::vector<float> possibleNext;
    float maxProb = -1;
    for (std::map<std::vector<float>, float>::iterator it = prediction.transitionProbs.begin(); it != prediction.transitionProbs.end(); it++){
      
      float prob = (*it).second;
      if (prob > maxProb){
        possibleNext = (*it).first;
        maxProb = prob;
      }
    }
    statePred = possibleNext;

  }
  if (DELAYDEBUG) cout << "predict current state is " << statePred[0] << ", " << statePred[1] << endl;
    
  // call get best action for that state
  int act = vi->getBestAction(statePred);

  if (DELAYDEBUG) cout << "best action is " << act << endl << endl;

  return act;

}


void MBS::planOnNewModel(){
  vi->planOnNewModel();
}

void MBS::savePolicy(const char* filename){
  vi->savePolicy(filename);
}

void MBS::setSeeding(bool seeding){

  seedMode = seeding;

}


void MBS::setFirst(){ 
  // first action, reset history vector
  actHistory.clear();
}
