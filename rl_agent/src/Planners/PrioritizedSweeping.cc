#include "PrioritizedSweeping.hh"
#include <algorithm>

//#include <time.h>
#include <sys/time.h>


PrioritizedSweeping::PrioritizedSweeping(int numactions, float gamma,
                                         float MAX_TIME, bool onlyAddLastSA,  int modelType,
                                         const std::vector<float> &fmax, 
                                         const std::vector<float> &fmin, 
                                         Random r):
  numactions(numactions), gamma(gamma), MAX_TIME(MAX_TIME),
  onlyAddLastSA(onlyAddLastSA),  modelType(modelType)
{
  rng = r;
  nstates = 0;
  nactions = 0;

  timingType = false; //true;

  model = NULL;
  planTime = getSeconds();

  // algorithm options
  MAX_STEPS = 10; //50; //60; //80; //0; //5; //10;

  lastModelUpdate = -1;

  PLANNERDEBUG = false;
  POLICYDEBUG = false; //true; //false;
  ACTDEBUG = false; //true;
  MODELDEBUG = false; //true;
  LISTDEBUG = false; // true; //false;

  featmax = fmax;
  featmin = fmin;

}

PrioritizedSweeping::~PrioritizedSweeping() {}

void PrioritizedSweeping::setModel(MDPModel* m){

  model = m;

}


/////////////////////////////
// Functional functions :) //
/////////////////////////////


void PrioritizedSweeping::initNewState(state_t s){
  if (PLANNERDEBUG) cout << "initNewState(s = " << s
                         << ") size = " << s->size() << endl;

  if (MODELDEBUG) cout << "New State: " << endl;

  // create state info and add to hash map
  state_info* info = &(statedata[s]);
  initStateInfo(info);

  // init these from model
  for (int i = 0; i < numactions; i++){
    model->getStateActionInfo(*s, i, &(info->modelInfo[i]));
  }

  // we have to make sure q-values are initialized properly
  // or we'll get bizarre results (if these aren't swept over)
  for (int j = 0; j < numactions; j++){
    // update q values
    updateQValues(*s, j);
    info->Q[j] += rng.uniform(0,0.01);
  }

  if (PLANNERDEBUG) cout << "done with initNewState()" << endl;

}

/** Use the latest experience to update state info and the model. */
bool PrioritizedSweeping::updateModelWithExperience(const std::vector<float> &laststate,
                                                    int lastact,
                                                    const std::vector<float> &currstate,
                                                    float reward, bool term){
  if (PLANNERDEBUG) cout << "updateModelWithExperience(last = " << &laststate
                         << ", curr = " << &currstate
                         << ", lastact = " << lastact
                         << ", r = " << reward
                         << ")" << endl;

  if (!timingType)
    planTime = getSeconds();

  // canonicalize these things
  state_t last = canonicalize(laststate);
  state_t curr = canonicalize(currstate);

  prevstate = laststate;
  prevact = lastact;

  // if not transition to terminal
  if (curr == NULL)
    return false;

  // get state info
  state_info* info = &(statedata[last]);

  // update the state visit count
  info->visits[lastact]++;

  // init model?
  if (model == NULL){
    cout << "ERROR IN MODEL OR MODEL SIZE" << endl;
    exit(-1);
  }

  experience e;
  e.s = *last;
  e.next = *curr;
  e.act = lastact;
  e.reward = reward;
  e.terminal = term;
  bool modelChanged = model->updateWithExperience(e);

  if (PLANNERDEBUG) cout << "Added exp: " << modelChanged << endl;
  if (timingType)
    planTime = getSeconds();

  return modelChanged;

}



/** Update our state info's from the model by calling the model function */
void PrioritizedSweeping::updateStatesFromModel(){
  if (PLANNERDEBUG || LISTDEBUG) cout << "updateStatesFromModel()" << endl;

  // for each state
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    for (int j = 0; j < numactions; j++){
      updateStateActionFromModel(*i, j);
    }

  }

}




/** Choose the next action */
int PrioritizedSweeping::getBestAction(const std::vector<float> &state){
  if (PLANNERDEBUG) cout << "getBestAction(s = " << &state
                         << ")" << endl;

  state_t s = canonicalize(state);

  // get state info
  state_info* info = &(statedata[s]);

  // Get Q values
  std::vector<float> &Q = info->Q;

  // Choose an action
  const std::vector<float>::iterator a =
    random_max_element(Q.begin(), Q.end()); // Choose maximum

  int act = a - Q.begin();
  float val = *a;

  if (ACTDEBUG){
    cout << endl << "chooseAction State " << (*s)[0] << "," << (*s)[1]
         << " act: " << act << " val: " << val << endl;
    for (int iAct = 0; iAct < numactions; iAct++){
      cout << " Action: " << iAct
           << " val: " << Q[iAct]
           << " visits: " << info->visits[iAct]
           << " modelsAgree: " << info->modelInfo[iAct].known << endl;
    }
  }

  nactions++;

  // return index of action
  return act;
}



void PrioritizedSweeping::planOnNewModel(){

  // update model info

  // print state
  if (PLANNERDEBUG){
    cout << endl << endl << "Before update" << endl << endl;
    printStates();
  }

  // tabular - can just update last state-action from model.
  if (false && modelType == RMAX){
    updateStateActionFromModel(prevstate, prevact);
  }
  else {
    updateStatesFromModel();
  }

  // just add last state action (this is normal prioritized sweeping).
  // if bool was false, will have checked for differences and added them in update above
  if (onlyAddLastSA || modelType == RMAX){
    float diff = updateQValues(prevstate, prevact);
    addSAToList(prevstate, prevact, diff);
  }

  if (PLANNERDEBUG){
    cout << endl << endl << "After update" << endl << endl;
    printStates();
  }

  // run value iteration
  createPolicy();

}


////////////////////////////
// Helper Functions       //
////////////////////////////

PrioritizedSweeping::state_t PrioritizedSweeping::canonicalize(const std::vector<float> &s) {
  if (PLANNERDEBUG) cout << "canonicalize(s = " << s[0] << ", "
                         << s[1] << ")" << endl;

  // get state_t for pointer if its in statespace
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s);
  state_t retval = &*result.first; // Dereference iterator then get pointer

  if (PLANNERDEBUG) cout << " returns " << retval
                         << " New: " << result.second << endl;

  // if not, init this new state
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    initNewState(retval);
    if (PLANNERDEBUG) cout << " New state initialized" << endl;
  }


  return retval;
}

// init state info
void PrioritizedSweeping::initStateInfo(state_info* info){
  if (PLANNERDEBUG) cout << "initStateInfo()";

  info->id = nstates++;
  if (PLANNERDEBUG) cout << " id = " << info->id << endl;

  info->fresh = true;

  // model data (transition, reward, known)
  info->modelInfo = new StateActionInfo[numactions];

  // model q values, visit counts
  info->visits.resize(numactions, 0);
  info->Q.resize(numactions, 0);
  info->lastUpdate.resize(numactions, nactions);

  for (int i = 0; i < numactions; i++){
    info->Q[i] = rng.uniform(0,1);
  }

  if (PLANNERDEBUG) cout << "done with initStateInfo()" << endl;

}


/** Print state info for debugging. */
void PrioritizedSweeping::printStates(){

  for (std::set< std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);

    state_info* info = &(statedata[s]);

    cout << endl << "State " << info->id << ": ";
    for (unsigned j = 0; j < s->size(); j++){
      cout << (*s)[j] << ", ";
    }
    cout << endl;

    for (int act = 0; act < numactions; act++){
      cout << " visits[" << act << "] = " << info->visits[act]
           << " Q: " << info->Q[act]
           << " R: " << info->modelInfo[act].reward << endl;

      cout << "  Next states: " << endl;
      for (std::map<std::vector<float>, float>::iterator outIt
             = info->modelInfo[act].transitionProbs.begin();
           outIt != info->modelInfo[act].transitionProbs.end(); outIt++){

        std::vector<float> nextstate = (*outIt).first;
        float prob = (*outIt).second;

        cout << "   State ";
        for (unsigned k = 0; k < nextstate.size(); k++){
          cout << nextstate[k] << ", ";
        }
        cout << " prob: " << prob << endl;

      } // end of next states

    } // end of actions

    // print predecessors
    for (std::list<saqPair>::iterator x = info->pred.begin();
         x != info->pred.end(); x++){

      std::vector<float> s = (*x).s;
      int a = (*x).a;

      cout << "Has predecessor state: ";
      for (unsigned k = 0; k < s.size(); k++){
        cout << s[k] << ", ";
      }
      cout << " action: " << a << endl;
    }

  }
}





void PrioritizedSweeping::deleteInfo(state_info* info){

  delete [] info->modelInfo;

}


double PrioritizedSweeping::getSeconds(){
  struct timezone tz;
  timeval timeT;
  gettimeofday(&timeT, &tz);
  return  timeT.tv_sec + (timeT.tv_usec / 1000000.0);
}


/** Create policy through prioritized sweeping. */
void PrioritizedSweeping::createPolicy(){
  if (POLICYDEBUG) cout << endl << "createPolicy()" << endl;

  /*
  // loop through all states, add them all to queue with some high value.
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
  i != statespace.end(); i++){

  saqPair saq;
  saq.s = *i;
  saq.q = 100.0;

  for (int j = 0; j < numactions; j++){
  saq.a = j;

  if (LISTDEBUG){
  cout << "Added state ";
  for (unsigned k = 0; k < saq.s.size(); k++){
  cout << saq.s[k] << ", ";
  }
  cout << " action: " << saq.a << endl;
  }

  priorityList.push_front(saq);
  }
  }
  */

  // add last state-action to priority list
  //addSAToList(prevstate, prevact, 100.0);

  int updates = 0;

  // go through queue, doing prioritized sweeping. until nothing left on queue.
  while (!priorityList.empty()){

    if ((getSeconds() - planTime) > MAX_TIME)
      break;

    // print list!
    if (LISTDEBUG){
      cout << endl << "Current List (" << updates << "):" << endl;
      for (std::list<saqPair>::iterator k = priorityList.begin(); k != priorityList.end(); k++){
        cout << "State: ";
        for (unsigned l = 0; l < (*k).s.size(); l++){
          cout << (*k).s[l] << ", ";
        }
        cout << " act: " << (*k).a << " Q: " << (*k).q << endl;
      }
    }

    updates++;

    // pull off first item
    saqPair currUpdate = priorityList.front();
    priorityList.pop_front();

    state_t s = canonicalize(currUpdate.s);

    // get state's info
    state_info* info = &(statedata[s]);

    updatePriorityList(info, *s);

  } // is list empty loop


  priorityList.clear();

  if (LISTDEBUG)
    cout << "priority list complete after updates to "
         << updates << " states." <<endl;

}


void PrioritizedSweeping::updatePriorityList(state_info* info,
                                             const std::vector<float> &next){
  if (LISTDEBUG) cout << "update priority list" << endl;

  float MIN_ERROR = 0.01;

  // find maxq at this state
  std::vector<float>::iterator maxAct =
    std::max_element(info->Q.begin(),
                     info->Q.end());
  float maxval = *maxAct;

  if (LISTDEBUG) cout << " maxQ at this state: " << maxval << endl;

  // loop through all s,a predicted to lead to this state
  for (std::list<saqPair>::iterator i = info->pred.begin();
       i != info->pred.end(); i++){

    if ((getSeconds() - planTime) > MAX_TIME)
      break;

    std::vector<float> s = (*i).s;
    int a = (*i).a;

    if (LISTDEBUG) {
      cout << endl << "  For predecessor state: ";
      for (unsigned j = 0; j < s.size(); j++){
        cout << s[j] << ", ";
      }
      cout << " action: " << a << endl;
    }

    // figure out amount of update
    float diff = updateQValues(s, a);

    if (LISTDEBUG) {
      cout << " diff: " << diff << endl;
    }
    // possibly add to queue
    if (diff > MIN_ERROR){
      saqPair saq;
      saq.s = s;
      saq.a = a;
      saq.q = diff;

      // find spot for it in queue
      if (priorityList.empty()){
        if (LISTDEBUG) cout << "  empty list" << endl;
        priorityList.push_front(saq);
      }
      else {

        // check that its not already in queue
        for (std::list<saqPair>::iterator k = priorityList.begin(); k != priorityList.end(); k++){
          // matched
          if (saqPairMatch(saq, *k)){
            if (LISTDEBUG)
              cout << "   found matching element already in list" << endl;

            priorityList.erase(k);
            break;
          }

        }

        int l = 0;
        std::list<saqPair>::iterator k;
        for (k = priorityList.begin(); k != priorityList.end(); k++){
          if (LISTDEBUG)
            cout << "    Element " << l << " has q value " << (*k).q << endl;
          if (diff > (*k).q){
            if (LISTDEBUG)
              cout << "   insert at " << l << endl;
            priorityList.insert(k, saq);
            break;
          }
          l++;
        }
        // put this at the end
        if (k == priorityList.end()){
          if (LISTDEBUG)
            cout << "   insert at end" << endl;
          priorityList.push_back(saq);
        }

      } // not empty


    } else {
      if (LISTDEBUG){
        cout << " Error " << diff << " not big enough to put on list." << endl;
      }
    }
  }
}


bool PrioritizedSweeping::saqPairMatch(saqPair a, saqPair b){
  if (a.a != b.a)
    return false;

  for (unsigned i = 0; i < a.s.size(); i++){
    if (a.s[i] != b.s[i])
      return false;
  }

  return true;
}



float PrioritizedSweeping::updateQValues(const std::vector<float> &state, int act){

  state_t s = canonicalize(state);

  // get state's info
  state_info* info = &(statedata[s]);

  // see if we should update mode for this state,action
  /*
    if (info->lastUpdate[act] < lastModelUpdate){
    if (LISTDEBUG) {
    cout << "Updating this state action. Last updated at "
    << info->lastUpdate[act]
    << " last model update: " << lastModelUpdate << endl;
    }
    updateStateActionFromModel(state, act);
    }
  */

  if (LISTDEBUG || POLICYDEBUG){
    cout << endl << " State: id: " << info->id << ": " ;
    for (unsigned si = 0; si < s->size(); si++){
      cout << (*s)[si] << ",";
    }
  }

  // get state action info for this action
  StateActionInfo *modelInfo = &(info->modelInfo[act]);

  if (LISTDEBUG || POLICYDEBUG)
    cout << "  Action: " << act
         << " State visits: " << info->visits[act] << endl;

  // Q = R + discounted val of next state
  // this is the R part :)
  float newQ = modelInfo->reward;

  float probSum = modelInfo->termProb;

  // for all next states, add discounted value appropriately
  // loop through next state's that are in this state-actions list
  for (std::map<std::vector<float>, float>::iterator outIt
         = modelInfo->transitionProbs.begin();
       outIt != modelInfo->transitionProbs.end(); outIt++){

    std::vector<float> nextstate = (*outIt).first;

    if (POLICYDEBUG){
      cout << "  Next state was: ";
      for (unsigned oi = 0; oi < nextstate.size(); oi++){
        cout << nextstate[oi] << ",";
      }
      cout << endl;
    }

    // get transition probability
    float transitionProb =  (1.0-modelInfo->termProb) *
      modelInfo->transitionProbs[nextstate];

    probSum += transitionProb;

    if (POLICYDEBUG)
      cout << "   prob: " << transitionProb << endl;

    if (transitionProb < 0 || transitionProb > 1.0001){
      cout << "Error with transitionProb: " << transitionProb << endl;
      exit(-1);
    }

    // if there is some probability of this transition
    if (transitionProb > 0.0){

      // assume maxval of qmax if we don't know the state
      float maxval = 0.0;

      // make sure its a real state
      bool realState = true;


      for (unsigned b = 0; b < nextstate.size(); b++){
        if (nextstate[b] < (featmin[b]-EPSILON)
            || nextstate[b] > (featmax[b]+EPSILON)){
          realState = false;
          if (POLICYDEBUG)
            cout << "    Next state is not valid (feature "
                 << b << " out of range)" << endl;
          break;
        }
      }



      // update q values for any states within MAX_STEPS of visited states
      if (realState){

        state_t next = canonicalize(nextstate);

        state_info* nextinfo = &(statedata[next]);
        //nextinfo->fresh = false;

        // find the max value of this next state
        std::vector<float>::iterator maxAct =
          std::max_element(nextinfo->Q.begin(),
                           nextinfo->Q.end());
        maxval = *maxAct;

      } // within max steps
      else {
        maxval = 0.0;
        if (POLICYDEBUG){
          cout << "This state is too far away, state: ";
          for (unsigned si = 0; si < s->size(); si++){
            cout << (*s)[si] << ",";
          }
          cout << " Action: " << act << endl;
        }
      }

      nextstate.clear();

      if (POLICYDEBUG) cout << "    Max value: " << maxval << endl;

      // update q value with this value
      newQ += (gamma * transitionProb * maxval);

    } // transition probability > 0

  } // outcome loop


  if (probSum < 0.9999 || probSum > 1.0001){
    cout << "Error: transition probabilities do not add to 1: Sum: "
         << probSum << endl;
    exit(-1);
  }


  // set q value
  float tdError = fabs(info->Q[act] - newQ);
  if (LISTDEBUG || POLICYDEBUG) cout << "  NewQ: " << newQ
                                     << " OldQ: " << info->Q[act] << endl;
  info->Q[act] = newQ;

  return tdError;
}

void PrioritizedSweeping::addSAToList(const std::vector<float> &s, int act, float q){

  saqPair saq;
  saq.s = s;
  saq.a = act;
  saq.q = q;

  if (LISTDEBUG){
    cout << "Added state ";
    for (unsigned k = 0; k < saq.s.size(); k++){
      cout << saq.s[k] << ", ";
    }
    cout << " action: " << saq.a
         << " value: " << saq.q << endl;
  }

  priorityList.push_front(saq);

}



/** Update a single state-action from the model */
void PrioritizedSweeping::updateStateActionFromModel(const std::vector<float> &state, int a){

  if ((getSeconds() - planTime) > MAX_TIME)
    return;

  state_t s = canonicalize(state);

  // get state's info
  state_info* info = &(statedata[s]);

  int j = a;

  // get updated model
  model->getStateActionInfo(*s, j, &(info->modelInfo[j]));
  info->lastUpdate[j] = nactions;

  if (info->modelInfo[j].termProb >= 1.0)
    return;

  // go through next states, for each one, add self to predecessor list
  for (std::map<std::vector<float>, float>::iterator outIt
         = info->modelInfo[j].transitionProbs.begin();
       outIt != info->modelInfo[j].transitionProbs.end(); outIt++){

    std::vector<float> nextstate = (*outIt).first;
    state_t next = canonicalize(nextstate);
    state_info* nextinfo = &(statedata[next]);
    //float prob = (*outIt).second;

    if (LISTDEBUG){
      cout << "State ";
      for (unsigned k = 0; k < nextstate.size(); k++){
        cout << nextstate[k] << ", ";
      }
      cout << " has predecessor: ";
      for (unsigned k = 0; k < nextstate.size(); k++){
        cout << (*s)[k] << ", ";
      }
      cout << " action: " << j << endl;
    }

    saqPair saq;
    saq.s = *s;
    saq.a = j;
    saq.q = 0.0;

    // add to list
    // check that its not already here
    bool nothere = true;
    for (std::list<saqPair>::iterator k = nextinfo->pred.begin();
         k != nextinfo->pred.end(); k++){
      if (saqPairMatch(saq, *k)){
        nothere = false;
        break;
      }
    }
    if (nothere)
      nextinfo->pred.push_front(saq);

  }

  info->fresh = false;

  //if (PLANNERDEBUG || LISTDEBUG) cout << " updateStatesFromModel i = " << &i << " complete" << endl;

}


