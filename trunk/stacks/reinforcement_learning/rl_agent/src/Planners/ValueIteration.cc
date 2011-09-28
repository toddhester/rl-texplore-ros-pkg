/** \file ValueIteration.cc
    Implements the ValueIteration class
    \author Todd Hester
*/

#include "ValueIteration.hh"
#include <algorithm>

#include <sys/time.h>


ValueIteration::ValueIteration(int numactions, float gamma,
                               int MAX_LOOPS, float MAX_TIME, int modelType,
                               const std::vector<float> &fmax, 
                               const std::vector<float> &fmin, 
                               const std::vector<int> &n, Random newRng):
  numactions(numactions), gamma(gamma),
  MAX_LOOPS(MAX_LOOPS), MAX_TIME(MAX_TIME), modelType(modelType),
  statesPerDim(n)
{
  rng = newRng;

  nstates = 0;
  nactions = 0;

  timingType = false; //true;

  model = NULL;
  planTime = getSeconds();

  // algorithm options
  MAX_STEPS = 100; //50; //60; //80; //0; //5; //10;

  PLANNERDEBUG = false;
  POLICYDEBUG = false;
  ACTDEBUG = false;
  MODELDEBUG = false;

  featmax = fmax;
  featmin = fmin;

  if (statesPerDim[0] > 0){
    cout << "Planner VI using discretization of " << statesPerDim[0] << endl;
  }


}

ValueIteration::~ValueIteration() {
  for (std::map<state_t, state_info>::iterator i = statedata.begin();
       i != statedata.end(); i++){
    
    // get state's info
    //cout << "  planner got info" << endl;
    state_info* info = &((*i).second);

    deleteInfo(info);
  }

  statedata.clear();
  
}

void ValueIteration::setModel(MDPModel* m){

  model = m;

  //  initStates();

}


// canonicalize all the states so we already have them in our statespace
void ValueIteration::initStates(){
  cout << "init states" << endl;
  std::vector<float> s(featmin.size());

  fillInState(s,0);
  cout << "init states complete" << endl;
}

void ValueIteration::fillInState(std::vector<float>s, int depth){

  // if depth == size, canonicalize and return
  if (depth == (int)featmin.size()){
    canonicalize(s);
    return;
  }

  // go through all features at depth
  for (float i = featmin[depth]; i < featmax[depth]+1; i++){
    s[depth] = i;
    fillInState(s, depth+1);
  }
}


/////////////////////////////
// Functional functions :) //
/////////////////////////////


void ValueIteration::initNewState(state_t s){
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


  if (PLANNERDEBUG) cout << "done with initNewState()" << endl;

}

bool ValueIteration::updateModelWithExperience(const std::vector<float> &laststate,
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
  e.s = laststate;
  e.next = currstate;
  e.act = lastact;
  e.reward = reward;
  e.terminal = term;
  bool modelChanged = model->updateWithExperience(e);

  if (PLANNERDEBUG) cout << "VI Added exp: " << modelChanged << endl;
  if (timingType)
    planTime = getSeconds();

  return modelChanged;

}


void ValueIteration::updateStateActionFromModel(const std::vector<float> &state, int a){
  if (PLANNERDEBUG) cout << "updateStateActionFromModel()" << endl;

  state_t s = canonicalize(state);

  // get state's info
  state_info* info = &(statedata[s]);

  // update state info
  // get state action info for each action
  model->getStateActionInfo(state, a, &(info->modelInfo[a]));

  info->fresh = false;

}


void ValueIteration::updateStatesFromModel(){
  if (PLANNERDEBUG) cout << "updateStatesFromModel()" << endl;

  // for each state
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    if (PLANNERDEBUG){
      cout << "updateStatesFromModel i = " << &(*i) << endl;
      cout << "State is ";
      for (unsigned j = 0; j < (*i).size(); j++){
        cout << (*i)[j] << ", ";
      }
      cout << endl;
    }

    state_t s = canonicalize(*i);

    // get state's info
    state_info* info = &(statedata[s]);

    // update state info
    // get state action info for each action
    for (int j = 0; j < numactions; j++){
      model->getStateActionInfo(*s, j, &(info->modelInfo[j]));
    }

    //s2.clear();

    info->fresh = false;

    if (PLANNERDEBUG) cout << "updateStatesFromModel i = " << &i << " complete" << endl;

  }

  if (PLANNERDEBUG) cout << "updateStatesFromModel " << " totally complete" << endl;

}


int ValueIteration::getBestAction(const std::vector<float> &state){
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




// use VI to compute new policy using model
void ValueIteration::createPolicy(){
  if (PLANNERDEBUG || POLICYDEBUG) cout << endl << "createPolicy()" << endl;

  float maxError = 5000;
  int nloops = 0;

  calculateReachableStates();

  float MIN_ERROR = 0.0001;
  //float initTime = getSeconds();
  //cout << "max time " << MAX_TIME  << " max loops: " << MAX_LOOPS << endl;
  int statesUpdated = 0;

  // until convergence (always at least MIN_LOOPS)
  while (maxError > MIN_ERROR){ // && nloops < MAX_LOOPS){

    //if ((getSeconds() - initTime) > MAX_TIME)
    // break;

    //    if ((getSeconds() - planTime) > MAX_TIME)
    //break;

    if (POLICYDEBUG)
      cout << "max error: " << maxError << " nloops: " << nloops
           << endl;

    maxError = 0;
    nloops++;

    // for all states
    for (std::set<std::vector<float> >::iterator i = statespace.begin();
         i != statespace.end(); i++){

      //      if ((getSeconds() - planTime) > MAX_TIME)
      //break;

      statesUpdated++;
      state_t s = canonicalize(*i);

      // get state's info
      state_info* info = &(statedata[s]);

      if (POLICYDEBUG){
        cout << endl << " State: id: " << info->id << ": " ;
        for (unsigned si = 0; si < s->size(); si++){
          cout << (*s)[si] << ",";
        }
        cout << " Steps: " << info->stepsAway << endl;

      }

      // skip states we can't reach
      if (info->stepsAway > 99999){
        if (POLICYDEBUG)
          cout << "State not reachable, ignoring." << endl;
        continue;
      }

      // for each action
      for (int act = 0; act < numactions; act++){



        // get state action info for this action
        StateActionInfo *modelInfo = &(info->modelInfo[act]);

        if (POLICYDEBUG)
          cout << "  Action: " << act
               << " State visits: " << info->visits[act]
               << " reward: " << modelInfo->reward 
	       << " term: " << modelInfo->termProb << endl;

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
          float transitionProb = (1.0-modelInfo->termProb) *
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

            state_t next;
            
            // update q values for any states within MAX_STEPS of visited states
            if (info->stepsAway >= MAX_STEPS || !realState){
              next = s;
            } else {
              next = canonicalize(nextstate);
            }
            
            state_info* nextinfo = &(statedata[next]);
            //nextinfo->fresh = false;
            
            int newSteps = info->stepsAway + 1;
            if (newSteps < nextinfo->stepsAway){
              if (POLICYDEBUG) {
                cout << "    Setting state to "
                     << newSteps << " steps away." << endl;
              }
              nextinfo->stepsAway = newSteps;
            }
            
            // find the max value of this next state
            std::vector<float>::iterator maxAct =
              std::max_element(nextinfo->Q.begin(),
                               nextinfo->Q.end());
            maxval = *maxAct;
            
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
        if (POLICYDEBUG) cout << "  NewQ: " << newQ
                              << " OldQ: " << info->Q[act] << endl;
        info->Q[act] = newQ;

        // check max error
        if (tdError > maxError)
          maxError = tdError;

        if (POLICYDEBUG)
          cout << "  TD error: " << tdError
               << " Max error: " << maxError << endl;

      } // action loop

    } // state loop

  } // while not converged loop

  if (false || nloops >= MAX_LOOPS){
    cout << nactions << " Policy creation ended with maxError: " << maxError
         << " nloops: " << nloops << " time: " << (getSeconds()-planTime)
         << " states: " << statesUpdated
         << endl;
  }

  // remove unreachable states
  removeUnreachableStates();

  if (POLICYDEBUG) cout << nactions
                        << " policy creation complete: maxError: "
                        << maxError << " nloops: " << nloops
                        << endl;


}


void ValueIteration::planOnNewModel(){

  // update model info
  // can just update one for tabular model
  if (modelType == RMAX){
    updateStateActionFromModel(prevstate, prevact);
  }
  else {
    updateStatesFromModel();
  }

  // run value iteration
  createPolicy();

}


////////////////////////////
// Helper Functions       //
////////////////////////////

ValueIteration::state_t ValueIteration::canonicalize(const std::vector<float> &s) {
  if (PLANNERDEBUG) cout << "canonicalize(s = " << s[0] << ", "
                         << s[1] << ")" << endl;

  std::vector<float> s2;
  if (statesPerDim[0] > 0){
    s2 = discretizeState(s);
  } else {
    s2 = s;
  }

  if (PLANNERDEBUG) cout << "discretized(" << s2[0] << ", " << s2[1] << ")" << endl;

  // get state_t for pointer if its in statespace
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s2);
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
void ValueIteration::initStateInfo(state_info* info){
  if (PLANNERDEBUG) cout << "initStateInfo()";

  info->id = nstates++;
  if (PLANNERDEBUG) cout << " id = " << info->id << endl;

  info->fresh = true;
  info->stepsAway = 100000;

  // model data (transition, reward, known)
  info->modelInfo = new StateActionInfo[numactions];

  // model q values, visit counts
  info->visits.resize(numactions, 0);
  info->Q.resize(numactions, 0);

  for (int i = 0; i < numactions; i++){
    info->Q[i] = rng.uniform(0,0.01);
  }

  if (PLANNERDEBUG) cout << "done with initStateInfo()" << endl;

}


void ValueIteration::printStates(){

  for (std::set< std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);

    state_info* info = &(statedata[s]);

    cout << "State " << info->id << ": ";
    for (unsigned j = 0; j < s->size(); j++){
      cout << (*s)[j] << ", ";
    }
    cout << endl;

    for (int act = 0; act < numactions; act++){
      cout << " visits[" << act << "] = " << info->visits[act]
           << " Q: " << info->Q[act]
           << " R: " << info->modelInfo[act].reward << endl;
    }

  }
}




void ValueIteration::calculateReachableStates(){
  // for all states
  // set plausible flag to see if we need to calc q value for this state
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);

    // get state's info
    state_info* info = &(statedata[s]);

    info->stepsAway = 100000;

    //if (info->fresh){
    //  info->stepsAway = 0;
    //  continue;
    //}

    for (int j = 0; j < numactions; j++){
      if (info->visits[j] > 0){
        info->stepsAway = 0;
        break;
      }
    }
  }
}

void ValueIteration::removeUnreachableStates(){

  return;

  // for all states
  // set plausible flag to see if we need to calc q value for this state
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);

    // get state's info
    state_info* info = &(statedata[s]);

    // state is unreachable
    if (info->stepsAway > MAX_STEPS){

      if (POLICYDEBUG){
        cout << "Removing unreachable state: " << (*s)[0];
        for (unsigned i = 1; i < s->size(); i++){
          cout << ", " << (*s)[i];
        }
        cout << endl;
      }

      // delete state
      deleteInfo(info);

      // remove from statespace
      statespace.erase(*s);

      // remove from statedata hashmap
      statedata.erase(s);

    }
  }
}






void ValueIteration::deleteInfo(state_info* info){

  delete [] info->modelInfo;

}


double ValueIteration::getSeconds(){
  struct timezone tz;
  timeval timeT;
  gettimeofday(&timeT, &tz);
  return  timeT.tv_sec + (timeT.tv_usec / 1000000.0);
}


void ValueIteration::savePolicy(const char* filename){

  ofstream policyFile(filename, ios::out | ios::binary | ios::trunc);

  // first part, save the vector size
  int fsize = featmin.size();
  policyFile.write((char*)&fsize, sizeof(int));

  // save numactions
  policyFile.write((char*)&numactions, sizeof(int));

  // go through all states, and save Q values
  for (std::set< std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);
    state_info* info = &(statedata[s]);

    // save state
    policyFile.write((char*)&((*i)[0]), sizeof(float)*fsize);

    // save q-values
    policyFile.write((char*)&(info->Q[0]), sizeof(float)*numactions);

  }

  policyFile.close();
}


// should do it such that an already discretized state stays the same
// mainly the numerical value of each bin should be the average of that bin
std::vector<float> ValueIteration::discretizeState(const std::vector<float> &s){
  std::vector<float> ds(s.size());

  for (unsigned i = 0; i < s.size(); i++){
    
    // since i'm sometimes doing this for discrete domains
    // want to center bins on 0, not edge on 0
    //cout << "feat " << i << " range: " << featmax[i] << " " << featmin[i] << " " << (featmax[i]-featmin[i]) << " n: " << (float)statesPerDim;

    float factor = (featmax[i] - featmin[i]) / (float)statesPerDim[i];
    int bin = 0;
    if (s[i] > 0){
      bin = (int)((s[i]+factor/2) / factor);
    } else {
      bin = (int)((s[i]-factor/2) / factor);
    }

    ds[i] = factor*bin;
    //cout << " factor: " << factor << " bin: " << bin;
    //cout << " Original: " << s[i] << " Discrete: " << ds[i] << endl;
  }

  return ds;
}
