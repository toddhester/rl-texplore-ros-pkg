/** \file ETUCT.cc
    Implements UCT with eligiblity traces. A modified version of UCT 
    as presented in:
    L. Kocsis and C. Szepesv´ari, "Bandit based monte-carlo planning," in
    ECML-06. Number 4212 in LNCS. Springer, 2006, pp. 282-293.
    \author Todd Hester
*/

#include "ETUCT.hh"
#include <algorithm>

#include <sys/time.h>


ETUCT::ETUCT(int numactions, float gamma, float rrange, float lambda,
             int MAX_ITER, float MAX_TIME, int MAX_DEPTH, int modelType,
             const std::vector<float> &fmax, const std::vector<float> &fmin,
             const std::vector<int> &nstatesPerDim, bool trackActual,
             int historySize, Random r):
  numactions(numactions), gamma(gamma), rrange(rrange), lambda(lambda),
  MAX_ITER(MAX_ITER), MAX_TIME(MAX_TIME),
  MAX_DEPTH(MAX_DEPTH), modelType(modelType), statesPerDim(nstatesPerDim),
  trackActual(trackActual), HISTORY_SIZE(historySize),
  HISTORY_FL_SIZE(historySize*numactions)//fmax.size())
{
  rng = r;

  nstates = 0;
  nactions = 0;
  lastUpdate = -1;
  seedMode = false;

  timingType = true;

  model = NULL;
  planTime = getSeconds();

  PLANNERDEBUG = false;//true;
  ACTDEBUG = false; //true;
  MODELDEBUG = false;//true;//false;
  UCTDEBUG = false; //true;//false;
  REALSTATEDEBUG = false;
  HISTORYDEBUG = false; //true; //false;

  featmax = fmax;
  featmin = fmin;

  if (statesPerDim[0] > 0){
    cout << "Planner ETUCT using discretization of " << statesPerDim[0] << endl;
  }
  if (trackActual){
    cout << "ETUCT tracking real state values" << endl;
  }
  cout << "Planner using history size: " << HISTORY_SIZE << endl;

  if (HISTORY_SIZE == 0){
    saHistory.push_back(0.0);
  }
  else {
    if (HISTORYDEBUG) {
      cout << "History size of " << HISTORY_SIZE
           << " float size of " << HISTORY_FL_SIZE
           << " with state size: " << fmin.size()
           << " and numact: " << numactions << endl;
    }
    for (int i = 0; i < HISTORY_FL_SIZE; i++){
      saHistory.push_back(0.0);
    }
  }

  //  initStates();

}

ETUCT::~ETUCT() {
  //cout << "planner delete" << endl;
  // clear all state info

  for (std::map<state_t, state_info>::iterator i = statedata.begin();
       i != statedata.end(); i++){

    // get state's info
    //cout << "  planner got info" << endl;
    state_info* info = &((*i).second);

    deleteInfo(info);
  }

  featmax.clear();
  featmin.clear();

  statespace.clear();
  statedata.clear();
  //cout << "planner done" << endl;
}

void ETUCT::setModel(MDPModel* m){

  model = m;

}

// canonicalize all the states so we already have them in our statespace
void ETUCT::initStates(){
  cout << "init states" << endl;
  std::vector<float> s(featmin.size());

  fillInState(s,0);
}

void ETUCT::fillInState(std::vector<float>s, int depth){

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


void ETUCT::initNewState(state_t s){
  //if (PLANNERDEBUG) cout << "initNewState(s = " << s
  //     << ") size = " << s->size() << endl;

  // create state info and add to hash map
  state_info* info = &(statedata[s]);
  initStateInfo(s, info);

  // dont init any model info
  // we'll get it during search if we have to

}

bool ETUCT::updateModelWithExperience(const std::vector<float> &laststate,
                                      int lastact,
                                      const std::vector<float> &currstate,
                                      float reward, bool term){
  //  if (PLANNERDEBUG) cout << "updateModelWithExperience(last = " << &laststate
  //     << ", curr = " << &currstate
  //        << ", lastact = " << lastact
  //     << ", r = " << reward
  //     << ", term = " << term
  //     << ")" << endl;

  if (!timingType)
    planTime = getSeconds();

  // canonicalize these things
  state_t last = canonicalize(laststate);

  prevstate = last;
  prevact = lastact;

  // get state info
  previnfo = &(statedata[last]);

  // init model?
  if (model == NULL){
    cout << "ERROR IN MODEL OR MODEL SIZE" << endl;
    exit(-1);
  }

  if (MODELDEBUG){
    cout << "Update with exp from state: ";
    for (unsigned i = 0; i < last->size(); i++){
      cout << (laststate)[i] << ", ";
    }
    cout << " action: " << lastact;
    cout << " to state: ";
    for (unsigned i = 0; i < currstate.size(); i++){
      cout << currstate[i] << ", ";
    }
    cout << " and reward: " << reward << endl;
  }

  experience e;
  e.s = laststate;
  e.next = currstate;
  e.act = lastact;
  e.reward = reward;
  e.terminal = term;

  if (HISTORY_SIZE > 0){
    if (HISTORYDEBUG) {
      cout << "Original state vector (size " << e.s.size() << ": " << e.s[0];
      for (unsigned i = 1; i < e.s.size(); i++){
        cout << "," << e.s[i];
      }
      cout << endl;
    }
    // add history onto e.s
    for (int i = 0; i < HISTORY_FL_SIZE; i++){
      e.s.push_back(saHistory[i]);
    }

    if (HISTORYDEBUG) {
      cout << "New state vector (size " << e.s.size() << ": " << e.s[0];
      for (unsigned i = 1; i < e.s.size(); i++){
        cout << "," << e.s[i];
      }
      cout << endl;
    }

    if (!seedMode){
      // push this state and action onto the history vector
      /*
      for (unsigned i = 0; i < last->size(); i++){
        saHistory.push_back((*last)[i]);
        saHistory.pop_front();
      }
      */
      
      for (int i = 0; i < numactions; i++){
        if (i == lastact)
          saHistory.push_back(1.0);
        else
          saHistory.push_back(0.0);
        saHistory.pop_front();
      }
      
      if (HISTORYDEBUG) {
        cout << "New history vector (size " << saHistory.size() << ": " << saHistory[0];
        for (unsigned i = 1; i < saHistory.size(); i++){
          cout << "," << saHistory[i];
        }
        cout << endl;
      }
    }
  }

  bool modelChanged = model->updateWithExperience(e);

  if (timingType)
    planTime = getSeconds();

  return modelChanged;

}

void ETUCT::updateStateActionFromModel(state_t s, int a, state_info* info){

  if (HISTORY_SIZE == 0){

    std::deque<float> history(1,0.0);
    StateActionInfo* newModel = NULL;
    newModel = &(info->historyModel[a][history]);

    updateStateActionHistoryFromModel(*s, a, newModel);

  }

  else {

    // fill in for all histories???
    for (std::map< std::deque<float>, StateActionInfo>::iterator it = info->historyModel[a].begin(); it != info->historyModel[a].end(); it++){

      std::deque<float> oneHist = (*it).first;
      StateActionInfo* newModel = &((*it).second);

      // add history to vector
      std::vector<float> modState = *s;
      for (int i = 0; i < HISTORY_FL_SIZE; i++){
        modState.push_back(oneHist[i]);
      }
      updateStateActionHistoryFromModel(modState, a, newModel);
    }

  }

}

void ETUCT::updateStateActionHistoryFromModel(const std::vector<float> modState, int a, StateActionInfo *newModel){

  // update state info
  // get state action info for each action
  model->getStateActionInfo(modState, a, newModel);
  newModel->frameUpdated = nactions;

  //canonNextStates(newModel);

}



void ETUCT::canonNextStates(StateActionInfo* modelInfo){

  // loop through all next states
  for (std::map<std::vector<float>, float>::iterator outIt
         = modelInfo->transitionProbs.begin();
       outIt != modelInfo->transitionProbs.end(); outIt++){

    std::vector<float> nextstate = (*outIt).first;

    // check that it is valid, otherwise replace with current
    bool badState = false;
    for (unsigned j = 0; j < nextstate.size(); j++){
      if (nextstate[j] < (featmin[j]-EPSILON)
          || nextstate[j] > (featmax[j]+EPSILON)){
        //cout << "next state out of range " << nextstate[j] << endl;
        badState = true;
        break;
      }
    }

    if (!badState){
      canonicalize(nextstate);
    }
  }
}




int ETUCT::getBestAction(const std::vector<float> &state){
  //  if (PLANNERDEBUG) cout << "getBestAction(s = " << &state << ")" << endl;

  //  resetUCTCounts();

  state_t s = canonicalize(state);

  int i = 0;
  for (i = 0; i < MAX_ITER; i++){

    std::deque<float> searchHistory = saHistory;
    uctSearch(state, s, 0, searchHistory);

    // break after some max time
    if ((getSeconds() - planTime) > MAX_TIME){ // && i > 500){
      break;
    }

  }
  double currTime = getSeconds();
  if (false || UCTDEBUG){
    cout << "Search complete after " << (currTime-planTime) << " seconds and "
         << i << " iterations." << endl;
  }

  // get state info
  state_info* info = &(statedata[s]);

  // Get Q values
  std::vector<float> &Q = info->Q;

  // Choose an action
  const std::vector<float>::iterator a =
    random_max_element(Q.begin(), Q.end()); // Choose maximum

  int act = a - Q.begin();
  nactions++;

  if (false){
    cout << "State " << (*s)[0];
    for (unsigned i = 1; i < s->size(); i++){
      cout << "," << (*s)[i];
    }
    cout << ", Took action " << act << ", "
         << "value: " << *a << endl;
  }

  // return index of action
  return act;
}






void ETUCT::planOnNewModel(){

  // reset visit counts/q values
  resetUCTCounts();

  // for rmax, only s-a's prediction has changed
  if (modelType == RMAX){
    updateStateActionFromModel(prevstate, prevact, previnfo);
  }

  // for other model types, it all could change, clear all cached model predictions
  else {

    // still update flagged s-a's
    // then less stuff to query while planning
    for (std::set<std::vector<float> >::iterator i = statespace.begin();
         i != statespace.end(); i++){
      state_t s = canonicalize(*i);
      state_info* info = &(statedata[s]);
      if (info->needsUpdate){
        for (int j = 0; j < numactions; j++){
          updateStateActionFromModel(s, j, info);
        }
        info->needsUpdate = false;
      }
    }
    lastUpdate = nactions;
  }

}


void ETUCT::resetUCTCounts(){
  // if (PLANNERDEBUG) cout << "Reset UCT Counts" << endl;
  const int MIN_VISITS = 10;

  // loop through here
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){
    state_t s = canonicalize(*i);

    state_info* info = &(statedata[s]);

    if (info->uctVisits > (MIN_VISITS * numactions))
      info->uctVisits = MIN_VISITS * numactions;

    for (int j = 0; j < numactions; j++){
      if (info->uctActions[j] > MIN_VISITS)
        info->uctActions[j] = MIN_VISITS;
    }

  }

}




////////////////////////////
// Helper Functions       //
////////////////////////////

ETUCT::state_t ETUCT::canonicalize(const std::vector<float> &s) {
  //if (PLANNERDEBUG) cout << "canonicalize(s = " << s[0] << ", "
  //                     << s[1] << ")" << endl;

  // discretize it
  std::vector<float> s2;
  if (statesPerDim[0] > 0){
    s2 = discretizeState(s);
  } else {
    s2 = s;
  }

  // get state_t for pointer if its in statespace
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s2);
  state_t retval = &*result.first; // Dereference iterator then get pointer

  //if (PLANNERDEBUG) cout << " returns " << retval
  //       << " New: " << result.second << endl;

  // if not, init this new state
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    initNewState(retval);
    if (PLANNERDEBUG) {
      cout << " New state initialized "
           << " orig:(" << s[0] << "," << s[1] << ")"
           << " disc:(" << s2[0] << "," << s2[1] << ")" << endl;
    }
  }


  return retval;
}


// init state info
void ETUCT::initStateInfo(state_t s, state_info* info){
  //if (PLANNERDEBUG) cout << "initStateInfo()";

  info->id = nstates++;
  if (PLANNERDEBUG){
    cout << " id = " << info->id;
    cout << ", (" << (*s)[0] << "," << (*s)[1] << ")" << endl;
  }

  info->historyModel = new std::map< std::deque<float>, StateActionInfo>[numactions];

  // model q values, visit counts
  info->Q.resize(numactions, 0);
  info->uctActions.resize(numactions, 1);
  info->uctVisits = 1;
  info->visited = 0; //false;
  info->needsUpdate = true;

  for (int i = 0; i < numactions; i++){
    info->Q[i] = rng.uniform(0,0.01);
  }

  //if (PLANNERDEBUG) cout << "done with initStateInfo()" << endl;

}


void ETUCT::printStates(){

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
      cout << " Q: " << info->Q[act] << endl;
    }

  }
}


void ETUCT::deleteInfo(state_info* info){

  delete [] info->historyModel;

}



double ETUCT::getSeconds(){
  struct timezone tz;
  timeval timeT;
  gettimeofday(&timeT, &tz);
  return  timeT.tv_sec + (timeT.tv_usec / 1000000.0);
}


float ETUCT::uctSearch(const std::vector<float> &actS, state_t discS, int depth,std::deque<float> searchHistory){
  if (UCTDEBUG){
    cout << " uctSearch state ";
    for (unsigned i = 0; i < actS.size(); i++){
      cout << actS[i] << ", ";
    }
    cout << " at depth " << depth << endl;
  }

  state_info* info = &(statedata[discS]);

  // if max depth
  // iterative deepening (probability inversely proportional to visits)
  //float terminateProb = 1.0/(2.0+(float)info->uctVisits);

  // already visited, stop here
  if (depth > MAX_DEPTH){
    // return max q value here
    std::vector<float>::iterator maxAct =
      std::max_element(info->Q.begin(),
                       info->Q.end());
    float maxval = *maxAct;

    if (UCTDEBUG)
      cout << "Terminated after depth: " << depth
        //   << " prob: " << terminateProb
           << " Q: " << maxval
           << " visited: " << info->visited << endl;

    return maxval;
  }

  // select action
  int action = selectUCTAction(info);

  // simulate action to get next state and reward
  // depending on exploration, may also terminate us
  float reward = 0;
  bool term = false;

  float learnRate;
  //float learnRate = 0.001;
  //float learnRate = 1.0 / info->uctActions[action];
  //    learnRate = 10.0 / (info->uctActions[action] + 100.0);
  learnRate = 10.0 / (info->uctActions[action] + 10.0);
  //if (learnRate < 0.001 && MAX_TIME < 0.5)
  //learnRate = 0.001;
  //learnRate = 0.05;
  //learnRate = 1.0;

  // tell model learning thread to update this state since we've visited it
  info->needsUpdate = true;

  // simulate next state, reward, terminal
  std::vector<float> actualNext = simulateNextState(actS, discS, info, searchHistory, action, &reward, &term);

  // simulate reward from this action
  if (term){
    // this one terminated
    if (UCTDEBUG) cout << "   Terminated on exploration condition" << endl;
    info->Q[action] += learnRate * (reward - info->Q[action]);
    info->uctVisits++;
    info->uctActions[action]++;
    if (UCTDEBUG)
      cout << " Depth: " << depth << " Selected action " << action
           << " r: " << reward
           << " StateVisits: " << info->uctVisits
           << " ActionVisits: " << info->uctActions[action] << endl;

    return reward;
  }

  // get discretized version of next
  state_t discNext = canonicalize(actualNext);

  if (UCTDEBUG)
    cout << " Depth: " << depth << " Selected action " << action
         << " r: " << reward  << endl;

  info->visited++; // = true;

  if (HISTORY_SIZE > 0){
    // update history vector for this state
    /*
    for (unsigned i = 0; i < (*discS).size(); i++){
      searchHistory.push_back((*discS)[i]);
      searchHistory.pop_front();
    }
    */
    for (int i = 0; i < numactions; i++){
      if (i == action)
        searchHistory.push_back(1.0);
      else
        searchHistory.push_back(0.0);
      searchHistory.pop_front();
    }
    
    if (HISTORYDEBUG) {
      cout << "New planning history vector (size " << searchHistory.size() << ": " << searchHistory[0];
      for (unsigned i = 1; i < searchHistory.size(); i++){
        cout << "," << searchHistory[i];
      }
      cout << endl;
    }
  }


  // new q value
  float newQ = reward + gamma * uctSearch(actualNext, discNext, depth+1, searchHistory);

  if (info->visited == 1){

    // update q and visit counts
    info->Q[action] += learnRate * (newQ - info->Q[action]);
    info->uctVisits++;
    info->uctActions[action]++;

    if (UCTDEBUG)
      cout << " Depth: " << depth << " newQ: " << newQ
           << " StateVisits: " << info->uctVisits
           << " ActionVisits: " << info->uctActions[action] << endl;

    if (lambda < 1.0){

      // new idea, return max of Q or new q
      std::vector<float>::iterator maxAct =
        std::max_element(info->Q.begin(),
                         info->Q.end());
      float maxval = *maxAct;

      if (UCTDEBUG)
        cout << " Replacing newQ: " << newQ;

      // replace with w avg of maxq and new val
      newQ = (lambda * newQ) + ((1.0-lambda) * maxval);

      if (UCTDEBUG)
        cout << " with wAvg: " << newQ << endl;
    }

  }

  info->visited--;

  // return q
  return newQ;

}


int ETUCT::selectUCTAction(state_info* info){
  //  if (UCTDEBUG) cout << "  selectUCTAction" << endl;

  std::vector<float> &Q = info->Q;

  // loop through
  float rewardBound = rrange;
  if (rewardBound < 1.0)
    rewardBound = 1.0;
  rewardBound /= (1.0 - gamma);
  if (UCTDEBUG) cout << "Reward bound: " << rewardBound << endl;

  std::vector<float> uctQ(numactions, 0.0);

  for (int i = 0; i < numactions; i++){

    // this actions value is Q + rMax * 2 sqrt (log N(s) / N(s,a))
    uctQ[i] = Q[i] +
      rewardBound * 2.0 * sqrt(log((float)info->uctVisits) /
                               (float)info->uctActions[i]);

    if (UCTDEBUG)
      cout << "  Action: " << i << " Q: " << Q[i]
           << " visits: " << info->uctActions[i]
           << " value: " << uctQ[i] << endl;
  }

  // max element of uctQ
  std::vector<float>::iterator maxAct =
    max_element(uctQ.begin(), uctQ.end());
  float maxval = *maxAct;
  int act = maxAct - uctQ.begin();

  if (UCTDEBUG)
    cout << "  Selected " << act << " val: " << maxval << endl;

  return act;

}



std::vector<float> ETUCT::simulateNextState(const std::vector<float> &actualState, state_t discState, state_info* info, const std::deque<float> &history, int action, float* reward, bool* term){

  StateActionInfo* modelInfo = &(info->historyModel[action][history]);
  bool upToDate = modelInfo->frameUpdated >= lastUpdate;

  if (!upToDate){
    // must put in appropriate history
    if (HISTORY_SIZE > 0){
      std::vector<float> modState = *discState;
      for (int i = 0; i < HISTORY_FL_SIZE; i++){
        modState.push_back(history[i]);
      }
      updateStateActionHistoryFromModel(modState, action, modelInfo);
    } else {
      updateStateActionHistoryFromModel(*discState, action, modelInfo);
    }
  }


  *reward = modelInfo->reward;
  *term = (rng.uniform() < modelInfo->termProb);

  if (*term){
    return actualState;
  }

  float randProb = rng.uniform();

  float probSum = 0.0;
  std::vector<float> nextstate;

  if (REALSTATEDEBUG) cout << "randProb: " << randProb << " numNext: " << modelInfo->transitionProbs.size() << endl;

  if (modelInfo->transitionProbs.size() == 0)
    nextstate = actualState;

  for (std::map<std::vector<float>, float>::iterator outIt
         = modelInfo->transitionProbs.begin();
       outIt != modelInfo->transitionProbs.end(); outIt++){

    float prob = (*outIt).second;
    probSum += prob;
    if (REALSTATEDEBUG) cout << randProb << ", " << probSum << ", " << prob << endl;

    if (randProb <= probSum){
      nextstate = (*outIt).first;
      if (REALSTATEDEBUG) cout << "selected state " << randProb << ", " << probSum << ", " << prob << endl;
      break;
    }
  }

  if (trackActual){


    // find the relative change from discrete center
    std::vector<float> relChange = subVec(nextstate, *discState);

    // add that on to actual current state value
    nextstate = addVec(actualState, relChange);


  }

  // check that next state is valid
  for (unsigned j = 0; j < nextstate.size(); j++){
    if (nextstate[j] < (featmin[j]-EPSILON)
        || nextstate[j] > (featmax[j]+EPSILON)){
      return actualState;
    }
  }

  if (UCTDEBUG) cout << "predicted next state: " << nextstate[0] << ", " << nextstate[1] << endl;

  // return new actual state
  return nextstate;

}


void ETUCT::savePolicy(const char* filename){

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

void ETUCT::logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax){
  std::vector<float> state(2, 0.0);
  for (int i = xmin ; i < xmax; i++){
    for (int j = ymin; j < ymax; j++){
      state[0] = j;
      state[1] = i;
      state_t s = canonicalize(state);
      state_info* info = &(statedata[s]);
      std::vector<float> &Q_s = info->Q;
      const std::vector<float>::iterator max =
        random_max_element(Q_s.begin(), Q_s.end());
      *of << (*max) << ",";
    }
  }
}


// should do it such that an already discretized state stays the same
// mainly the numerical value of each bin should be the average of that bin
std::vector<float> ETUCT::discretizeState(const std::vector<float> &s){
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
    //cout << "P factor: " << factor << " bin: " << bin;
    //cout << " Original: " << s[i] << " Discrete: " << ds[i] << endl;
  }

  return ds;
}

std::vector<float> ETUCT::addVec(const std::vector<float> &a, const std::vector<float> &b){
  if (a.size() != b.size())
    cout << "ERROR: add vector sizes wrong " << a.size() << ", " << b.size() << endl;

  std::vector<float> c(a.size(), 0.0);
  for (unsigned i = 0; i < a.size(); i++){
    c[i] = a[i] + b[i];
  }

  return c;
}

std::vector<float> ETUCT::subVec(const std::vector<float> &a, const std::vector<float> &b){
  if (a.size() != b.size())
    cout << "ERROR: sub vector sizes wrong " << a.size() << ", " << b.size() << endl;

  std::vector<float> c(a.size(), 0.0);
  for (unsigned i = 0; i < a.size(); i++){
    c[i] = a[i] - b[i];
  }

  return c;
}


void ETUCT::setFirst(){
  if (HISTORY_SIZE == 0) return;

  if (HISTORYDEBUG) cout << "first action, set sahistory to 0s" << endl;

  // first action, reset history vector
  saHistory.resize(saHistory.size(), 0.0);
}

void ETUCT::setSeeding(bool seeding){

  if (HISTORYDEBUG) cout << "set seed mode to " << seeding << endl;
  seedMode = seeding;

}
