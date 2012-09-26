/** \file ParallelETUCT.cc
    Implements my real-time model-based RL architecture which uses UCT with eligiblity traces for planning.
    The modified version of UCT used is presented in:
    L. Kocsis and C. Szepesv´ari, "Bandit based monte-carlo planning," in
    ECML-06. Number 4212 in LNCS. Springer, 2006, pp. 282-293.
    The real-time architecture is presented in:
    Hester, Quinlan, and Stone, "A Real-Time Model-Based Reinforcement Learning Architecture for Robot Control", arXiv 1105.1749, 2011.
    \author Todd Hester
*/

#include "ParallelETUCT.hh"
#include <algorithm>

#include <sys/time.h>


ParallelETUCT::ParallelETUCT(int numactions, float gamma, float rrange, float lambda,
                             int MAX_ITER, float MAX_TIME, int MAX_DEPTH, int modelType,
                             const std::vector<float> &fmax, const std::vector<float> &fmin,
                             const std::vector<int> &nstatesPerDim, bool trackActual, int historySize, Random r):
  numactions(numactions), gamma(gamma), rrange(rrange), lambda(lambda),
  MAX_ITER(MAX_ITER), MAX_TIME(MAX_TIME),
  MAX_DEPTH(MAX_DEPTH), modelType(modelType), statesPerDim(nstatesPerDim),
  trackActual(trackActual), HISTORY_SIZE(historySize),
  HISTORY_FL_SIZE(historySize*numactions),
  CLEAR_SIZE(25)
{
  rng = r;

  nstates = 0;
  nsaved = 0;
  nactions = 0;
  lastUpdate = -1;

  seedMode = false;
  timingType = true;

  previnfo = NULL;
  model = NULL;
  planTime = getSeconds();
  initTime = getSeconds();
  setTime = getSeconds();

  PLANNERDEBUG = false;
  POLICYDEBUG = false; //true; //false; //true; //false;
  ACTDEBUG = false; //true;
  MODELDEBUG = false;
  UCTDEBUG = false;
  PTHREADDEBUG = false;
  ATHREADDEBUG = false;//true;
  MTHREADDEBUG = false; //true;
  TIMINGDEBUG = false;
  REALSTATEDEBUG = false;
  HISTORYDEBUG = false; //true;

  if (statesPerDim[0] > 0){
    cout << "Planner Parallel ETUCT using discretization of " << statesPerDim[0] << endl;
  }
  if (trackActual){
    cout << "Parallel ETUCT tracking real state values" << endl;
  }
  cout << "Planner using history size: " << HISTORY_SIZE << endl;

  featmax = fmax;
  featmin = fmin;

  pthread_mutex_init(&update_mutex, NULL);
  pthread_mutex_init(&history_mutex, NULL);
  pthread_mutex_init(&nactions_mutex, NULL);
  pthread_mutex_init(&plan_state_mutex, NULL);
  pthread_mutex_init(&statespace_mutex, NULL);
  pthread_mutex_init(&model_mutex, NULL);
  pthread_mutex_init(&list_mutex, NULL);
  pthread_cond_init(&list_cond, NULL);

  // start parallel search thread
  actualPlanState = std::vector<float>(featmax.size());
  discPlanState = NULL;
  doRandom = true;
  modelThreadStarted = false;
  planThreadStarted = false;
  expList.clear();

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
  expfile.initFile("experiences.bin", featmax.size());
}

ParallelETUCT::~ParallelETUCT() {
  // join threads

  //pthread_kill(planThread);
  //pthread_kill(modelThread);

  pthread_detach(planThread);//, NULL);
  pthread_detach(modelThread);//, NULL);

  pthread_cancel(planThread);//, NULL);
  pthread_cancel(modelThread);//, NULL);

  //pthread_join(planThread, NULL);
  //pthread_join(modelThread, NULL);

  //pthread_detach(planThread);//, NULL);
  //pthread_detach(modelThread);//, NULL);


  pthread_mutex_lock(&plan_state_mutex);
  pthread_mutex_lock(&statespace_mutex);
  pthread_mutex_lock(&model_mutex);
  pthread_mutex_lock(&list_mutex);

  // delete exp list
  expList.clear();

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

  pthread_mutex_unlock(&plan_state_mutex);
  pthread_mutex_unlock(&statespace_mutex);
  pthread_mutex_unlock(&model_mutex);
  pthread_mutex_unlock(&list_mutex);

}

void ParallelETUCT::setModel(MDPModel* m){

  model = m;

}


/////////////////////////////
// Functional functions :) //
/////////////////////////////



/** Use the latest experience to update state info and the model. */
bool ParallelETUCT::updateModelWithExperience(const std::vector<float> &laststate,
                                              int lastact,
                                              const std::vector<float> &currstate,
                                              float reward, bool term){
  //  if (PLANNERDEBUG) cout << "updateModelWithExperience(last = " << &laststate
  //     << ", curr = " << &currstate
  //        << ", lastact = " << lastact
  //     << ", r = " << reward
  //     << ", term = " << term
  //     << ")" << endl;

  //cout << "updateModel" << endl << flush;

  if (!timingType)
    planTime = getSeconds();
  initTime = getSeconds();

  // canonicalize these things
  state_t last = canonicalize(laststate);

  prevstate = last;
  prevact = lastact;

  // get state info
  pthread_mutex_lock(&statespace_mutex);
  previnfo = &(statedata[last]);
  pthread_mutex_unlock(&statespace_mutex);

  if (MODELDEBUG){
    cout << "Update with exp from state: ";
    for (unsigned i = 0; i < last->size(); i++){
      cout << (laststate)[i] << ", ";
    }
    cout << " action: " << lastact;
    cout << " to state: ";
    for (unsigned i = 0; i < currstate.size(); i++){
      cout << (currstate)[i] << ", ";
    }
    cout << " and reward: " << reward << endl;
  }

  // add experiences to list to later be updated into model
  if (ATHREADDEBUG)
    cout << "*** Action thread wants list lock ***" << endl << flush;
  if (TIMINGDEBUG) cout << "Want list mutex, time: " << (getSeconds()-initTime) << endl;
  pthread_mutex_lock(&list_mutex);
  if (TIMINGDEBUG) cout << "got list mutex, time: " << (getSeconds()-initTime) << endl;
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
    pthread_mutex_lock(&history_mutex);
    for (int i = 0; i < HISTORY_FL_SIZE; i++){
      e.s.push_back(saHistory[i]);
    }
    pthread_mutex_unlock(&history_mutex);

    if (HISTORYDEBUG) {
      cout << "New state vector (size " << e.s.size() << ": " << e.s[0];
      for (unsigned i = 1; i < e.s.size(); i++){
        cout << "," << e.s[i];
      }
      cout << endl;
    }

    if (!seedMode){
      pthread_mutex_lock(&history_mutex);
      // push this state and action onto the history vector
      /*
        for (unsigned i = 0; i < last->size(); i++){
        saHistory.push_back((*last)[i]);
        saHistory.pop_front();
        }

        saHistory.push_back((*last)[3]);
        saHistory.pop_front();
      */
      for (int i = 0; i < numactions; i++){
        if (i == lastact)
          saHistory.push_back(1.0);
        else
          saHistory.push_back(0.0);
        saHistory.pop_front();
      }

      //      saHistory.push_back(lastact);
      //saHistory.pop_front();
      if (HISTORYDEBUG) {
        cout << "New history vector (size " << saHistory.size() << ": " << saHistory[0];
        for (unsigned i = 1; i < saHistory.size(); i++){
          cout << "," << saHistory[i];
        }
        cout << endl;
      }
      pthread_mutex_unlock(&history_mutex);
    }
  }

  expList.push_back(e);
  //expfile.saveExperience(e);
  if (ATHREADDEBUG || MTHREADDEBUG)
    cout << "added exp to list, size: " << expList.size() << endl << flush;
  if (TIMINGDEBUG) cout << "list updated, time: " << (getSeconds()-initTime) << endl;
  pthread_cond_signal(&list_cond);
  pthread_mutex_unlock(&list_mutex);

  /*
    if (e.reward > -0.5 && e.reward < 0){
    expfile.saveExperience(e);
    nsaved++;
    cout << "Saved Experience " << e.reward << endl;
    }
  */

  if (timingType)
    planTime = getSeconds();

  if (TIMINGDEBUG) cout << "leaving updateModel, time: " << (getSeconds()-initTime) << endl;


  return false;

}

/** Update a single state-action from the model */
void ParallelETUCT::updateStateActionFromModel(state_t s, int a, state_info* info){

  if (HISTORY_SIZE == 0){
    pthread_mutex_lock(&info->statemodel_mutex);

    std::deque<float> history(1,0.0);
    StateActionInfo* newModel = NULL;
    newModel = &(info->historyModel[a][history]);

    updateStateActionHistoryFromModel(*s, a, newModel);
    pthread_mutex_unlock(&info->statemodel_mutex);

  }

  else {
    pthread_mutex_lock(&info->statemodel_mutex);

    // clear large ones
    if (info->historyModel[a].size() > CLEAR_SIZE){

      //      cout << "clearing model map of size " << info->historyModel[a].size() << endl;

      // instead, clear because these take too much memory to keep around
      info->historyModel[a].clear();

    } else {

      // fill in for all histories???
      for (std::map< std::deque<float>, StateActionInfo>::iterator it = info->historyModel[a].begin();
           it != info->historyModel[a].end(); it++){

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

    pthread_mutex_unlock(&info->statemodel_mutex);
  }

}

/** Update a single state-action from the model */
void ParallelETUCT::updateStateActionHistoryFromModel(const std::vector<float> modState, int a, StateActionInfo *newModel){

  // update state info
  // get state action info for each action
  pthread_mutex_lock(&model_mutex);

  model->getStateActionInfo(modState, a, newModel);

  pthread_mutex_lock(&nactions_mutex);
  newModel->frameUpdated = nactions;
  pthread_mutex_unlock(&nactions_mutex);

  pthread_mutex_unlock(&model_mutex);

  //canonNextStates(newModel);

}

void ParallelETUCT::canonNextStates(StateActionInfo* modelInfo){


  // loop through all next states
  for (std::map<std::vector<float>, float>::iterator outIt
         = modelInfo->transitionProbs.begin();
       outIt != modelInfo->transitionProbs.end(); outIt++){

    std::vector<float> nextstate = (*outIt).first;
    bool badState = false;

    // check that it is valid, otherwise replace with current
    for (unsigned j = 0; j < nextstate.size(); j++){
      float factor = EPSILON;
      if (statesPerDim[j] > 0)
        factor = (featmax[j] - featmin[j]) / (float)statesPerDim[j];
      if (nextstate[j] < (featmin[j]-factor)
          || nextstate[j] > (featmax[j]+factor)){
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




/** Choose the next action */
int ParallelETUCT::getBestAction(const std::vector<float> &state){
  //  if (PLANNERDEBUG) cout << "getBestAction(s = " << &state << ")" << endl;

  pthread_mutex_lock(&nactions_mutex);
  nactions++;
  pthread_mutex_unlock(&nactions_mutex);

  if (TIMINGDEBUG) cout << "getBestAction, time: " << (getSeconds()-initTime) << endl;


  state_t s = canonicalize(state);

  // set plan state so uct will search from here
  if (ATHREADDEBUG)
    cout << "*** Action thread wants plan state lock ***" << endl << flush;
  if (TIMINGDEBUG) cout << "want planStateMut, time: " << (getSeconds()-initTime) << endl;

  pthread_mutex_lock(&(plan_state_mutex));
  if (TIMINGDEBUG) cout << "got planStateMut, time: " << (getSeconds()-initTime) << endl;

  doRandom = false;
  actualPlanState = state;
  discPlanState = s;
  setTime = getSeconds();

  if (ATHREADDEBUG){
    cout << "Set planning state as: ";
    for (unsigned i = 0; i < state.size(); i++){
      cout << state[i] << ", ";
    }
    cout << endl << flush;
  }

  // call uct search on it
  pthread_mutex_unlock(&(plan_state_mutex));
  if (TIMINGDEBUG) cout << "set planState, time: " << (getSeconds()-initTime) << endl;

  // get state info
  pthread_mutex_lock(&statespace_mutex);
  state_info* info = &(statedata[s]);
  pthread_mutex_unlock(&statespace_mutex);

  // wait a bit for some planning from this state

  // depending on how you run the code, this has to be setup differently
  // if someone else calls this method at the appropriate rate, do nothing here

  // or this can be where we wait to ensure we run at some rate:
  while (((getSeconds()- initTime) < MAX_TIME)){
    if (TIMINGDEBUG)
      cout << "waiting for time: " << (getSeconds()-initTime) << endl;

    pthread_yield();
  }

  if (TIMINGDEBUG) cout << "time up: " << (getSeconds()-initTime) << endl;

  if (TIMINGDEBUG && (getSeconds()-initTime) > 0.15) cout << "**********" << endl;

  pthread_mutex_lock(&info->stateinfo_mutex);

  // Get Q values
  std::vector<float> &Q = info->Q;


  if (ATHREADDEBUG) {
    if (previnfo != NULL)
      cout << " ... now " << previnfo->uctVisits << " times." << endl;
    cout << "Getting best action from state ";
    for (unsigned i = 0; i < s->size(); i++){
      cout << (*s)[i] << ", ";
    }
    cout << " sampled " << info->uctVisits << " times.";// << endl << flush;
  }

  // Choose an action
  const std::vector<float>::iterator a =
    random_max_element(Q.begin(), Q.end()); // Choose maximum
  int act = a - Q.begin();

  if (TIMINGDEBUG) cout << "got action: " << (getSeconds()-initTime) << endl;

  pthread_mutex_unlock(&info->stateinfo_mutex);

  // return index of action
  return act;
}






void ParallelETUCT::planOnNewModel(){
  //return;
  //  cout << "planOnNewModel" << endl << flush;
  // start model learning thread here
  if (!modelThreadStarted){
    modelThreadStarted = true;
    pthread_create(&modelThread, NULL, parallelModelLearningStart, this);
  }

  if (!planThreadStarted){
    planThreadStarted = true;
    pthread_create(&(planThread), NULL, parallelSearchStart, this);
  }

}


void* parallelModelLearningStart(void* arg){
  cout << "Start model learning thread" << endl << flush;
  ParallelETUCT* pe = reinterpret_cast<ParallelETUCT*>(arg);
  while(true){
    pe->parallelModelLearning();
    /*
      if (!pe->planThreadStarted){
      pe->planThreadStarted = true;
      pthread_create(&(pe->planThread), NULL, parallelSearchStart, pe);
      }
    */
  }
  return NULL;
}

void ParallelETUCT::parallelModelLearning(){
  //while(true){

  // wait for experience list to be non-empty
  pthread_mutex_lock(&list_mutex);
  while (expList.size() == 0){
    pthread_cond_wait(&list_cond,&list_mutex);
  }
  pthread_mutex_unlock(&list_mutex);

  // copy over experience list
  std::vector<experience> updateList;
  if (MTHREADDEBUG) cout << "  *** Model thread wants list lock ***" << endl << flush;
  pthread_mutex_lock(&list_mutex);
  updateList = expList;
  expList.clear();
  if (MTHREADDEBUG) cout << "  *** Model thread done with list lock ***" << endl << flush;
  pthread_mutex_unlock(&list_mutex);

  /*
  // update model
  //cout << "*** Model thread wants tree lock ***" << endl << flush;
  pthread_mutex_lock(&model_mutex);
  if (MTHREADDEBUG) cout << "  Model thread: going to update model with " << updateList.size() << " new experiences" << endl << flush;
  //cout << "****update tree with " << updateList.size() << endl << flush;
  bool modelChanged = model->updateWithExperiences(updateList);
  if (MTHREADDEBUG) cout << "  Model updated" << endl << flush;
  pthread_mutex_unlock(&model_mutex);
  */

  modelcopy = model->getCopy();
  //if (COPYDEBUG) cout << "*** PO: model copied" << endl;

  // update model copy with new experience
  bool modelChanged = modelcopy->updateWithExperiences(updateList);

  // set model pointer to point at copy, delete original model                    cout << "acquire model_mutex for update" << endl;
  pthread_mutex_lock(&model_mutex);
  //cout << "model_mutex acquired for update" << endl;
  //if (COPYDEBUG) cout << "*** PO: delete original model and change pointer" << endl;
  delete model;
  model = modelcopy;
  if (MTHREADDEBUG) cout << "  Model updated" << endl << flush;
  //if (COPYDEBUG) cout << "*** PO: pointer set to updated model copy" << endl;
  pthread_mutex_unlock(&model_mutex);



  // if it changed, reset counts, update state actions
  if (modelChanged) resetAndUpdateStateActions();

  pthread_yield();

  //}// while loop
} // method


void ParallelETUCT::setBetweenEpisodes(){
  // TODO: for now, I know this means we just ended an episode, lets plan
  // from a different random state (or a state we know to be the initial one)
  if (ATHREADDEBUG) cout << "*** Action thread wants planning state lock (bet eps)***" << endl;
  pthread_mutex_lock(&(plan_state_mutex));

  doRandom = true;
  discPlanState = NULL;

  // call uct search on it
  pthread_mutex_unlock(&(plan_state_mutex));

}




void ParallelETUCT::resetAndUpdateStateActions(){
  //cout << "*** Model changed, updating state actions ***" << endl << flush;
  const int MIN_VISITS = 10;

  pthread_mutex_lock(&nactions_mutex);
  int updateTime = nactions;
  pthread_mutex_unlock(&nactions_mutex);

  // loop through here

  pthread_mutex_lock(&statespace_mutex);

  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){
    pthread_mutex_unlock(&statespace_mutex);

    state_t s = canonicalize(*i);

    if (MTHREADDEBUG) cout << "  *** Model thread wants search lock ***" << endl;

    if (MTHREADDEBUG) cout << "  *** Model thread got search lock " << endl;

    pthread_mutex_lock(&statespace_mutex);
    state_info* info = &(statedata[s]);
    pthread_mutex_unlock(&statespace_mutex);

    pthread_mutex_lock(&info->stateinfo_mutex);

    if (info->uctVisits > (MIN_VISITS * numactions))
      info->uctVisits = MIN_VISITS * numactions;

    for (int j = 0; j < numactions; j++){
      if (info->uctActions[j] > MIN_VISITS)
        info->uctActions[j] = MIN_VISITS;
      if (info->needsUpdate || info->historyModel[j].size() > CLEAR_SIZE){
        updateStateActionFromModel(s, j, info);
      }
    }
    info->needsUpdate = false;
    pthread_mutex_unlock(&info->stateinfo_mutex);

    pthread_yield();

    pthread_mutex_lock(&statespace_mutex);

  }
  pthread_mutex_unlock(&statespace_mutex);

  pthread_mutex_lock(&update_mutex);
  lastUpdate = updateTime;
  pthread_mutex_unlock(&update_mutex);

}




////////////////////////////
// Helper Functions       //
////////////////////////////

ParallelETUCT::state_t ParallelETUCT::canonicalize(const std::vector<float> &s) {
  if (PLANNERDEBUG) cout << "canonicalize(s = " << s[0] << ", "
                         << s[1] << ")" << endl;

  // discretize it
  std::vector<float> s2;
  if (statesPerDim[0] > 0){
    s2 = discretizeState(s);
  } else {
    s2 = s;
  }

  pthread_mutex_lock(&statespace_mutex);

  // get state_t for pointer if its in statespace
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s2);
  state_t retval = &*result.first; // Dereference iterator then get pointer


  // if not, init this new state
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    state_info* info = &(statedata[retval]);
    int id = nstates++;
    pthread_mutex_unlock(&statespace_mutex);
    initStateInfo(retval, info, id);
  } else {
    pthread_mutex_unlock(&statespace_mutex);
  }

  return retval;
}


// init state info
void ParallelETUCT::initStateInfo(state_t s, state_info* info, int id){
  //if (PLANNERDEBUG) cout << "initStateInfo()";

  // init mutex's for this state info
  pthread_mutex_init(&info->statemodel_mutex, NULL);
  pthread_mutex_init(&info->stateinfo_mutex, NULL);

  pthread_mutex_lock(&info->stateinfo_mutex);

  // model data (transition, reward, known)

  pthread_mutex_lock(&info->statemodel_mutex);
  info->historyModel = new std::map< std::deque<float>, StateActionInfo>[numactions];
  pthread_mutex_unlock(&info->statemodel_mutex);

  info->id = id;
  if (PLANNERDEBUG) cout << " id = " << info->id << endl;

  // model q values, visit counts
  info->Q.resize(numactions, 0);
  info->uctActions.resize(numactions, 1);
  info->uctVisits = 1;
  info->visited = 0; //false;

  for (int i = 0; i < numactions; i++){
    info->Q[i] = rng.uniform(0,0.01);
  }

  info->needsUpdate = true;

  pthread_mutex_unlock(&info->stateinfo_mutex);

  //if (PLANNERDEBUG) cout << "done with initStateInfo()" << endl;

}


/** Print state info for debugging. */
void ParallelETUCT::printStates(){

  pthread_mutex_lock(&statespace_mutex);
  for (std::set< std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){
    pthread_mutex_unlock(&statespace_mutex);

    state_t s = canonicalize(*i);

    pthread_mutex_lock(&statespace_mutex);
    state_info* info = &(statedata[s]);
    pthread_mutex_unlock(&statespace_mutex);

    cout << "State " << info->id << ": ";
    for (unsigned j = 0; j < s->size(); j++){
      cout << (*s)[j] << ", ";
    }
    cout << endl;

    pthread_mutex_lock(&info->stateinfo_mutex);
    //pthread_mutex_lock(&info->statemodel_mutex);
    for (int act = 0; act < numactions; act++){
      cout << " Q: " << info->Q[act] << endl;
      // << " R: " << info->modelInfo[act].reward << endl;
    }
    // pthread_mutex_unlock(&info->statemodel_mutex);
    pthread_mutex_unlock(&info->stateinfo_mutex);

    pthread_mutex_lock(&statespace_mutex);

  }
  pthread_mutex_unlock(&statespace_mutex);

}


void ParallelETUCT::deleteInfo(state_info* info){

  delete [] info->historyModel;

}



double ParallelETUCT::getSeconds(){
  struct timezone tz;
  timeval timeT;
  gettimeofday(&timeT, &tz);
  return  timeT.tv_sec + (timeT.tv_usec / 1000000.0);
}


/** Execute the uct search from state state at depth depth.
    If terminal or at depth, return some reward.
    Otherwise, select an action based on UCB.
    Simulate action to get reward and next state.
    Call search on next state at depth+1 to get reward return from there on.
    Update q value towards new value: reward + gamma * searchReturn
    Update visit counts for confidence bounds
    Return q

    From "Bandit Based Monte Carlo Planning" by Kocsis and Csaba.
*/
float ParallelETUCT::uctSearch(const std::vector<float> &actS, state_t discS, int depth, std::deque<float> &searchHistory){
  if (UCTDEBUG){
    cout << " uctSearch state ";
    for (unsigned i = 0; i < actS.size(); i++){
      cout << actS[i] << ", ";
    }
    cout << " at depth " << depth << endl;
  }

  pthread_mutex_lock(&statespace_mutex);
  state_info* info = &(statedata[discS]);
  pthread_mutex_unlock(&statespace_mutex);

  // if max depth
  // iterative deepening (probability inversely proportional to visits)
  //float terminateProb = 1.0/(2.0+(float)info->uctVisits);

  // already visited, stop here
  if (depth > MAX_DEPTH){
    pthread_mutex_lock(&info->stateinfo_mutex);

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

    pthread_mutex_unlock(&info->stateinfo_mutex);

    return maxval;
  }

  // select action
  int action = selectUCTAction(info);

  // simulate action to get next state and reward
  // depending on exploration, may also terminate us
  float reward = 0;
  bool term = false;

  pthread_mutex_lock(&info->stateinfo_mutex);

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

  pthread_mutex_unlock(&info->stateinfo_mutex);

  std::vector<float> actualNext = simulateNextState(actS, discS, info, searchHistory, action, &reward, &term);

  // simulate reward from this action
  if (term){
    // this one terminated
    if (UCTDEBUG) cout << "   Terminated on exploration condition" << endl;
    pthread_mutex_lock(&info->stateinfo_mutex);

    info->Q[action] += learnRate * (reward - info->Q[action]);
    info->uctVisits++;
    info->uctActions[action]++;

    if (UCTDEBUG)
      cout << " Depth: " << depth << " Selected action " << action
           << " r: " << reward
           << " StateVisits: " << info->uctVisits
           << " ActionVisits: " << info->uctActions[action] << endl;

    pthread_mutex_unlock(&info->stateinfo_mutex);

    return reward;
  }

  // simulate next state from this action
  state_t discNext = canonicalize(actualNext);

  if (UCTDEBUG)
    cout << " Depth: " << depth << " Selected action " << action
         << " r: " << reward  << endl;

  pthread_mutex_lock(&info->stateinfo_mutex);
  info->visited++; // = true;
  pthread_mutex_unlock(&info->stateinfo_mutex);

  if (HISTORY_SIZE > 0){
    // update history vector for this state
    /*
      for (unsigned i = 0; i < (*discS).size(); i++){
      searchHistory.push_back((*discS)[i]);
      searchHistory.pop_front();
      }

      searchHistory.push_back((*discS)[3]);
      searchHistory.pop_front();
    */
    for (int i = 0; i < numactions; i++){
      if (i == action)
        searchHistory.push_back(1.0);
      else
        searchHistory.push_back(0.0);
      searchHistory.pop_front();
    }

    //    searchHistory.push_back(action);
    //searchHistory.pop_front();
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

  pthread_mutex_lock(&info->stateinfo_mutex);

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
  pthread_mutex_unlock(&info->stateinfo_mutex);

  // return q
  return newQ;

}


int ParallelETUCT::selectUCTAction(state_info* info){
  //  if (UCTDEBUG) cout << "  selectUCTAction" << endl;

  pthread_mutex_lock(&info->stateinfo_mutex);

  std::vector<float> &Q = info->Q;

  if (info->uctActions.size() < (unsigned)numactions){
    cout << "ERROR: uctActions has size " << info->uctActions.size() << endl << flush;
    info->uctActions.resize(numactions);
  }

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

  pthread_mutex_unlock(&info->stateinfo_mutex);

  return act;

}

/** sample from next state distribution */
std::vector<float> ParallelETUCT::simulateNextState(const std::vector<float> &actualState, state_t discState, state_info* info, const std::deque<float> &history, int action, float* reward, bool* term){
  //if (UCTDEBUG) cout << "  simulateNextState" << endl;


  // check if its up to date
  pthread_mutex_lock(&info->statemodel_mutex);
  StateActionInfo* modelInfo = NULL;
  modelInfo = &(info->historyModel[action][history]);

  pthread_mutex_lock(&update_mutex);
  bool upToDate = modelInfo->frameUpdated >= lastUpdate;
  pthread_mutex_unlock(&update_mutex);

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
    pthread_mutex_unlock(&info->statemodel_mutex);
    return actualState;
  }

  float randProb = rng.uniform();

  float probSum = 0.0;
  std::vector<float> nextstate;

  if (REALSTATEDEBUG) cout << "randProb: " << randProb << " numNext: " << modelInfo->transitionProbs.size() << endl;

  //if (modelInfo->transitionProbs.size() == 0)
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

  pthread_mutex_unlock(&info->statemodel_mutex);

  if (trackActual){

    // find the relative change from discrete center
    std::vector<float> relChange = subVec(nextstate, *discState);

    // add that on to actual current state value
    nextstate = addVec(actualState, relChange);

  }

  // check that next state is valid
  for (unsigned j = 0; j < nextstate.size(); j++){
    float factor = EPSILON;
    if (statesPerDim[j] > 0)
      factor = (featmax[j] - featmin[j]) / (float)statesPerDim[j];
    if (nextstate[j] < (featmin[j]-factor)
        || nextstate[j] > (featmax[j]+factor)){
      return actualState;
    }
  }

  // return new actual state
  return nextstate;

}

std::vector<float> ParallelETUCT::selectRandomState(){

  pthread_mutex_lock(&statespace_mutex);
  if (statespace.size() == 0){
    pthread_mutex_unlock(&statespace_mutex);
    return std::vector<float>(featmax.size());
  }
  pthread_mutex_unlock(&statespace_mutex);

  // take a random state from the space of ones we've visited
  int index = 0;
  std::vector<float> state;

  pthread_mutex_lock(&statespace_mutex);
  if (statespace.size() > 1){
    index = rng.uniformDiscrete(0, statespace.size()-1);
  }
  pthread_mutex_unlock(&statespace_mutex);

  int cnt = 0;

  if (PTHREADDEBUG) cout << "*** Planning thread wants search lock (randomstate) ***" << endl << flush;

  pthread_mutex_lock(&statespace_mutex);
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++, cnt++){
    if (cnt == index){
      state = *i;
      break;
    }
  }
  pthread_mutex_unlock(&statespace_mutex);

  return state;
}


void* parallelSearchStart(void* arg){
  ParallelETUCT* pe = reinterpret_cast<ParallelETUCT*>(arg);

  cout << "start parallel uct planning search thread" << endl << flush;

  while(true){
    pe->parallelSearch();
  }

  return NULL;
}

void ParallelETUCT::parallelSearch(){

  std::vector<float> actS;
  state_t discS;
  std::deque<float> searchHistory;

  // get new planning state
  if (PTHREADDEBUG) {
    cout << "*** Planning thread wants planning state lock ***" << endl << flush;
  }
  pthread_mutex_lock(&(plan_state_mutex));
  if (HISTORY_SIZE > 0) pthread_mutex_lock(&history_mutex);

  // too long on one state, lets do random
  if(!doRandom && (getSeconds()-setTime) > 0.5){
    //cout << (getSeconds()-setTime) << " seconds since plan time." << endl;
    doRandom = true;
  }

  // possibly take random state (bet episodes)
  if (doRandom){
    actS = selectRandomState();
    discS = canonicalize(actS);
    searchHistory.resize(saHistory.size(), 0);
    //    cout << "selected random state for search" << endl << flush;
  }
  // or take the state we're in (during episodes)
  else {
    actS = actualPlanState;
    discS = discPlanState;
    searchHistory = saHistory;
  }

  // wait for non-null
  if (discS == NULL){
    pthread_mutex_unlock(&(plan_state_mutex));
    if (HISTORY_SIZE > 0) pthread_mutex_unlock(&history_mutex);
    return;
  }

  if (PTHREADDEBUG){
    pthread_mutex_lock(&statespace_mutex);
    cout << "  uct search from state s ("
         << statedata[discS].uctVisits <<"): ";
    pthread_mutex_unlock(&statespace_mutex);

    for (unsigned i = 0; i < discS->size(); i++){
      cout << (*discS)[i] << ", ";
    }
    cout << endl << flush;
  }

  // call uct search on it
  pthread_mutex_unlock(&(plan_state_mutex));
  if (HISTORY_SIZE > 0) pthread_mutex_unlock(&history_mutex);

  if (PTHREADDEBUG) cout << "*** Planning thread wants search lock ***" << endl;
  uctSearch(actS, discS, 0, searchHistory);

  pthread_yield();

}


// canonicalize all the states so we already have them in our statespace
void ParallelETUCT::initStates(){
  cout << "init states" << endl;
  std::vector<float> s(featmin.size());

  fillInState(s,0);
}

void ParallelETUCT::fillInState(std::vector<float>s, int depth){

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



void ParallelETUCT::savePolicy(const char* filename){

  ofstream policyFile(filename, ios::out | ios::binary | ios::trunc);

  // first part, save the vector size
  int fsize = featmin.size();
  policyFile.write((char*)&fsize, sizeof(int));

  // save numactions
  policyFile.write((char*)&numactions, sizeof(int));

  // go through all states, and save Q values
  pthread_mutex_lock(&statespace_mutex);

  for (std::set< std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){
    pthread_mutex_unlock(&statespace_mutex);

    state_t s = canonicalize(*i);

    pthread_mutex_lock(&statespace_mutex);
    state_info* info = &(statedata[s]);
    pthread_mutex_unlock(&statespace_mutex);

    // save state
    policyFile.write((char*)&((*i)[0]), sizeof(float)*fsize);

    // save q-values
    pthread_mutex_lock(&info->stateinfo_mutex);
    policyFile.write((char*)&(info->Q[0]), sizeof(float)*numactions);
    pthread_mutex_unlock(&info->stateinfo_mutex);

    pthread_mutex_lock(&statespace_mutex);
  }
  pthread_mutex_unlock(&statespace_mutex);

  policyFile.close();
}



void ParallelETUCT::loadPolicy(const char* filename){

  ifstream policyFile(filename, ios::in | ios::binary);

  // first part, save the vector size
  int fsize;
  policyFile.read((char*)&fsize, sizeof(int));
  cout << "Numfeats loaded: " << fsize << endl << flush;

  // save numactions
  int nact;
  policyFile.read((char*)&nact, sizeof(int));
  cout << "nact loaded: " << nact << endl << flush;
  cout << " numactions: " << numactions << endl << flush;

  if (nact != numactions){
    cout << "this policy is not valid loaded nact: " << nact
         << " was told: " << numactions << endl << flush;
    exit(-1);
  }

  // go through all states, loading q values
  while(!policyFile.eof()){
    std::vector<float> state(fsize, 0.0);

    // load state
    policyFile.read((char*)&(state[0]), sizeof(float)*fsize);
    //if (LOADDEBUG){
    //cout << "load policy for state: ";
    // printState(state);
    //}

    state_t s = canonicalize(state);

    pthread_mutex_lock(&statespace_mutex);
    state_info* info = &(statedata[s]);
    pthread_mutex_unlock(&statespace_mutex);

    if (policyFile.eof()) break;

    // load q values
    pthread_mutex_lock(&info->stateinfo_mutex);

    policyFile.read((char*)&(info->Q[0]), sizeof(float)*numactions);

    info->uctVisits = numactions * 100;

    for (int j = 0; j < numactions; j++){
      info->uctActions[j] = 100;
    }

    info->needsUpdate = true;

    pthread_mutex_unlock(&info->stateinfo_mutex);

    //if (LOADDEBUG){
    //cout << "Q values: " << endl;
    //for (int iAct = 0; iAct < numactions; iAct++){
    //  cout << " Action: " << iAct << " val: " << info->Q[iAct] << endl;
    //}
    //}
  }

  policyFile.close();
  cout << "Policy loaded!!!" << endl << flush;
}

void ParallelETUCT::logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax){
  std::vector<float> state(2, 0.0);
  for (int i = xmin ; i < xmax; i++){
    for (int j = ymin; j < ymax; j++){
      state[0] = j;
      state[1] = i;
      state_t s = canonicalize(state);

      pthread_mutex_lock(&statespace_mutex);
      state_info* info = &(statedata[s]);
      pthread_mutex_unlock(&statespace_mutex);

      pthread_mutex_lock(&info->stateinfo_mutex);

      std::vector<float> &Q_s = info->Q;
      const std::vector<float>::iterator max =
        random_max_element(Q_s.begin(), Q_s.end());
      *of << (*max) << ",";

      pthread_mutex_unlock(&info->stateinfo_mutex);

    }
  }
}


// should do it such that an already discretized state stays the same
// mainly the numerical value of each bin should be the average of that bin
std::vector<float> ParallelETUCT::discretizeState(const std::vector<float> &s){
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


std::vector<float> ParallelETUCT::addVec(const std::vector<float> &a, const std::vector<float> &b){
  if (a.size() != b.size())
    cout << "ERROR: add vector sizes wrong " << a.size() << ", " << b.size() << endl;

  std::vector<float> c(a.size(), 0.0);
  for (unsigned i = 0; i < a.size(); i++){
    c[i] = a[i] + b[i];
  }

  return c;
}

std::vector<float> ParallelETUCT::subVec(const std::vector<float> &a, const std::vector<float> &b){
  if (a.size() != b.size())
    cout << "ERROR: sub vector sizes wrong " << a.size() << ", " << b.size() << endl;

  std::vector<float> c(a.size(), 0.0);
  for (unsigned i = 0; i < a.size(); i++){
    c[i] = a[i] - b[i];
  }

  return c;
}

void ParallelETUCT::setFirst(){
  if (HISTORY_SIZE == 0) return;

  if (HISTORYDEBUG) cout << "first action, set sahistory to 0s" << endl;

  pthread_mutex_lock(&(history_mutex));
  // first action, reset history vector
  saHistory.resize(saHistory.size(), 0.0);
  pthread_mutex_unlock(&(history_mutex));
}

void ParallelETUCT::setSeeding(bool seeding){

  if (HISTORYDEBUG) cout << "set seed mode to " << seeding << endl;
  seedMode = seeding;

}
