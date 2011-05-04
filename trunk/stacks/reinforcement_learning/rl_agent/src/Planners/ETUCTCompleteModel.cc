#include "ETUCTCompleteModel.hh"
#include <algorithm>

//#include <time.h>
#include <sys/time.h>


ETUCTCompleteModel::ETUCTCompleteModel(int numactions, float gamma, 
				       float rmax, float lambda,
				       int MAX_ITER, float MAX_TIME, 
				       int MAX_DEPTH,
				       char env, Random rng):
  numactions(numactions), gamma(gamma), rmax(rmax), lambda(lambda),
  MAX_ITER(MAX_ITER), MAX_TIME(MAX_TIME), MAX_DEPTH(MAX_DEPTH), env(env),
  rng(rng)
{
  nstates = 0;
  nactions = 0;

  PLANNERDEBUG = false; //true; ///false;
  POLICYDEBUG = false; //true; //false; //true; //false;
  ACTDEBUG = false; //true; //false; //true;
  MODELDEBUG = false; //true;
  UCTDEBUG = false; //true; //false;

  // init env
  setEnvironment();

}

ETUCTCompleteModel::~ETUCTCompleteModel() {}

void ETUCTCompleteModel::setModel(MDPModel* m){

}


/////////////////////////////
// Functional functions :) //
/////////////////////////////


void ETUCTCompleteModel::initNewState(state_t s){
  if (PLANNERDEBUG) cout << "initNewState(s = " << s 
		     << ") size = " << s->size() << endl;

  if (MODELDEBUG) cout << "New State: " << endl;

  // create state info and add to hash map
  state_info* info = &(statedata[s]);
  initStateInfo(info);

  if (PLANNERDEBUG) cout << "done with initNewState()" << endl;

}

/** Use the latest experience to update state info and the model. */
bool ETUCTCompleteModel::updateModelWithExperience(const std::vector<float> &laststate, 
					       int lastact, 
					       const std::vector<float> &currstate, 
					       float reward){
  // do nothing
  return false;

}




/** Choose the next action */
int ETUCTCompleteModel::getBestAction(const std::vector<float> &state){
  if (UCTDEBUG || PLANNERDEBUG) cout << "getBestAction(s = " << &state 
		      << ")" << endl;
  
  state_t s = canonicalize(state);

  double initTime = getSeconds();
  double currTime = getSeconds();
  int i = 0;
  //for (i = 0; i < MAX_ITER; i++){
  while(true){

    if (UCTDEBUG) cout << endl << "UCT Search Iter: " << i << endl;
    uctSearch(*s, 0);
    i++;
    
    // break after some max time 
    if ((getSeconds() - initTime) > MAX_TIME){
      break;
    }

  }
  currTime = getSeconds();
  if (false || UCTDEBUG){
    cout << "Search complete after " << (currTime-initTime) << " seconds and "
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
  float val = *a;

  if (ACTDEBUG){
    cout << endl << "chooseAction State " << (*s)[0] << "," << (*s)[1] 
	 << " act: " << act << " val: " << val << endl;
    for (int iAct = 0; iAct < numactions; iAct++){
      cout << " Action: " << iAct 
	   << " val: " << Q[iAct] 
	   << " visits: " << info->visits[iAct] << endl;
    }
  }

  nactions++;

  // return index of action
  return act;
}






void ETUCTCompleteModel::planOnNewModel(){

  // do nothing
}



////////////////////////////
// Helper Functions       //
////////////////////////////

ETUCTCompleteModel::state_t ETUCTCompleteModel::canonicalize(const std::vector<float> &inputS) {
  if (PLANNERDEBUG) cout << "canonicalize(s = " << inputS[0] << ", " 
			     << inputS[1] << ")" << endl; 

  //std::vector<float> s = modifyState(inputS);
  std::vector<float> s = inputS;

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


std::vector<float>::iterator
ETUCTCompleteModel::random_max_element(std::vector<float>::iterator start,
			      std::vector<float>::iterator end) {
  if (PLANNERDEBUG) cout << "random_max_element(start, end)" << endl;

  std::vector<float>::iterator max =
    std::max_element(start, end);
  int n = std::count(max, end, *max);
  if (n > 1) {
    n = rng.uniformDiscrete(1, n);
    while (n > 1) {
      max = std::find(max + 1, end, *max);
      --n;
    }
  }
  return max;
}




// init state info
void ETUCTCompleteModel::initStateInfo(state_info* info){
  if (PLANNERDEBUG) cout << "initStateInfo()";

  info->id = nstates++;
  if (PLANNERDEBUG) cout << " id = " << info->id << endl;

  info->fresh = true;
  info->stepsAway = 100000;

  // model q values, visit counts
  info->visits.resize(numactions, 0);
  info->Q.resize(numactions, 0); //-1.3);
  info->uctActions.resize(numactions, 0);
  info->uctVisits = 0;
  info->visited = false;
  
  if (PLANNERDEBUG) cout << "done with initStateInfo()" << endl;

}


/** Print state info for debugging. */
void ETUCTCompleteModel::printStates(){
  
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
	   << " Q: " << info->Q[act] << endl;
    }

  }
}


void ETUCTCompleteModel::deleteInfo(state_info* info){

}


double ETUCTCompleteModel::getSeconds(){
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
float ETUCTCompleteModel::uctSearch(std::vector<float> state, int depth){
  if (UCTDEBUG){
    cout << " uctSearch state ";
    for (unsigned i = 0; i < state.size(); i++){
      cout << state[i] << ", ";
    }
    cout << " at depth " << depth << endl;
  }
  

  state_t s = canonicalize(state);
  state_info* info = &(statedata[s]); 

  // if max depth 
  // iterative deepening (probability inversely proportional to visits)
  //float terminateProb = 1.0/(2.0+(float)info->uctVisits);
  
  //if (UCTDEBUG) cout << "Terminate prob: " << terminateProb << endl;
  //if (depth > 0 && (depth > MAX_DEPTH || rng.bernoulli(terminateProb))){
  // already visited, stop here
  //if (depth > MAX_DEPTH || info->visited){
  if (info->visited){
    //      || (depth > 0 && rng.bernoulli(terminateProb))){
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
  float reward;

  //  float learnRate = 10.0 / (info->uctActions[action]+10.0); 
  //float learnRate=  0.001; //1000.0 / (info->uctActions[action]+1000.0);//.001; 
  float learnRate=  10.0 / (info->uctActions[action]+100.0);
  //learnRate = 1.0;


  // simulate reward from this action
  if (simulateReward(s, action, &reward)){
    // this one terminated
    if (UCTDEBUG) cout << "   Terminated " << endl;
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

  // simulate next state from this action
  std::vector<float> next = simulateNextState(s, action);

  if (UCTDEBUG)
    cout << " Depth: " << depth << " Selected action " << action 
	 << " r: " << reward  << endl;

  info->visited = true;

  // new q value
  float newQ = reward + gamma * uctSearch(next, depth+1);

  // update q and visit counts
  info->Q[action] += learnRate * (newQ - info->Q[action]);
  info->uctVisits++;
  info->uctActions[action]++;
  info->visited = false;

  if (UCTDEBUG)
    cout << " Depth: " << depth << " Qtarget: " << newQ 
	 << " updated Q: " << info->Q[action]
	 << " learnRate: " << learnRate
	 << " StateVisits: " << info->uctVisits 
	 << " ActionVisits: " << info->uctActions[action] << endl;

  // new idea, return max of Q or new q
  std::vector<float>::iterator maxAct =
    std::max_element(info->Q.begin(), 
		     info->Q.end());
  float maxval = *maxAct;
  
  // replace with w avg of maxq and new val
  float newVal = (lambda * newQ) + ((1.0-lambda) * maxval);
  
  if (UCTDEBUG)
    cout << " Replacing newQ: " << newQ
	 << " with wAvg: " << newVal << endl;
  newQ = newVal;
  
  // return q
  return newQ;

}


int ETUCTCompleteModel::selectUCTAction(state_info* info){
  if (UCTDEBUG) cout << "  selectUCTAction" << endl;

  // uniform sampling
  //return rng.uniformDiscrete(0, numactions-1);

  // if there are some unvisited actions, select randomly from them
  int numUnvisited = 0;
  std::vector<int> unvisited;
  for (int i = 0; i < numactions; i++){
    if (info->uctActions[i] == 0){
      if (UCTDEBUG) cout << "  Action " << i << " is unvisited." << endl;
      numUnvisited++;
      unvisited.push_back(i);
    }
  }

  if (numUnvisited > 1){
    int index = rng.uniformDiscrete(0, numUnvisited-1);
    if (UCTDEBUG) cout << "  Randomly selected " << unvisited[index] << endl;
    return unvisited[index];
  } else if (numUnvisited == 1) {
    return unvisited[0];
  }

  std::vector<float> &Q = info->Q;

  // loop through
  float rewardBound = rmax*2;
  if (rewardBound < 1.0)
    rewardBound = 1.0;
  rewardBound /= (1.0 - gamma);
  if (UCTDEBUG) cout << "Reward bound: " << rewardBound << endl;

  std::vector<float> uctQ(numactions, 0.0);

  for (int i = 0; i < numactions; i++){

    // this actions value is Q + rMax * sqrt (2 log N(s) / N(s,a))
    uctQ[i] = Q[i] + 
      rewardBound * 2.0 * sqrt(log((float)info->uctVisits) / 
			       (float)info->uctActions[i]);
    
    if (UCTDEBUG) 
      cout << "  Action: " << i << " Q: " << Q[i]
	   << " visits: " << info->uctActions[i] 
	   << " value: " << uctQ[i] << endl;
  }

  // random max
  std::vector<float>::iterator maxAct =
    random_max_element(uctQ.begin(), 
		       uctQ.end());
  float maxval = *maxAct;
  int act = maxAct - uctQ.begin();

  if (UCTDEBUG)
    cout << "  Selected " << act << " val: " << maxval << endl;

  return act;

}

/** get reward from model (including exploration bonuses, etc) */
bool ETUCTCompleteModel::simulateReward(state_t s, int act, float* reward){

  // set sensation
  if (env == 'n'){
    ((NFL*)domain)->setSensation((*s)[0], (*s)[1], (*s)[2], (*s)[3], (*s)[4], (*s)[5], (*s)[6], (*s)[7], (*s)[8], (*s)[9], (*s)[10], (*s)[11]);
    //  } else if (env == 'm'){
    //((Minesweeper*)domain)->setSensation(*s);
  } else if (env == 'w'){
    ((Stocks*)domain)->setSensation(*s);
  } 
  *reward = domain->apply(act);
  return domain->terminal();
  
}


/** sample from next state distribution */
std::vector<float> ETUCTCompleteModel::simulateNextState(state_t s, int action){
  if (UCTDEBUG) cout << "  simulateNextState" << endl;

  return domain->sensation();

}


void ETUCTCompleteModel::setEnvironment(){
  if (env == 'n'){
    domain = new NFL(rng, true, true);
    //domain->NFL_DEBUG = false;
    //domain->FOURTH_DEBUG = false;
    //  } else if (env == 'm'){
    //domain = new Minesweeper(rng, 5);
    //domain->MINEDEBUG = false;
  } else if (env == 'w'){
    domain = new Stocks(rng, true, 3, 3); 
  }
}


std::vector<float> ETUCTCompleteModel::modifyState(const std::vector<float> &input){

  // modify the state to make fewer states to do updates on
  std::vector<float> output(input.size());
  //cout << "Original state: ";
  for (unsigned i = 0; i < input.size(); i++){
    output[i] = input[i];
    //cout << output[i] << ", ";
  }
  //  cout << endl;

  // consider all opptimeouts values to be the same
  output[10] = 1.0;

  // all of my timeouts > 0 are the same
  if (output[7] > 0)
    output[7] = 1;

  // all play clocks greater than 10 are the same
  if (output[5] > 10)
    output[5] = 20;

  // greater than 1 minute, 20 sec segments
  if (output[4] > 60){
    int seg = (output[4]+10) / 20;
    output[4] = seg*20.0;
  }

  // all yardlines between the 10s, split into 10 yard increments
  if (output[0] > 10 && output[0] < 90){
    int seg = (output[0]+5) / 10;
    output[0] = seg*10.0;
  }

  /*
  cout << "Modified state: ";
  for (unsigned i = 0; i < input.size(); i++){
    cout << output[i] << ", ";
  }
  cout << endl << endl;
  */

  return output;

}
