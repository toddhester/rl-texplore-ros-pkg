/** \file RMaxModel.cc
    Implements the RMaxModel class.
    \author Todd Hester
*/

#include "RMaxModel.hh"




RMaxModel::RMaxModel(int m, int nact, Random rng):
  M(m), nact(nact), rng(rng)
{

  nstates = 0;
  RMAX_DEBUG = false; //true;
  //initMDPModel(nfactors);  
}

RMaxModel::RMaxModel(const RMaxModel &rm):
M(rm.M), nact(rm.nact), rng(rm.rng)
{

  nstates = rm.nstates;
  RMAX_DEBUG = rm.RMAX_DEBUG;

  statespace = rm.statespace;

  // copy info into map for each of these
  // because map is by address, must get correct address for each
  for (std::map<state_t, state_info>::const_iterator i = rm.statedata.begin();
       i != rm.statedata.end(); i++){

    state_t s = canonicalize(*((*i).first));
    statedata[s] = (*i).second;

  }

}

RMaxModel* RMaxModel::getCopy(){
  RMaxModel* copy = new RMaxModel(*this);
  return copy;
}

RMaxModel::~RMaxModel() {}


bool RMaxModel::updateWithExperiences(std::vector<experience> &instances){

  bool changed = false;
  
  for (unsigned i = 0; i < instances.size(); i++){
    bool singleChange = updateWithExperience(instances[i]);
    changed = changed || singleChange;
  }
  return changed;
}


// update all the counts, check if model has changed
// stop counting at M
bool RMaxModel::updateWithExperience(experience &e){
  if (RMAX_DEBUG) cout << "updateWithExperience " << &(e.s) << ", " << e.act 
                       << ", " << &(e.next) << ", " << e.reward << endl;

  // get state info for last state
  state_t l = canonicalize(e.s);
  state_info* info = &(statedata[l]);

  bool modelChanged = false;

  // stop at M
  //if (info->visits[e.act] == M)
  if (info->known[e.act])
    return false;

  // update visit count for action just executed
  info->visits[e.act]++;

  // update termination count
  if (e.terminal) info->terminations[e.act]++;

  // update reward sum for this action
  info->Rsum[e.act] += e.reward;

  // update transition count for outcome that occured
  std::vector<int> &transCounts = info->outCounts[e.next];

  // first, check that we have resized this counter
  checkTransitionCountSize(&transCounts);

  // only update state transition counts for non-terminal transitions
  if (!e.terminal){
    // then update transition count for this action/outcome
    transCounts[e.act]++;
  }    

  // check if count becomes known 
  if (!info->known[e.act] && info->visits[e.act] >= M){
    info->known[e.act] = true;
    modelChanged = true;
  }

  if (RMAX_DEBUG) cout << "s" << info->id << " act: " << e.act 
                               << " transCounts[act] = " << transCounts[e.act] 
                               << " visits[act] = " << info->visits[e.act] << endl;
  
  // anything that got past the 'return false' above is a change in conf or predictions
  return true; //modelChanged;

}


// calculate state info such as transition probs, known/unknown, reward prediction
float RMaxModel::getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval){
  if (RMAX_DEBUG) cout << "getStateActionInfo, " << &state <<  ", " << act << endl;


  retval->transitionProbs.clear();

  // get state-action info for this state
  state_t l = canonicalize(state);
  state_info* info = &(statedata[l]);

  
  // see if it has any visits (could still be unknown)
  if (info->visits[act] == 0){
    if (RMAX_DEBUG) cout << "This outcome is unknown" << endl;
    retval->reward = -0.001;

    // add to transition map
    retval->transitionProbs[state] = 1.0;
    retval->known = false;
    retval->termProb = 0.0;
    return 0;
  }
  
  
  // fill in transition probs
  for (std::map<std::vector<float>, std::vector<int> >::iterator it = info->outCounts.begin(); 
       it != info->outCounts.end(); it++){

    // get key from iterator
    std::vector<float> next = (*it).first;
    int count = ((*it).second)[act];

    // add to transition map
    if (count > 0.0){
      retval->transitionProbs[next] = (float)count / (float)(info->visits[act] - info->terminations[act]);
      if (RMAX_DEBUG) cout << "Outcome " << &next << " has prob " << retval->transitionProbs[next]
			   << " from count of " << count << " on " 
			   << info->visits[act] << " visits." << endl;
    }
  }
  

  // add in avg rewrad
  retval->reward = (float)info->Rsum[act] / (float)info->visits[act];
  if (RMAX_DEBUG) cout << "Avg Reward of  " << retval->reward << " from reward sum of " 
		       << info->Rsum[act] 
		       << " on " << info->visits[act] << " visits." << endl;

  // termination probability
  retval->termProb = (float)info->terminations[act] / (float)info->visits[act];
  if (RMAX_DEBUG) cout << "termProb: " << retval->termProb << endl;
  if (retval->termProb < 0 || retval->termProb > 1){
    cout << "Problem with termination probability: " << retval->termProb << endl;
  }


  retval->known = info->known[act];
  // conf as a pct of float m (so 0.5 is exactly M)
  float conf = (float)info->visits[act]/ (2.0 * (float)M);

  return conf;

}





RMaxModel::state_t RMaxModel::canonicalize(const std::vector<float> &s) {
  if (RMAX_DEBUG) cout << "canonicalize, s = " << &s << endl;

  // get state_t for pointer if its in statespace
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s);
  state_t retval = &*result.first; // Dereference iterator then get pointer 

  if (RMAX_DEBUG) cout << " returns " << retval << endl;

  // if not, init this new state
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    initNewState(retval);
  }

  return retval; 
}

void RMaxModel::initNewState(state_t s){
  if (RMAX_DEBUG) cout << "initNewState(s = " << s 
		       << ")" << endl;
  
  // create state info and add to hash map
  state_info* info = &(statedata[s]);
  initStateInfo(info);

}


// init state info
void RMaxModel::initStateInfo(state_info* info){
  if (RMAX_DEBUG) cout << "initStateInfo()";

  info->id = nstates++;
  if (RMAX_DEBUG) cout << " id = " << info->id << endl;

  // model data (q values, state-action counts, transition, etc)
  info->visits.resize(nact, 0);
  info->Rsum.resize(nact, 0);
  info->known.resize(nact, false);
  info->terminations.resize(nact, 0);

}


void RMaxModel::checkTransitionCountSize(std::vector<int>* transCounts){
  if (RMAX_DEBUG) cout << "checkTransitionCountSize(transCounts) " 
		    << "size: " << transCounts->size() << endl;

  // resize to numactions if not initialized for this outcome yet
  if (transCounts->size() == 0)
    transCounts->resize(nact, 0);

}
