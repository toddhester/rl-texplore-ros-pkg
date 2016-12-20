#include <rl_agent/SavedPolicy.hh>
#include <algorithm>

SavedPolicy::SavedPolicy(int numactions, const char* filename):
  numactions(numactions)
{

  ACTDEBUG = false;
  LOADDEBUG = false;
  loaded = false;

  loadPolicy(filename);

  
}

SavedPolicy::~SavedPolicy() {}

int SavedPolicy::first_action(const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "First - in state: ";
    printState(s);
    cout << endl;
  }

  // Get action values
  std::vector<float> &Q_s = Q[canonicalize(s)];

  // Choose an action
  const std::vector<float>::iterator a =
    std::max_element(Q_s.begin(), Q_s.end()); // Choose maximum

  if (ACTDEBUG){
    cout << " act: " << (a-Q_s.begin()) << " val: " << *a << endl;
    for (int iAct = 0; iAct < numactions; iAct++){
      cout << " Action: " << iAct 
	   << " val: " << Q_s[iAct] << endl;
    }
    cout << "Took action " << (a-Q_s.begin()) << " from state ";
    printState(s);
    cout << endl;
  }

  return a - Q_s.begin();
}

int SavedPolicy::next_action(float r, const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "Next: got reward " << r << " in state: ";
    printState(s);
    cout << endl;
  }

  // Get action values
  std::vector<float> &Q_s = Q[canonicalize(s)];
  const std::vector<float>::iterator max =
    std::max_element(Q_s.begin(), Q_s.end());

  // Choose an action
  const std::vector<float>::iterator a = max;

  if (ACTDEBUG){
    cout << " act: " << (a-Q_s.begin()) << " val: " << *a << endl;
    for (int iAct = 0; iAct < numactions; iAct++){
      cout << " Action: " << iAct 
	   << " val: " << Q_s[iAct] << endl;
    }
    cout << "Took action " << (a-Q_s.begin()) << " from state ";
    printState(s);
    cout << endl;
  }

  return a - Q_s.begin();
}

void SavedPolicy::last_action(float r) {

  if (ACTDEBUG){
    cout << "Last: got reward " << r << endl;
  }

}

SavedPolicy::state_t SavedPolicy::canonicalize(const std::vector<float> &s) {
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s);
  state_t retval = &*result.first; // Dereference iterator then get pointer 
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    if (loaded){
      cout << "State unknown in policy!!!" << endl;
      for (unsigned i = 0; i < s.size(); i++){
        cout << s[i] << ", ";
      }
      cout << endl;
    }
    std::vector<float> &Q_s = Q[retval];
    Q_s.resize(numactions,0.0);
  }
  return retval; 
}



void SavedPolicy::printState(const std::vector<float> &s){
  for (unsigned j = 0; j < s.size(); j++){
    cout << s[j] << ", ";
  }
}



void SavedPolicy::seedExp(std::vector<experience> seeds){
  return;
}


void SavedPolicy::loadPolicy(const char* filename){

  ifstream policyFile(filename, ios::in | ios::binary);

  // first part, save the vector size
  int fsize;
  policyFile.read((char*)&fsize, sizeof(int));
  if (LOADDEBUG) cout << "Numfeats loaded: " << fsize << endl;

  // save numactions
  int nact;
  policyFile.read((char*)&nact, sizeof(int));

  if (nact != numactions){
    cout << "this policy is not valid loaded nact: " << nact 
         << " was told: " << numactions << endl;
    exit(-1);
  }

  // go through all states, loading q values
  while(!policyFile.eof()){
    std::vector<float> state;
    state.resize(fsize, 0.0);

    // load state
    policyFile.read((char*)&(state[0]), sizeof(float)*fsize);
    if (LOADDEBUG){
      cout << "load policy for state: ";
      printState(state);
    }

    state_t s = canonicalize(state);

    if (policyFile.eof()) break;

    // load q values
    policyFile.read((char*)&(Q[s][0]), sizeof(float)*numactions);
    
    if (LOADDEBUG){
      cout << "Q values: " << endl;
      for (int iAct = 0; iAct < numactions; iAct++){
        cout << " Action: " << iAct << " val: " << Q[s][iAct] << endl;
      }
    }
  }
  
  policyFile.close();
  cout << "Policy loaded!!!" << endl;
  loaded = true;
}
