#include <rl_agent/QLearner.hh>
#include <algorithm>

QLearner::QLearner(int numactions, float gamma,
                   float initialvalue, float alpha, float ep,
                   Random rng):
  numactions(numactions), gamma(gamma),
  initialvalue(initialvalue), alpha(alpha),
  rng(rng), currentq(NULL)
{

  epsilon = ep;
  ACTDEBUG = false; //true; //false;

}

QLearner::~QLearner() {}

int QLearner::first_action(const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "First - in state: ";
    printState(s);
    cout << endl;
  }

  // Get action values
  std::vector<float> &Q_s = Q[canonicalize(s)];

  // Choose an action
  const std::vector<float>::iterator a =
    rng.uniform() < epsilon
    ? Q_s.begin() + rng.uniformDiscrete(0, numactions - 1) // Choose randomly
    : random_max_element(Q_s.begin(), Q_s.end()); // Choose maximum

  // Store location to update value later
  currentq = &*a;

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

int QLearner::next_action(float r, const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "Next: got reward " << r << " in state: ";
    printState(s);
    cout << endl;
  }

  // Get action values
  std::vector<float> &Q_s = Q[canonicalize(s)];
  const std::vector<float>::iterator max =
    random_max_element(Q_s.begin(), Q_s.end());

  // Update value of action just executed
  *currentq += alpha * (r + gamma * (*max) - *currentq);

  // Choose an action
  const std::vector<float>::iterator a =
    rng.uniform() < epsilon
    ? Q_s.begin() + rng.uniformDiscrete(0, numactions - 1)
    : max;

  // Store location to update value later
  currentq = &*a;

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

void QLearner::last_action(float r) {

  if (ACTDEBUG){
    cout << "Last: got reward " << r << endl;
  }

  *currentq += alpha * (r - *currentq);
  currentq = NULL;
}

QLearner::state_t QLearner::canonicalize(const std::vector<float> &s) {
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s);
  state_t retval = &*result.first; // Dereference iterator then get pointer
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    std::vector<float> &Q_s = Q[retval];
    Q_s.resize(numactions,initialvalue);
  }
  return retval;
}



std::vector<float>::iterator
QLearner::random_max_element(
                             std::vector<float>::iterator start,
                             std::vector<float>::iterator end) {

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




void QLearner::setDebug(bool d){
  ACTDEBUG = d;
}


void QLearner::printState(const std::vector<float> &s){
  for (unsigned j = 0; j < s.size(); j++){
    cout << s[j] << ", ";
  }
}



void QLearner::seedExp(std::vector<experience> seeds){

  // for each seeding experience, update our model
  for (unsigned i = 0; i < seeds.size(); i++){
    experience e = seeds[i];

    std::vector<float> &Q_s = Q[canonicalize(e.s)];
    std::vector<float> &Q_next = Q[canonicalize(e.next)];

    // get max value of next state
    const std::vector<float>::iterator max =
      random_max_element(Q_next.begin(), Q_next.end());

    // Get q value for action taken
    const std::vector<float>::iterator a = Q_s.begin() + e.act;
    currentq = &*a;

    // Update value of action just executed
    *currentq += alpha * (e.reward + gamma * (*max) - *currentq);


    /*
      cout << "Seeding with experience " << i << endl;
      cout << "last: " << (e.s)[0] << ", " << (e.s)[1] << ", "
      << (e.s)[2] << endl;
      cout << "act: " << e.act << " r: " << e.reward << endl;
      cout << "next: " << (e.next)[0] << ", " << (e.next)[1] << ", "
      << (e.next)[2] << ", " << e.terminal << endl;
      cout << "Q: " << *currentq << " max: " << *max << endl;
    */

  }


}

void QLearner::logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax){
  std::vector<float> s;
  s.resize(2, 0.0);
  for (int i = xmin ; i < xmax; i++){
    for (int j = ymin; j < ymax; j++){
      s[0] = j;
      s[1] = i;
      std::vector<float> &Q_s = Q[canonicalize(s)];
      const std::vector<float>::iterator max =
        random_max_element(Q_s.begin(), Q_s.end());
      *of << (*max) << ",";
    }
  }
}


float QLearner::getValue(std::vector<float> state){

  state_t s = canonicalize(state);

  // Get Q values
  std::vector<float> &Q_s = Q[s];

  // Choose an action
  const std::vector<float>::iterator a =
    random_max_element(Q_s.begin(), Q_s.end()); // Choose maximum

  // Get avg value
  float valSum = 0.0;
  float cnt = 0;
  for (std::set<std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);

    // get state's info
    std::vector<float> &Q_s = Q[s];

    for (int j = 0; j < numactions; j++){
      valSum += Q_s[j];
      cnt++;
    }
  }

  cout << "Avg Value: " << (valSum / cnt) << endl;

  return *a;
}


void QLearner::savePolicy(const char* filename){

  ofstream policyFile(filename, ios::out | ios::binary | ios::trunc);

  // first part, save the vector size
  std::set< std::vector<float> >::iterator i = statespace.begin();
  int fsize = (*i).size();
  policyFile.write((char*)&fsize, sizeof(int));

  // save numactions
  policyFile.write((char*)&numactions, sizeof(int));

  // go through all states, and save Q values
  for (std::set< std::vector<float> >::iterator i = statespace.begin();
       i != statespace.end(); i++){

    state_t s = canonicalize(*i);
    std::vector<float> *Q_s = &(Q[s]);

    // save state
    policyFile.write((char*)&((*i)[0]), sizeof(float)*fsize);

    // save q-values
    policyFile.write((char*)&((*Q_s)[0]), sizeof(float)*numactions);

  }

  policyFile.close();
}


void QLearner::loadPolicy(const char* filename){
  bool LOADDEBUG = false;

  ifstream policyFile(filename, ios::in | ios::binary);
  if (!policyFile.is_open())
    return;

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
    std::vector<float> *Q_s = &(Q[s]);

    if (policyFile.eof()) break;

    // load q values
    policyFile.read((char*)&((*Q_s)[0]), sizeof(float)*numactions);

    if (LOADDEBUG){
      cout << "Q values: " << endl;
      for (int iAct = 0; iAct < numactions; iAct++){
        cout << " Action: " << iAct << " val: " << (*Q_s)[iAct] << endl;
      }
    }
  }

  policyFile.close();
  cout << "Policy loaded!!!" << endl;
  //loaded = true;
}


