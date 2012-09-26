#include <rl_agent/Sarsa.hh>
#include <algorithm>

Sarsa::Sarsa(int numactions, float gamma,
             float initialvalue, float alpha, float ep, float lambda,
             Random rng):
  numactions(numactions), gamma(gamma),
  initialvalue(initialvalue), alpha(alpha),
  epsilon(ep), lambda(lambda),
  rng(rng)
{

  currentq = NULL;
  ACTDEBUG = false; //true; //false;
  ELIGDEBUG = false;

}

Sarsa::~Sarsa() {}

int Sarsa::first_action(const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "First - in state: ";
    printState(s);
    cout << endl;
  }

  // clear all eligibility traces
  for (std::map<state_t, std::vector<float> >::iterator i = eligibility.begin();
       i != eligibility.end(); i++){

    std::vector<float> & elig_s = (*i).second;
    for (int j = 0; j < numactions; j++){
      elig_s[j] = 0.0;
    }
  }

  // Get action values
  state_t si = canonicalize(s);
  std::vector<float> &Q_s = Q[si];

  // Choose an action
  const std::vector<float>::iterator a =
    rng.uniform() < epsilon
    ? Q_s.begin() + rng.uniformDiscrete(0, numactions - 1) // Choose randomly
    : random_max_element(Q_s.begin(), Q_s.end()); // Choose maximum

  // set eligiblity to 1
  std::vector<float> &elig_s = eligibility[si];
  elig_s[a-Q_s.begin()] = 1.0;

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

int Sarsa::next_action(float r, const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "Next: got reward " << r << " in state: ";
    printState(s);
    cout << endl;
  }

  // Get action values
  state_t st = canonicalize(s);
  std::vector<float> &Q_s = Q[st];
  const std::vector<float>::iterator max =
    random_max_element(Q_s.begin(), Q_s.end());

  // Choose an action
  const std::vector<float>::iterator a =
    rng.uniform() < epsilon
    ? Q_s.begin() + rng.uniformDiscrete(0, numactions - 1)
    : max;

  // Update value for all with positive eligibility
  for (std::map<state_t, std::vector<float> >::iterator i = eligibility.begin();
       i != eligibility.end(); i++){

    state_t si = (*i).first;
    std::vector<float> & elig_s = (*i).second;
    for (int j = 0; j < numactions; j++){
      if (elig_s[j] > 0.0){
        if (ELIGDEBUG) {
          cout << "updating state " << (*((*i).first))[0] << ", " << (*((*i).first))[1] << " act: " << j << " with elig: " << elig_s[j] << endl;
        }
        // update
        Q[si][j] += alpha * elig_s[j] * (r + gamma * (*a) - Q[si][j]);
        elig_s[j] *= lambda;
      }
    }
        
  }

  // Set elig to 1
  eligibility[st][a-Q_s.begin()] = 1.0;

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

void Sarsa::last_action(float r) {

  if (ACTDEBUG){
    cout << "Last: got reward " << r << endl;
  }

  // Update value for all with positive eligibility
  for (std::map<state_t, std::vector<float> >::iterator i = eligibility.begin();
       i != eligibility.end(); i++){
    
    state_t si = (*i).first;
    std::vector<float> & elig_s = (*i).second;
    for (int j = 0; j < numactions; j++){
      if (elig_s[j] > 0.0){
        if (ELIGDEBUG){
          cout << "updating state " << (*((*i).first))[0] << ", " << (*((*i).first))[1] << " act: " << j << " with elig: " << elig_s[j] << endl;
        }
        // update
        Q[si][j] += alpha * elig_s[j] * (r - Q[si][j]);
        elig_s[j] = 0.0;
      }
    }  
  }
  
}

Sarsa::state_t Sarsa::canonicalize(const std::vector<float> &s) {
  const std::pair<std::set<std::vector<float> >::iterator, bool> result =
    statespace.insert(s);
  state_t retval = &*result.first; // Dereference iterator then get pointer 
  if (result.second) { // s is new, so initialize Q(s,a) for all a
    std::vector<float> &Q_s = Q[retval];
    Q_s.resize(numactions,initialvalue);
    std::vector<float> &elig = eligibility[retval];
    elig.resize(numactions,0);
  }
  return retval; 
}



  std::vector<float>::iterator
Sarsa::random_max_element(
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




void Sarsa::setDebug(bool d){
  ACTDEBUG = d;
}


void Sarsa::printState(const std::vector<float> &s){
  for (unsigned j = 0; j < s.size(); j++){
    cout << s[j] << ", ";
  }
}



void Sarsa::seedExp(std::vector<experience> seeds){

  // for each seeding experience, update our model
  for (unsigned i = 0; i < seeds.size(); i++){
    experience e = seeds[i];
     
    std::vector<float> &Q_s = Q[canonicalize(e.s)];
    
    // Get q value for action taken
    const std::vector<float>::iterator a = Q_s.begin() + e.act;

    // Update value of action just executed
    Q_s[e.act] += alpha * (e.reward + gamma * (*a) - Q_s[e.act]);
    
 
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

void Sarsa::logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax){
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


float Sarsa::getValue(std::vector<float> state){

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


void Sarsa::savePolicy(const char* filename){

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


