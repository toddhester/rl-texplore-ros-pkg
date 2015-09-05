#include <rl_agent/Dyna.hh>
#include <algorithm>

#include <sys/time.h>


Dyna::Dyna(int numactions, float gamma,
           float initialvalue, float alpha, int k, float ep,
		   Random rng):
  numactions(numactions), gamma(gamma),
  initialvalue(initialvalue), alpha(alpha), k(k),
  rng(rng), currentq(NULL), laststate(NULL), lastact(0)
{

  epsilon = ep;
  ACTDEBUG = false; //true; //false;
  cout << "Dyna agent with k:" << k << endl;

}

Dyna::~Dyna() {}

int Dyna::first_action(const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "First - in state: ";
    printState(s);
    cout << endl;
  }

  return getBestAction(s);

}

int Dyna::getBestAction(const std::vector<float> &s){
  //cout << "get best action" << endl;

  // for some amount of time, update based on randomly sampled experiences
  int numExp = (int)experiences.size();
  for (int i = 0; i < k && numExp > 0; i++){
    
    // update from randoml sampled action
    int exp = 0;
    if (numExp > 1)
      exp = rng.uniformDiscrete(0, numExp-1);
    //cout << count << " Update exp " << exp << endl;

    dynaExperience e = experiences[exp];

    std::vector<float> &Q_s = Q[e.s];
    if (e.term){
      Q_s[e.a] += alpha * (e.r - Q_s[e.a]);
    } else {
      std::vector<float> &Q_next = Q[e.next];
      const std::vector<float>::iterator max =
        random_max_element(Q_next.begin(), Q_next.end());
      Q_s[e.a] += alpha * (e.r + (gamma * *max) - Q_s[e.a]);
    }

  }


  // then do normal action selection
  // Get action values
  state_t st = canonicalize(s);
  std::vector<float> &Q_s = Q[st];

  // Choose an action
  const std::vector<float>::iterator a =
    rng.uniform() < epsilon
    ? Q_s.begin() + rng.uniformDiscrete(0, numactions - 1) // Choose randomly
    : random_max_element(Q_s.begin(), Q_s.end()); // Choose maximum

  // Store location to update value later
  currentq = &*a;
  laststate = st;
  lastact = a - Q_s.begin();

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

void Dyna::addExperience(float r, state_t s, bool term){

  dynaExperience e;
  e.s = laststate;
  e.a = lastact;
  e.next = s;
  e.r = r;
  e.term = term;

  experiences.push_back(e);

}

int Dyna::next_action(float r, const std::vector<float> &s) {

  if (ACTDEBUG){
    cout << "Next: got reward " << r << " in state: ";
    printState(s);
    cout << endl;
  }

  state_t st = canonicalize(s);

  addExperience(r,st,false);

  // Get action values
  std::vector<float> &Q_s = Q[st];
  const std::vector<float>::iterator max =
    random_max_element(Q_s.begin(), Q_s.end());

  // Update value of action just executed
  *currentq += alpha * (r + gamma * (*max) - *currentq);

  return getBestAction(s);

}




void Dyna::last_action(float r) {

  if (ACTDEBUG){
    cout << "Last: got reward " << r << endl;
  }

  addExperience(r,NULL,true);

  *currentq += alpha * (r - *currentq);
  currentq = NULL;
  laststate = NULL;
}

Dyna::state_t Dyna::canonicalize(const std::vector<float> &s) {
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
Dyna::random_max_element(
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




void Dyna::setDebug(bool d){
  ACTDEBUG = d;
}


void Dyna::printState(const std::vector<float> &s){
  for (unsigned j = 0; j < s.size(); j++){
    cout << s[j] << ", ";
  }
}



void Dyna::seedExp(std::vector<experience> seeds){

  // for each seeding experience, update our model
  for (unsigned i = 0; i < seeds.size(); i++){
    experience e = seeds[i];
     
    laststate = canonicalize(e.s);
    lastact = e.act;
    state_t st = canonicalize(e.next);
    std::vector<float> &Q_s = Q[laststate];
    std::vector<float> &Q_next = Q[st];
    
    // add experience
    addExperience(e.reward,st,e.terminal);

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

void Dyna::logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax){
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


float Dyna::getValue(std::vector<float> state){

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


void Dyna::savePolicy(const char* filename){

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



double Dyna::getSeconds(){
  struct timezone tz;
  timeval timeT;
  gettimeofday(&timeT, &tz);
  return  timeT.tv_sec + (timeT.tv_usec / 1000000.0);
}
