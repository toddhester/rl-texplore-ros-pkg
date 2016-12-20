#include <rl_agent/DiscretizationAgent.hh>
#include <algorithm>

DiscretizationAgent::DiscretizationAgent(int n, Agent* a, 
                                         std::vector<float> fmin,
                                         std::vector<float> fmax,
                                         bool d){

  statesPerDim.resize(fmin.size(), n);
  initEverything(a, fmin, fmax, d);

}

DiscretizationAgent::DiscretizationAgent(std::vector<int> n, Agent* a, 
                                         std::vector<float> fmin,
                                         std::vector<float> fmax,
                                         bool d){

  if (n.size() != fmin.size()){
    cout << "ERROR: discretition vector size is different than # features" << endl;
    exit(-1);
  }

  statesPerDim = n;
  initEverything(a, fmin, fmax, d);

}

void DiscretizationAgent::initEverything(Agent* a, 
                                         std::vector<float> fmin,
                                         std::vector<float> fmax,
                                         bool d){
  agent = a;
  featmin = fmin;
  featmax = fmax;
  DEBUG = d;

  // print number of features
  int totalFeatures = 1;
  for (unsigned i = 0; i < featmin.size(); i++){
    if (DEBUG) cout << "Dim " << i << " has " 
                    << (1+statesPerDim[i]) << " values." << endl;
    totalFeatures *= (1+statesPerDim[i]);
  }
  if (DEBUG) cout << "Total # States: " << totalFeatures << endl;

}

DiscretizationAgent::~DiscretizationAgent(){
  delete agent;
}

int DiscretizationAgent::first_action(const std::vector<float> &s) {
  std::vector<float> ds = discretizeState(s);

  return agent->first_action(ds);
}

int DiscretizationAgent::next_action(float r, const std::vector<float> &s) {
  std::vector<float> ds = discretizeState(s);

  return agent->next_action((int)r, ds);
}

void DiscretizationAgent::last_action(float r) {
  return agent->last_action((int)r);
}


void DiscretizationAgent::setDebug(bool b){}

std::vector<float> DiscretizationAgent::discretizeState(const std::vector<float> &s){
  std::vector<float> ds;
  ds.resize(s.size());
  
  // since i'm sometimes doing this for discrete domains
  // want to center bins on 0, not edge on 0
  //cout << "feat " << i << " range: " << featmax[i] << " " << featmin[i] << " " << (featmax[i]-featmin[i]) << " n: " << (float)statesPerDim;
  for (unsigned i = 0; i < s.size(); i++){
    float factor = (featmax[i] - featmin[i]) / (float)statesPerDim[i];
    int bin = 0;
    if (s[i] > 0){
      bin = (int)((s[i]+factor/2) / factor);
    } else {
      bin = (int)((s[i]-factor/2) / factor);
    }
    
    ds[i] = factor*bin;
    //cout << "DA factor: " << factor << " bin: " << bin;
    //cout << " Original: " << s[i] << " Discrete: " << ds[i] << endl;
  }

  return ds;

}


void DiscretizationAgent::seedExp(std::vector<experience> seedings){
  // discretize each experience
  for (unsigned i = 0; i < seedings.size(); i++){
    experience* e = &(seedings[i]);

    e->next = discretizeState(e->next);
    e->s = discretizeState(e->s);
    e->reward = ((int)e->reward);

  }

  // and pass to internal agent
  agent->seedExp(seedings);
}

void DiscretizationAgent::savePolicy(const char* filename){
  agent->savePolicy(filename);
}
