#ifndef _DISCAGENT_HH_
#define _DISCAGENT_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>

#include <vector>


class DiscretizationAgent: public Agent {
public:

  DiscretizationAgent(int statesPerDim, Agent* a, 
                      std::vector<float> featmin, std::vector<float> featmax,
                      bool d);
  DiscretizationAgent(std::vector<int> statesPerDim, Agent* a, 
                      std::vector<float> featmin, std::vector<float> featmax,
                      bool d);
  ~DiscretizationAgent();
  
  void initEverything(Agent* a, std::vector<float> fmin,
                      std::vector<float> fmax, bool d);
  virtual int first_action(const std::vector<float> &s);
  virtual int next_action(float r, const std::vector<float> &s);
  virtual void last_action(float r);
  virtual void setDebug(bool b);
  virtual void seedExp(std::vector<experience> seeds);
  virtual void savePolicy(const char* filename);

  
  std::vector<float> discretizeState(const std::vector<float> &s);
  std::vector<int> statesPerDim;
  Agent* agent;
  std::vector<float> featmin;
  std::vector<float> featmax;
  bool DEBUG;

};

#endif
