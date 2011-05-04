/** \file
    Interface for an implementation of the saved policy
    algorithm. */

#ifndef _SAVEDPOLICY_HH_
#define _SAVEDPOLICY_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>

#include <ext/hash_map>
#include <set>
#include <vector>

/** Agent that uses a saved policy from a file. */
class SavedPolicy: public Agent {
public:
  /** Standard constructor
      \param numactions The number of possible actions
       */
  SavedPolicy(int numactions, const char* filename);

  virtual ~SavedPolicy();

  virtual int first_action(const std::vector<float> &s);
  virtual int next_action(float r, const std::vector<float> &s);
  virtual void last_action(float r);
  virtual void setDebug(bool d) {};
  virtual void seedExp(std::vector<experience>);

  void loadPolicy(const char* filename);

protected:
  /** The implementation maps all sensations to a set of canonical
      pointers, which serve as the internal representation of
      environment state. */
  typedef const std::vector<float> *state_t;

  /** Produces a canonical representation of the given sensation.
      \param s The current sensation from the environment.
      \return A pointer to an equivalent state in statespace. */
  state_t canonicalize(const std::vector<float> &s);
  void printState(const std::vector<float> &s);


private:
  /** Set of all distinct sensations seen.  Pointers to elements of
      this set serve as the internal representation of the environment
      state. */
  std::set<std::vector<float> > statespace;

  /** The primary data structure of the learning algorithm, the value
      function Q.  For state_t s and int a, Q[s][a] gives the
      learned maximum expected future discounted reward conditional on
      executing action a in state s. */
  std::map<state_t, std::vector<float> > Q;

  const int numactions;

  bool ACTDEBUG;
  bool LOADDEBUG;
  bool loaded;

};

#endif
