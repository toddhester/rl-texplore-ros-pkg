/**
 The LightWorld domain from 
 "Building Portable Options: Skill Transfer in Reinforcement Learning"
 by Konidaris and Barto
*/

#ifndef _LIGHTWORLD_H_
#define _LIGHTWORLD_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>


class LightWorld: public Environment {
public:
  /** Creates a PlayRoom domain using the specified map.
      Puts objects in random locations.
      \param rand Random number generator to use.
      \param stochastic Whether to use nondeterministic actions 
      \param nrooms Number of rooms to have */
  LightWorld(Random &rand, bool stochastic, int nrooms);

  virtual ~LightWorld();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();
  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual bool isEpisodic() { return false; };
  virtual void getMinMaxReward(float* minR, float* maxR);


  friend std::ostream &operator<<(std::ostream &out, const LightWorld &playroom);


  void resetKey();
  void setKey(std::vector<float> testS);

  struct room_info {
    int height;
    int width;
    int key_ns;
    int key_ew;
    int lock_ns;
    int lock_ew;
    int door_ns;
    int door_ew;
  };

  typedef std::pair<float,float> coord_t;
  enum lightworld_action_t {NORTH, EAST, WEST, SOUTH, PICKUP, PRESS, NUM_ACTIONS};

  bool LWDEBUG;

  const bool noisy;
  int nrooms;
  Random &rng;

  std::vector<float> s;
  float& ns;
  float& ew;
  float& have_key;
  float& door_open;
  float& room_id;
  float& key_n;
  float& key_e;
  float& key_w;
  float& key_s;
  float& lock_n;
  float& lock_e;
  float& lock_w;
  float& lock_s;
  float& door_n;
  float& door_e;
  float& door_w; 
  float& door_s;

  std::vector<room_info> rooms;

  void updateSensors();
  int applyNoise(int action);

  /** Prints the current state */
  void print_state() const;

  /** Prints the current map. */
  void print_map() const;

  void updateVisits();

  int totalVisited;
  int keyVisited;
  int lockVisited;
  int doorVisited;
  int haveKey;
  int doorOpen;
  int leaveRoom;
  int pressKey;
  int pressLockCorrect;
  int pressLockIncorrect;
  int pressDoor;
  int pressOther;
  int pickupKeyCorrect;
  int pickupKeyIncorrect;
  int pickupLock;
  int pickupDoor;
  int pickupOther;

  int MAX_SENSE;

};

#endif
