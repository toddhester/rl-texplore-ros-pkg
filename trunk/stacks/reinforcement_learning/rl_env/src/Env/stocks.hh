#ifndef _STOCKS_H_
#define _STOCKS_H_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <set>


class Stocks: public Environment {
public:
  /** Creates a Stocks domain using the specified map.
      \param rand Random number generator to use.
      \param stochastic Whether to use nondeterministism. 
      \param nsectors number of stock sectors
      \param nstocks number of stocks per sector */
  Stocks(Random &rand, bool stochastic, int nsectors, int nstocks);

  /** Creates a deterministic Stocks domain.
      \param rand Random number generator used.
      \param stochastic Whether to use nondeterministism. */
  Stocks(Random &rand, bool stochastic);

  virtual ~Stocks();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();
  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual bool isEpisodic() { return false; };
  virtual void getMinMaxReward(float* minR, float* maxR);

  void calcStockRising();
  float reward();
  void setSensation(std::vector<float> s);
  void initStocks();
  
protected:



private:

  const int nsectors;
  const int nstocks;
  const bool noisy;
  Random &rng;

  std::vector<float> s;

  // lets just have an index into the array
  int** rising;
  int* owners;

  bool STOCK_DEBUG;

};

#endif
