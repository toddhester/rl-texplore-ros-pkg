#include <rl_env/stocks.hh>


Stocks::Stocks(Random &rand, bool stochastic, int nsectors, int nstocks):
  nsectors(nsectors),
  nstocks(nstocks),
  noisy(stochastic),
  rng(rand),
  s(nsectors*nstocks + nsectors)
{
  STOCK_DEBUG = false; //true;

  initStocks();
  reset();
}


Stocks::Stocks(Random &rand, bool stochastic):
  nsectors(3),
  nstocks(2),
  noisy(stochastic),
  rng(rand),
  s(nsectors*nstocks + nsectors)
{
  STOCK_DEBUG = false; //true;

  initStocks();
  reset();
}


Stocks::~Stocks() {
  delete [] owners;
  for (int i = 0; i < nsectors; i++){
    delete [] rising[i];
  }
  delete [] rising;
}

const std::vector<float> &Stocks::sensation() const { return s; }

float Stocks::apply(int action) {

  // figure out ownership of stocks with this action
  if (action < nsectors){
    // flip that bit of sensation array
    s[owners[action]] = !s[owners[action]];
  }
  else if (action == nsectors){
    // do nothing
  }
  else {
    cout << "Invalid action!" << endl;
  }

  if (STOCK_DEBUG){
    cout << "Action: " << action << " Ownership now: ";
    for (int i = 0; i < nsectors; i++){
      cout << s[owners[i]] << ", ";
    }
    cout << endl;
  }

  float r = reward();

  calcStockRising();

  return r;

}

void Stocks::calcStockRising() {

  // for each sector
  for (int i = 0; i < nsectors; i++){
    // get average of stocks
    float sum = 0;
    for (int j = 0; j < nstocks; j++){
      sum += s[rising[i][j]];
    }
    float riseProb = 0.1 + 0.8 * (sum / (float)nstocks);

    // set value for each stock
    for (int j = 0; j < nstocks; j++){
      if (noisy) {
        s[rising[i][j]] = rng.bernoulli(riseProb);
      } else {
        s[rising[i][j]] = (riseProb > 0.5);
      }
    }

    if (STOCK_DEBUG){
      cout << "S" << i << " Prob: " << riseProb << " stocks now: ";
      for (int j = 0; j < nstocks; j++){
        cout << s[rising[i][j]] << ", ";
      }
      cout << endl;
    }

  }

}

float Stocks::reward() {

  float sum = 0.0;

  // for each sector
  for (int i = 0; i < nsectors; i++){
    // skip if we don't own it
    if (!s[owners[i]])
      continue;

    if (STOCK_DEBUG) cout << "Own sector " << i << ": ";


    // add 1 for each rising, sub 1 for each not
    for (int j = 0; j < nstocks; j++){
      if (s[rising[i][j]]){
        sum++;
        if (STOCK_DEBUG) cout << "rising, ";
      } else {
        sum--;
        if (STOCK_DEBUG) cout << "falling, ";
      }
    }

    if (STOCK_DEBUG) cout << endl;
  }

  if (STOCK_DEBUG)
    cout << "Reward is " << sum << endl;

  return sum;

}

bool Stocks::terminal() const {
  return false;
}

void Stocks::initStocks(){
  // set owners and rising variables
  owners = new int[nsectors];
  rising = new int*[nsectors];
  for (int i = 0; i < nsectors; i++){
    rising[i] = new int[nstocks];
    owners[i] = i;
    if (STOCK_DEBUG)
      cout << "Owners[" << i << "] is " << owners[i] << endl;
    for (int j = 0; j < nstocks; j++){
      rising[i][j] = nsectors + (i*nstocks) + j;
      if (STOCK_DEBUG)
        cout << "Rising[" << i << "][" << j << "] is " << rising[i][j] << endl;
    }
  }
}

void Stocks::reset() {

  // random values for each variable
  for (unsigned i = 0; i < s.size(); i++){
    s[i] = rng.uniformDiscrete(0,1);
  }
}


int Stocks::getNumActions() {
  return nsectors+1;
}

void Stocks::setSensation(std::vector<float> sIn){
  for (unsigned i = 0; i < s.size(); i++){
    s[i] = sIn[i];
  }
}


void Stocks::getMinMaxFeatures(std::vector<float> *minFeat,
                               std::vector<float> *maxFeat){

  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 1.0);

}

void Stocks::getMinMaxReward(float *minR,
                             float *maxR){

  *minR = -(nsectors*nstocks);
  *maxR = (nsectors*nstocks);
}
