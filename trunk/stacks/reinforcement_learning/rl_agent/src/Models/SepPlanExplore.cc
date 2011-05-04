#include "SepPlanExplore.hh"



SepPlanExplore::SepPlanExplore(int id, int modelType, int predType,
                               int nModels, int trainMode,
                               int trainFreq,
                               float featPct, float expPct,
                               float treeThreshold, bool stoch,
                               Random rng):
  id(id), modelType(modelType), predType(predType), nModels(nModels),
  mode(trainMode), freq(trainFreq),
  featPct(featPct), expPct(expPct),
  treeThresh(treeThreshold), stoch(stoch),
  rng(rng)
{
  SPEDEBUG = false;//true;

  cout << "Created Sep Plan & Explore models " << id << " with nModels: " << nModels << endl;

  for (int i = 0; i < id; i++)
    rng.uniform(0,1);

  initModels();

}

SepPlanExplore::SepPlanExplore(const SepPlanExplore& spe):
  id(spe.id), modelType(spe.modelType), 
  predType(spe.predType), nModels(spe.nModels),
  mode(spe.mode), freq(spe.freq),
  featPct(spe.featPct), expPct(spe.expPct),
  treeThresh(spe.treeThresh), stoch(spe.stoch),
  rng(spe.rng)
{
  cout << "spe get copy" << endl;
  SPEDEBUG = spe.SPEDEBUG;
  expModel = spe.expModel->getCopy();
  planModel = spe.planModel->getCopy();
}

SepPlanExplore* SepPlanExplore::getCopy(){
  SepPlanExplore* copy = new SepPlanExplore(*this);
  return copy;
}

SepPlanExplore::~SepPlanExplore() {
  delete expModel;
  delete planModel;
}


bool SepPlanExplore::trainInstances(std::vector<classPair> &instances){
  if (SPEDEBUG) cout << id << "SPE trainInstances: " << instances.size() << endl;

  // train both
  bool expChanged = expModel->trainInstances(instances);
  bool planChanged = planModel->trainInstances(instances);

  return (expChanged || planChanged);

}



// here the target output will be a single value
bool SepPlanExplore::trainInstance(classPair &instance){
  if (SPEDEBUG) cout << id << "SPE trainInstance: " << endl;

  // train both
  bool expChanged = expModel->trainInstance(instance);
  bool planChanged = planModel->trainInstance(instance);

  return (expChanged || planChanged);

}

// get all the models outputs and combine them somehow
void SepPlanExplore::testInstance(const std::vector<float> &input, std::map<float, float>* retval){
  if (SPEDEBUG) cout << id << " testInstance" << endl;

  retval->clear();

  // use planning model to get predictions
  planModel->testInstance(input, retval);

}



float SepPlanExplore::getConf(const std::vector<float> &input){
  if (SPEDEBUG) cout << "getConf" << endl;

  // use exploreation model for confidence measure
  return expModel->getConf(input);

}


// init models
void SepPlanExplore::initModels(){
  if (SPEDEBUG) cout << "initModels()" << endl;

  if (nModels < 2){
    cout << "Should really use Sep plan & explore models with multiple models" << endl;
    exit(-1);
  }

  // explore model should be of type MultipleClassifiers
  expModel = new MultipleClassifiers(id, modelType, predType,
                                     nModels, mode, freq,
                                     featPct, expPct, treeThresh, stoch, rng);

  // init the trees or stumps
  if (modelType == C45TREE){
    planModel = new C45Tree(id, mode, freq, 0, 0.0, rng);
  }
  else if (modelType == M5MULTI){
    planModel = new M5Tree(id, mode, freq, 0, 0.0, false, false, treeThresh, rng);
  }
  else if (modelType == M5ALLMULTI){
    planModel = new M5Tree(id, mode, freq, 0, 0.0, false, true, treeThresh, rng);
  }
  else if (modelType == M5ALLSINGLE){
    planModel = new M5Tree(id, mode, freq, 0, 0.0, true, true, treeThresh, rng);
  }
  else if (modelType == M5SINGLE){
    planModel = new M5Tree(id, mode, freq, 0, 0.0, true, false, treeThresh, rng);
  }
  else if (modelType == LSTSINGLE){
    planModel = new LinearSplitsTree(id, mode, freq, 0, 0.0, true, treeThresh, rng);
  }
  else if (modelType == LSTMULTI){
    planModel = new LinearSplitsTree(id, mode, freq, 0, 0.0, false, treeThresh, rng);
  }
  else if (modelType == STUMP){
    planModel = new Stump(id, mode, freq, 0, 0.0, rng);
  }
  else if (modelType == ALLM5TYPES){
    // select an m5 type randomly.  so multivariate v single and allfeats v subtree feats
    bool simple = rng.bernoulli(0.5);
    bool allFeats = rng.bernoulli(0.5);
    //cout << "ALL types init tree " << i << " with simple: " << simple << " and allFeats: " << allFeats << endl;
    planModel = new M5Tree(id, mode, freq, 0, 0.0, simple, allFeats, treeThresh, rng);
  }
  else {
    cout << "Invalid model type for this committee" << endl;
    exit(-1);
  }
}


