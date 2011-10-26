/** \file MultipleClassifiers.cc
    Implements the Multiple Classifiers class.
    \author Todd Hester
*/

#include "MultipleClassifiers.hh"



MultipleClassifiers::MultipleClassifiers(int id, int modelType, int predType,
                                         int nModels, int trainMode,
                                         int trainFreq,
                                         float featPct, float expPct,
                                         float treeThreshold, bool stoch,
                                         Random rng):
  id(id), modelType(modelType), predType(predType), nModels(nModels),
  mode(trainMode), freq(trainFreq),
  featPct(featPct), expPct(expPct), 
  treeThresh(treeThreshold), stoch(stoch), 
  addNoise(!stoch && (modelType == M5MULTI || modelType == M5SINGLE || modelType == M5ALLMULTI || modelType == M5ALLSINGLE || modelType == LSTMULTI || modelType == LSTSINGLE)),
  rng(rng)
{
  STDEBUG = false;//true;
  ACC_DEBUG = false;//true;
  PRED_DEBUG = false; //true;
  CONF_DEBUG = false;
  COPYDEBUG = false;
  nsteps = 0;
  
  cout << "Created MultClass " << id << " with nModels: " << nModels << ", addNoise: " << addNoise << endl;

  for (int i = -1; i < id; i++)
    rng.uniform(0,1);


  initModels();

}

MultipleClassifiers::MultipleClassifiers(const MultipleClassifiers &t):
  id(t.id), modelType(t.modelType), predType(t.predType), nModels(t.nModels),
  mode(t.mode), freq(t.freq),
  featPct(t.featPct), expPct(t.expPct), 
  treeThresh(t.treeThresh), stoch(t.stoch), addNoise(t.addNoise),
  rng(t.rng)
{
  COPYDEBUG = t.COPYDEBUG;
  if (COPYDEBUG) cout << "  MC copy constructor id " << id << endl;
  STDEBUG = t.STDEBUG;
  ACC_DEBUG = t.ACC_DEBUG;
  PRED_DEBUG = t.PRED_DEBUG;
  CONF_DEBUG = t.CONF_DEBUG;
  nsteps = t.nsteps;

  accuracy = t.accuracy;

  infos.resize(nModels);
  models.resize(nModels);
  if (COPYDEBUG) cout << "models size now " << models.size() << " nModels: " << nModels << endl;
  // copy each model
  for (unsigned i = 0; i < models.size(); i++){
    if (COPYDEBUG)cout << "MC copy model " << i << endl;
    models[i] = t.models[i]->getCopy();
  }

  if (COPYDEBUG) cout << "MC copy complete models size: " << models.size() << endl;
}

MultipleClassifiers* MultipleClassifiers::getCopy(){
  
  MultipleClassifiers* copy = new MultipleClassifiers(*this);
  return copy;

}

MultipleClassifiers::~MultipleClassifiers() {
  for (unsigned i = 0; i < models.size(); i++){
    delete models[i];
  }
  models.clear();
  accuracy.clear();
  infos.clear();
}


bool MultipleClassifiers::trainInstances(std::vector<classPair> &instances){
  if (STDEBUG) cout << id << " MultClass trainInstances: " << instances.size() << endl;

  bool changed = false;
  
  std::vector< std::vector<classPair> >subsets(nModels);
  for (int i = 0; i < nModels; i++){
    subsets[i].reserve(instances.size());
  }

  for (unsigned j = 0; j < instances.size(); j++){
    bool didUpdate = false;
    
    // train each model
    for (int i = 0; i < nModels; i++){
      
      // check accuracy of this model
      if (predType == BEST || predType == WEIGHTAVG){
	for (unsigned j = 0; j < instances.size(); j++){
	  updateModelAccuracy(i, instances[j].in, instances[j].out);
	}
      }

      // create new vector with some random subset of experiences
      if (rng.uniform() < expPct){
	float origOutput = instances[j].out;
	if (addNoise) instances[j].out += rng.uniform(-0.2,0.2)*treeThresh;
        subsets[i].push_back(instances[j]);
	if (addNoise) instances[j].out = origOutput;
	didUpdate = true;
      }
    } // model loop
    
    // make sure someone got this instance
    if (!didUpdate){
      int model = rng.uniformDiscrete(0,nModels-1);
      //cout << "none got instance, update model " << model << endl;
      if (addNoise) instances[j].out += rng.uniform(-0.2,0.2)*treeThresh;
      subsets[model].push_back(instances[j]);
    }
  } // instances loop
    
  // now update models
  for (int i = 0; i < nModels; i++){
    if (subsets[i].size() > 0){
      if (STDEBUG) cout << id << " train model " << i << " on subset of size "
                        << subsets[i].size() << endl;
      bool singleChange = models[i]->trainInstances(subsets[i]);
      changed = changed || singleChange;
    }
  }
  
  nsteps += instances.size();

  return changed;
}

// here the target output will be a single value
bool MultipleClassifiers::trainInstance(classPair &instance){
  if (STDEBUG) cout << id << " trainInstance" << endl;

  bool changed = false;

  // train each model
  bool didUpdate = false;
  for (int i = 0; i < nModels; i++){

    // check accuracy of this model
    if (predType == BEST || predType == WEIGHTAVG){
      updateModelAccuracy(i, instance.in, instance.out);
    }

    // update with new experience
    if (rng.uniform() < expPct){
      didUpdate = true;
      float origOutput = instance.out;
      if (addNoise) instance.out += rng.uniform(-0.2,0.2)*treeThresh;
      bool singleChange = models[i]->trainInstance(instance);
      if (addNoise) instance.out = origOutput;
      changed = changed || singleChange;
    }
  }

  // make sure some model got the transition
  if (!didUpdate){
    int model = rng.uniformDiscrete(0,nModels-1);
    if (addNoise) instance.out += rng.uniform(-0.2,0.2)*treeThresh;
    bool singleChange = models[model]->trainInstance(instance);
    changed = singleChange || changed;
  }

  nsteps++;

  return changed;

}


// get all the models outputs and combine them somehow
void MultipleClassifiers::testInstance(const std::vector<float> &input, std::map<float, float>* retval){
  if (STDEBUG) cout << id << " testInstance" << endl;

  if ((int)infos.size() != nModels){
    infos.resize(nModels);
  }

  for (unsigned i = 0; i < infos.size(); i++){
    infos[i].clear();
  }

  retval->clear();

  // possibly have to init our model
  if ((int)models.size() != nModels)
    initModels();

  ///////////////////////////////////
  // return best
  if (predType == BEST){
    float acc = -1.0;
    int best = -1;
    for (int i = 0; i < nModels; i++){
      if (PRED_DEBUG) cout << "Model " << i << " has acc "
                           << accuracy[i] << endl;
      if (accuracy[i] > acc){
        acc = accuracy[i];
        best = i;
      }
    }
    if (PRED_DEBUG) cout << id << " Returning model " << best
                         << " with accuracy " << acc << endl;
    models[best]->testInstance(input, retval);
    return;
  }
  //////////////////////////////////////


  /////////////////////////////////////////
  // calculate weights for weighted avg
  std::vector<float> weights(nModels, 1.0);
  if (predType == WEIGHTAVG){
    float accSum = 0.0;
    
    for (int j = 0; j < nModels; j++){
      accSum += accuracy[j];
    }

    if (accSum > 0.0){
      for (int j  = 0; j < nModels; j++){
        weights[j] = (float)nModels * accuracy[j] / accSum;
        if (PRED_DEBUG || ACC_DEBUG) cout << "Model " << j << " acc: "
                                          << accuracy[j] << " weight: "
                                          << weights[j] << endl;
      }
    }
  }
  /////////////////////////////////////////

  // get state action info from each tree in our set
  for (int j = 0; j < nModels; j++){
    models[j]->testInstance(input, &(infos[j]));
  }

  // make a list of all outcomes any model predicted
  for (int i = 0; i < nModels; i++){
    for (std::map<float, float>::iterator it = infos[i].begin();
         it != infos[i].end(); it++){

      float outcome = (*it).first;
      float prob = (*it).second;

      // only if prob > 0
      if (prob > 0){

        // see if its not already in there
        if (!retval->count(outcome)){
          if (PRED_DEBUG) cout << " new outcome " << outcome << endl;
          // check if it really is a new one
          for (std::map<float, float>::iterator j = retval->begin();
               j != retval->end(); j++){
            if (fabs(outcome - j->first) < treeThresh){
              if (PRED_DEBUG) cout << "within treeThresh of " << j->first << endl;
              outcome = j->first;
              break;
            }
          }
        }

        (*retval)[outcome] += (prob * weights[i] / (float)nModels);
        
        if (PRED_DEBUG) cout << "inserting outcome " << outcome
                             << " from model " << i
                             << " with prob: " << prob
                             << " and weight: " << weights[i]
                             << " now total prob: " << (*retval)[outcome] << endl;
        
      }
    }
  }
}



float MultipleClassifiers::getConf(const std::vector<float> &input){
  if (STDEBUG || CONF_DEBUG) cout << id << " getConf" << endl;

  if ((int)infos.size() != nModels){
    infos.resize(nModels);
  }

  // get predictions if we haven't
  if (predType == BEST || predType == SEPARATE){
    if (CONF_DEBUG) cout << "conf: getting predictions again" << endl;
    for (int j = 0; j < nModels; j++){
      infos[j].clear();
      models[j]->testInstance(input, &(infos[j]));
    }
  }

  float conf = 0;

  // for deterministic trees, calculating a distribution of outcomes
  // calculate kl divergence
  if (modelType == RMAX || modelType == SLF || modelType == C45TREE ||
      modelType == STUMP ){ 
    conf = 1.0 - klDivergence(input);
  }

  // for continuous trees, providing a single continuous prediction
  // calcluate variance
  else {
    conf = 1.0 - variance(input);
  }

  if (CONF_DEBUG) cout << "return conf: " << conf << endl;

  return conf;

}

// calculate kl divergence of these predictions
float MultipleClassifiers::klDivergence(const std::vector<float> &input){

  // KL divergence equation
  // D_KL(P||Q) = SUM_i P(i) log (P(i)/Q(i))
  float totalKL = 0.0;

  // I guess let's take the avg KL divergence of comparisons 
  // between all pairs of models
  for (int i = 0; i < nModels; i++){
    for (int j = 0; j < nModels; j++){
      if (i == j) continue;
      float singleKL = 0.0;

      for (std::map<float, float>::iterator it = infos[i].begin();
           it != infos[i].end(); it++){
        
        float outcome = (*it).first;
        float prob = (*it).second;
        
        if (prob == 0){
          continue;
        }

        if (CONF_DEBUG) 
          cout << "model " << i << " predicts " << outcome << " with prob " 
               << prob << ", model " << j << " has prob " 
               << infos[j][outcome] << endl;

        float jProb = infos[j][outcome];
        if (jProb == 0) jProb = 0.01;

        singleKL += prob * log(prob / jProb);
        
      }
      totalKL += singleKL;

      if (CONF_DEBUG) 
        cout << "D_KL(" << i << "||" << j << ") = " << singleKL
             << " klsum: " << totalKL << endl;

    } // outcomes of model i
  } // models

  int npairs = nModels * (nModels - 1);
  float avgKL = totalKL / (float)npairs;

  // normalize.  since my 0 prob is really 0.01, and log 1/0.01 = 4.60517, we'll divide by that
  avgKL /= 4.60517;

  if (CONF_DEBUG) cout << "AvgKL: " << avgKL << endl << endl;

  return avgKL;

}

// calculate the variance of these predictions
float MultipleClassifiers::variance(const std::vector<float> &input){

  float sum = 0.0;
  float sumSqr = 0.0;

  for (int i = 0; i < nModels; i++){
    float val = infos[i].begin()->first;
    sum += val;
    sumSqr += (val*val);
  }

  float mean = sum / (float)nModels;
  float variance = (sumSqr - sum*mean) / (float)(nModels-1.0);

  if (CONF_DEBUG) cout << "variance of predictions is " << variance << endl;

  return variance;

}

// init models
void MultipleClassifiers::initModels(){
  if (STDEBUG) cout << "initModels()" << endl;

  models.resize(nModels);
  accuracy.resize(nModels, 0.0);

  // init the trees or stumps
  for (int i = 0; i < nModels; i++){

    if (modelType == C45TREE){
      models[i] = new C45Tree(id + i*(1+nModels), mode, freq, 0, featPct, rng);
    }
    else if (modelType == M5MULTI){
      models[i] = new M5Tree(id + i*(1+nModels), mode, freq, 0, featPct, false, false, treeThresh, rng);
    }
    else if (modelType == M5ALLMULTI){
      models[i] = new M5Tree(id + i*(1+nModels), mode, freq, 0, featPct, false, true, treeThresh, rng);
    }
    else if (modelType == M5ALLSINGLE){
      models[i] = new M5Tree(id + i*(1+nModels), mode, freq, 0, featPct, true, true, treeThresh, rng);
    }
    else if (modelType == M5SINGLE){
      models[i] = new M5Tree(id + i*(1+nModels), mode, freq, 0, featPct, true, false, treeThresh, rng);
    }
    else if (modelType == LSTSINGLE){
      models[i] = new LinearSplitsTree(id + i*(1+nModels), mode, freq, 0, featPct, true, treeThresh, rng);
    }
    else if (modelType == LSTMULTI){
      models[i] = new LinearSplitsTree(id + i*(1+nModels), mode, freq, 0, featPct, false, treeThresh, rng);
    }
    else if (modelType == STUMP){
      models[i] = new Stump(id + 1*(1+nModels), mode, freq, 0, featPct, rng);
    }
    else if (modelType == ALLM5TYPES){
      // select an m5 type randomly.  so multivariate v single and allfeats v subtree feats
      bool simple = rng.bernoulli(0.5);
      bool allFeats = rng.bernoulli(0.5);
      //cout << "ALL types init tree " << i << " with simple: " << simple << " and allFeats: " << allFeats << endl;
      models[i] = new M5Tree(id + i*(1+nModels), mode, freq, 0, featPct, simple, allFeats, treeThresh, rng);
    }
    else {
      cout << "Invalid model type for this committee" << endl;
      exit(-1);
    }
  }

}


void MultipleClassifiers::updateModelAccuracy(int i, const std::vector<float> &in,
                                              float output){

  if (ACC_DEBUG) cout << "Check model accuracy " << i << " curr: " << accuracy[i] << endl;

  // get prediction
  std::map<float, float> pred;
  models[i]->testInstance(in, &pred);

  // see if it was correct
  bool correct = true;
  // check trans prob
  // for now, just see if prob of next was ML from this model
  float prob = pred[output];
  if (ACC_DEBUG) cout << "Model " << i << " predicted outcome "
                      << output << " with prob: " << prob << endl;
  if (prob <= 0.5)
    correct = false;

  // update accuracy
  accuracy[i] = accuracy[i] * (float)nsteps/(float)(nsteps+1.0);
  if (correct)
    accuracy[i] += 1.0/(float)(nsteps+1.0);

  if (ACC_DEBUG) cout << "Model " << i << " new accuracy: " << accuracy[i] << endl;

}

