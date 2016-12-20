#include "MultipleModels.hh"
#include "NeuralNetwork.hh"
#include "DecisionTree.hh"
#include "StochDecisionTree.hh"
#include "KNN.hh"


const bool MM_DEBUG = false; //true;


MultipleModels::MultipleModels(int id, int nIn, int nOut, int modeltype, Random rng):
  id(id), nInput(nIn), nOutput(nOut), type(modeltype), rng(rng)
{

  
  if (MM_DEBUG) 
    cout << "nIn: " << nInput << " nOut: " << nOutput << endl;

  // create a model for each output variable
  createModels();

  if (MM_DEBUG)
    cout << "multiple models created." << endl;

}

MultipleModels::~MultipleModels() {}



bool MultipleModels::trainInstance(std::vector<float> input, 
				 std::vector<float> targetOutput){
  if (MM_DEBUG) cout << "Multimodel trainInstance()" << endl;
  bool modelChanged = false;

  // for each model
  for (int i = 0; i < nOutput; i++){

    std::vector<float> output;
    output.push_back(targetOutput[i]);

    // call train instance, with just its output variable
    bool singleChange = models[i]->trainInstance(input, output);
    if (singleChange)
      modelChanged = true;
  }

  return modelChanged;

}



// TODO: figure out how this will work.  for each output feature, we'll actually want a vector with the probability of each possible value
// TODO: for now, just assume deterministic and return the only outcome
std::vector<float> MultipleModels::testInstance(std::vector<float> input){
  if (MM_DEBUG) cout << "Multimodel testInstance()" << endl;

  std::vector<float> outputs;
  outputs.resize(nOutput);

  // call each model
  for (int i = 0; i < nOutput; i++){

    // combine output variables into one vector
    outputs[i] = models[i]->testInstance(input)[0];

  }

  return outputs;

}

void MultipleModels::createModels(){
  if (MM_DEBUG) cout << "createModels" << endl;

  // create a model for each output variable
  models.resize(nOutput);

  for (int i = 0; i < nOutput; i++){

    // NN
    if (type == 0){
      //models[i] = new NeuralNetwork(id*nOutput + i, nInput, 
      //				    10, 3, 0.3, 0.7, rng);
    }

    // DT
    else if (type == 1){
      if (MM_DEBUG) cout << "Creating DT for output " << i << endl;
      models[i] = new DecisionTree(id*nOutput + i, 1, 1000, rng);
      //models[i] = new StochDecisionTree(id*nOutput + i, 1, 1000, rng);
    }

    // KNN
    else {
      if (MM_DEBUG) cout << "Creating KNN for output " << i 
			 << " with k " << id+1 << endl;
      models[i] = new KNN(id*nOutput + i, id+2, rng);

    }


  }
}



