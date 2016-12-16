#ifndef _LINEARSPLITS_HH_
#define _LINEARSPLITS_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>
#include <set>
#include <map>

#define N_LST_EXP 200000
#define N_LS_NODES 2500

#define BUILD_EVERY 0
#define BUILD_ON_ERROR 1
#define BUILD_EVERY_N 2
#define BUILD_ON_TERMINAL 3
#define BUILD_ON_TERMINAL_AND_ERROR 4

/** M5 decision tree class */
class LinearSplitsTree: public Classifier {

public:

  // mode - re-build tree every step?  
  // re-build only on misclassifications? or rebuild every 'trainFreq' steps
  LinearSplitsTree(int id, int trainMode, int trainFreq, int m, 
                   float featPct, bool simple, float min_er, Random rng);

  LinearSplitsTree(const LinearSplitsTree&);
  virtual LinearSplitsTree* getCopy();

  ~LinearSplitsTree();

  // structs to be defined
  struct tree_node;
  struct tree_experience;
  
    
  /** Tree node struct */
  struct tree_node {
    int id;

    // split criterion
    int dim;
    float val;
    float avgError;

    bool leaf;
    
    // for regression model
    float constant;
    std::vector<float> coefficients;

    // next nodes in tree
    tree_node *l;
    tree_node *r;

    // set of all outputs seen at this leaf/node
    int nInstances;

  };

  struct tree_experience {
    std::vector<float> input;
    float output;
  };

  bool trainInstance(classPair &instance);
  bool trainInstances(std::vector<classPair> &instances);
  void testInstance(const std::vector<float> &input, std::map<float, float>* retval);
  float getConf(const std::vector<float> &input);

  void buildTree(tree_node* node, const std::vector<tree_experience*> &instances,
                 bool changed);
  void copyTree(tree_node* newNode, tree_node* origNode);


  // helper functions
  void initTree();
  void rebuildTree();
  void initTreeNode(tree_node* node);
  tree_node* traverseTree(tree_node* node, const std::vector<float> &input);
  tree_node* getCorrectChild(tree_node* node, const std::vector<float> &input);
  bool passTest(int dim, float val, const std::vector<float> &input);
  float calcER(int dim, float val, 
               const std::vector<tree_experience*> &instances, float error,
               std::vector<tree_experience*> &left,
               std::vector<tree_experience*> &right,
               float *leftError, float *rightError);
  float* sortOnDim(int dim, const std::vector<tree_experience*> &instances);
  std::set<float> getUniques(int dim, const std::vector<tree_experience*> &instances, float & minVal, float& maxVal);
  void deleteTree(tree_node* node);
  float calcAvgErrorforSet(const std::vector<tree_experience*> &instances);
  void printTree(tree_node *t, int level);
  void testPossibleSplits(float avgError, const std::vector<tree_experience*> &instances, 
                          float *bestER, int *bestDim, 
                          float *bestVal, 
                          std::vector<tree_experience*> *bestLeft, 
                          std::vector<tree_experience*> *bestRight,
                          float *bestLeftError, float *bestRightError);
  void implementSplit(tree_node* node, const std::vector<tree_experience*> &instances, 
                      float bestER, int bestDim,
                      float bestVal, 
                      const std::vector<tree_experience*> &left, 
                      const std::vector<tree_experience*> &right,
                      bool changed, float leftError, float rightError);
  void compareSplits(float er, int dim, float val, 
                     const std::vector<tree_experience*> &left, 
                     const std::vector<tree_experience*> &right,
                     int *nties, float leftError, float rightError,
                     float *bestER, int *bestDim, 
                     float *bestVal, 
                     std::vector<tree_experience*> *bestLeft, 
                     std::vector<tree_experience*> *bestRight,
                     float *bestLeftError, float *bestRightError);
  void leafPrediction(tree_node *t, const std::vector<float> &in, std::map<float, float>* retval);
  void makeLeaf(tree_node* node, const std::vector<tree_experience*> &instances);

  float fitSimpleLinearModel(const std::vector<tree_experience*> &instances,
                             float* constant, std::vector<float> *coeff);
  float fitMultiLinearModel(const std::vector<tree_experience*> &instances,
                            float* constant, std::vector<float> * coeff);

  tree_node* allocateNode();
  void deallocateNode(tree_node* node);
  void initNodes();

  bool INCDEBUG;
  bool DTDEBUG;
  bool LMDEBUG;
  bool SPLITDEBUG;
  bool STOCH_DEBUG;
  bool NODEDEBUG;
  int nExperiences;
  bool COPYDEBUG;

  float SPLIT_MARGIN;

private:

  const int id;
  
  const int mode;
  const int freq;
  const int M;
  const float featPct; 
  const bool SIMPLE;
  const float MIN_ER;

  Random rng;

  int nOutput;
  int nnodes;
  bool hadError;
  int totalnodes;
  int maxnodes;

  // INSTANCES
  std::vector<tree_experience*> experiences;
  tree_experience allExp[N_LST_EXP];
  tree_node allNodes[N_LS_NODES];
  std::vector<int> freeNodes;

  // TREE
  tree_node* root;
  tree_node* lastNode;

};


#endif
  
