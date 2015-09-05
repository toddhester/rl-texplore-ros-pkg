/** \file M5Tree.hh
    Defines the M5 Decision tree, as described in:
    "Learning with Continuous Classes" by J.R. Quinlan
    "Inducing Model Trees for Continuous Classes" by Y. Wang and I.H. Witten
    \author Todd Hester
*/

#ifndef _M5TREE_HH_
#define _M5TREE_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>
#include <set>
#include <map>

#define N_M5_EXP 200000
#define N_M5_NODES 2500

#define BUILD_EVERY 0
#define BUILD_ON_ERROR 1
#define BUILD_EVERY_N 2
#define BUILD_ON_TERMINAL 3
#define BUILD_ON_TERMINAL_AND_ERROR 4

/** M5 regression tree class */
class M5Tree: public Classifier {

public:

  /** Default constructor 
      \param id id of the tree for debug
      \param trainMode build every step? only on errors? every freq steps?
      \param trainFreq frequency of model building if using latter mode
      \param m # of visits for a given state-action to be considered known
      \param featPct pct of features to remove from set used for each tree split
      \param simple do simple linear regression (predict from one variable) rather than full multivariate linear regression
      \param allowAllFeats all linear regression to use all features, regardless of if they were in the subtree being replaced
      \param min_sdr Minimum standard deviation reduction for a split to be implemented.
      \param rng Random Number Generator 
  */
  M5Tree(int id, int trainMode, int trainFreq, int m, 
         float featPct, bool simple, bool allowAllFeats, 
	 float min_sdr, Random rng);

  /** Copy constrcutor. */
  M5Tree(const M5Tree&);

  virtual M5Tree* getCopy();

  ~M5Tree();

  // structs to be defined
  struct tree_node;
  struct tree_experience;
  
    
  /** Tree node struct. For decision nodes, it contains split information and pointers to child nodes. For leaf nodes, the regression coefficients. */
  struct tree_node {
    int id;

    // split criterion
    int dim;
    float val;
    bool leaf;

    // next nodes in tree
    tree_node *l;
    tree_node *r;
    
    // set of all outputs seen at this leaf/node
    int nInstances;

    // for regression model
    float constant;
    std::vector<float> coefficients;

  };

  /** Experiences the tree is trained on. A vector of inputs and one float output to predict */
  struct tree_experience {
    std::vector<float> input;
    float output;
  };
  
  /** Make a copy of the subtree from origNode to newNode */
  void copyTree(tree_node* newNode, tree_node* origNode);

  virtual bool trainInstance(classPair &instance);
  virtual bool trainInstances(std::vector<classPair> &instances);
  virtual void testInstance(const std::vector<float> &input, std::map<float, float>* retval);
  virtual float getConf(const std::vector<float> &input);

  /** Build the tree with the given instances from the given tree node */
  void buildTree(tree_node* node, const std::vector<tree_experience*> &instances,  bool changed);

  // helper functions
  /** Initialize the tree */
  void initTree();

  /** Rebuild the tree */
  void rebuildTree();

  /** Initialize the tree_node struct */
  void initTreeNode(tree_node* node);

  /** Traverse the tree to a leaf for the given input */
  tree_node* traverseTree(tree_node* node, const std::vector<float> &input);

  /** Get the correct child of this node for a given input */
  tree_node* getCorrectChild(tree_node* node, const std::vector<float> &input);

  /** Determine if the input passes the test defined by dim, val, type */
  bool passTest(int dim, float val, const std::vector<float> &input);

  /** Calculate the reduction in standard deviation on each side of the proposed tree split */
  float calcSDR(int dim, float val, 
		       const std::vector<tree_experience*> &instances, float sd,
		        std::vector<tree_experience*> &left,
		        std::vector<tree_experience*> &right);

  /** Returns an array of the values of features at the index dim, sorted from lowest to highest */
  float* sortOnDim(int dim, const std::vector<tree_experience*> &instances);

  /** Get all the unique values of the features on dimension dim */
  std::set<float> getUniques(int dim, const std::vector<tree_experience*> &instances, float & minVal, float& maxVal);

  /** Delete this tree node and all nodes below it in the tree. */
  void deleteTree(tree_node* node);

  /** Calculate the standard deviation for the given vector of experiences */
  float calcSDforSet(const std::vector<tree_experience*> &instances);

  /** Print the tree for debug purposes. */
  void printTree(tree_node *t, int level);

  /** Test the possible splits for the given set of instances */
  void testPossibleSplits(const std::vector<tree_experience*> &instances, 
                          float *bestSDR, int *bestDim, 
                          float *bestVal, 
                          std::vector<tree_experience*> *bestLeft, 
                          std::vector<tree_experience*> *bestRight);
  
  /** Implement the given split at the given node */
  void implementSplit(tree_node* node, 
                      const std::vector<tree_experience*> &instances,
                      float bestSDR, int bestDim,
                      float bestVal, 
                      const std::vector<tree_experience*> &left, 
                      const std::vector<tree_experience*> &right,
                      bool changed);
  
  /** Compare the current split to determine if it is the best split. */
  void compareSplits(float sdr, int dim, float val, 
                     const std::vector<tree_experience*> &left, 
                     const std::vector<tree_experience*> &right,
                     int *nties, float *bestSDR, int *bestDim, 
                     float *bestVal, 
                     std::vector<tree_experience*> *bestLeft, 
                     std::vector<tree_experience*> *bestRight);
  
  /** Get the prediction for the given inputs at the leaf node t */
  void leafPrediction(tree_node *t, const std::vector<float> &in, std::map<float, float>* retval);
  
  /** Make the given node into a leaf node. */
  void makeLeaf(tree_node* node);

  /** Remove the children of the given node (to turn it into a regression node) */
  void removeChildren(tree_node* node);

  /** Possibly prune the tree back at this node. Compare sub tree error with error of a linear regression model. */
  void pruneTree(tree_node* node, const std::vector<tree_experience*> &instances);
  
  /** Fit a multivariate linear regression model to the given instances. */
  int fitLinearModel(tree_node* node, const std::vector<tree_experience*> &instances,
                     std::vector<bool> featureMask, int nFeats, float* resSum);

  /** Fit a simple linear regression model to the given instances. */
  int fitSimpleLinearModel(tree_node* node, const std::vector<tree_experience*> &instances,
                     std::vector<bool> featureMask, int nFeats, float* resSum);

  /** Determine the features used for splits in the given subtree */
  void getFeatsUsed(tree_node* node, std::vector<bool> *featsUsed);

  /** Allocate a new node from our pre-allocated store of tree nodes */
  tree_node* allocateNode();
  
  /** Return tree node back to store of nodes */
  void deallocateNode(tree_node* node);
  
  /** Initialize our store of tree nodes */
  void initNodes();

  bool INCDEBUG;
  bool DTDEBUG;
  bool LMDEBUG;
  bool SPLITDEBUG;
  bool STOCH_DEBUG;
  bool NODEDEBUG;
  bool COPYDEBUG;
  int nExperiences;

  float SPLIT_MARGIN;

private:

  const int id;
  
  const int mode;
  const int freq;
  const int M;
  float featPct; 
  const bool SIMPLE;
  const bool ALLOW_ALL_FEATS;
  const float MIN_SDR;

  Random rng;

  int nfeat;

  int nOutput;
  int nnodes;
  bool hadError;
  int totalnodes;
  int maxnodes;

  // INSTANCES
  /** Vector of all experiences used to train the tree */
  std::vector<tree_experience*> experiences;
  
  /** Pre-allocated array of experiences to be filled during training. */
  tree_experience allExp[N_M5_EXP];
  
  /** Pre-allocated array of tree nodes to be used for tree */
  tree_node allNodes[N_M5_NODES];
  std::vector<int> freeNodes;

  // TREE
  /** Pointer to root node of tree. */
  tree_node* root;
  /** Pointer to last node of tree used (leaf used in last prediction made). */
  tree_node* lastNode;

};


#endif
  
