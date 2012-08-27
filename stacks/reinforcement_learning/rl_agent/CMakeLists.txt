cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

rosbuild_add_library(agentlib
  src/Agent/DiscretizationAgent.cc 
  src/Agent/QLearner.cc
  src/Agent/ModelBasedAgent.cc
  src/Agent/SavedPolicy.cc
  src/Agent/Dyna.cc
  src/Agent/Sarsa.cc
  src/Models/FactoredModel.cc
  src/Models/M5Tree.cc
  src/Models/LinearSplitsTree.cc
  src/Models/C45Tree.cc
  src/Models/Stump.cc
  src/Models/MultipleClassifiers.cc
  src/Models/ExplorationModel.cc
  src/Models/RMaxModel.cc 
  src/Models/SepPlanExplore.cc 
  src/Planners/ValueIteration.cc
  src/Planners/PolicyIteration.cc
  src/Planners/PrioritizedSweeping.cc 
  src/Planners/ETUCT.cc
  src/Planners/PO_ParallelETUCT.cc
  src/Planners/ParallelETUCT.cc
  src/Planners/PO_ETUCT.cc
  src/Planners/MBS.cc
  src/newmat/newmat1.cc 
  src/newmat/newmat2.cc 
  src/newmat/newmat3.cc 
  src/newmat/newmat4.cc 
  src/newmat/newmat5.cc 
  src/newmat/newmat6.cc 
  src/newmat/newmat7.cc 
  src/newmat/newmat8.cc 
  src/newmat/newmatex.cc 
  src/newmat/bandmat.cc 
  src/newmat/submat.cc 
  src/newmat/myexcept.cc 
  src/newmat/cholesky.cc 
  src/newmat/evalue.cc 
  src/newmat/fft.cc 
  src/newmat/hholder.cc 
  src/newmat/jacobi.cc 
  src/newmat/newfft.cc 
  src/newmat/sort.cc 
  src/newmat/svd.cc 
  src/newmat/newmatrm.cc 
  src/newmat/newmat9.cc
)


rosbuild_add_executable(agent
  src/agent.cpp
)

target_link_libraries(agent agentlib)
