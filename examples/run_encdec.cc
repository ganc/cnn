#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3) {
    cerr << "argc: " << argc;
    cerr << "Usage: " << argv[0] << " input.txt model.params\n";
    return 1;
  }

  //RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
  //EncoderDecoder<SimpleRNNBuilder> lm(model);
  EncoderDecoder<LSTMBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }
