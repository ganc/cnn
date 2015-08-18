#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

//from command prompt
unsigned LAYERS = 0;
unsigned INPUT_DIM = 0;  
unsigned HIDDEN_DIM = 0;  
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
int kSOS;
int kEOS;


string seq2str(vector<int> seq, cnn::Dict d, bool no_wrapping) {
  string str;
  vector<int>::iterator it, it_end;
  if (no_wrapping) {
      it = seq.begin() + 1;
      it_end = seq.end() - 1;
  } else {
      it = seq.begin();
      it_end = seq.end();
  }

  for (; it != it_end; ++it) {
    str.append(d.Convert(*it));
  }
  return str;
}

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_c;
  Parameters* p_R;
  Parameters* p_bias;
  Builder builder;
  explicit RNNLanguageModel(Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  } 

  // return float with score of sentence
  float GetSeqScore(const vector<int>& sent, ComputationGraph& cg) {
    const unsigned slen = sent.size() - 1;
    float score = 0;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t =  i_bias + i_R * i_y_t;
      Expression i_ydist = log_softmax(i_r_t);
    
      vector<float> ydist = as_vector(cg.incremental_forward());
      score += ydist[sent[t+1]];
    }
    return score;
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 7 && argc != 8) {
    cerr << "argc: " << argc;
    cerr << "Usage: " << argv[0] << " test.txt model.params LAYERS INPUT_DIM HIDDEN_DIM VOCAB_SIZE [dict.params]\n";
    return 1;
  }
  LAYERS = stoi(argv[3]);
  INPUT_DIM = stoi(argv[4]);
  HIDDEN_DIM = stoi(argv[5]);
  VOCAB_SIZE = stoi(argv[6]);




  time_t     start_now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&start_now);
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

  cerr << "\n[time now=" << string(buf) << "]\n";

  Model model;

  RNNLanguageModel<LSTMBuilder> lm(model);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  cerr << "Reading model from " << argv[2] << "...\n";
  string fname = argv[2];
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;

  if (argc == 8) {
    cerr << "Reading dict from " << argv[7] << "...\n";
    fname = argv[7];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> d;
  } else {
    cerr << "Reading dict from " << argv[2] << "...\n";
    ia >> d;
  }

  d.Freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();



  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<vector<int>> test;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading test data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      test.push_back(ReadSentence(line, &d));
      ttoks += test.back().size();
      if (test.back().front() != kSOS && test.back().back() != kEOS) {
        cerr << "Sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  for (uint i = 0; i < test.size(); i++) {
    
      float score = 0;
      auto& isent = test[i];

      // build graph for this instance
      ComputationGraph cg;
      score = lm.GetSeqScore(isent, cg);
      if (i % 50 == 0) {
          cerr << "##"  << seq2str(isent, d, true) << endl; 
          cerr << "Percent done: " << ((float)i)/((float)test.size()) << endl;
      }

      cout << seq2str(isent, d, true) << " ||| " << score << endl;
      
  }

  cerr << "Done." << endl;
}
