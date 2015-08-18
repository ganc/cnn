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

typedef RNNPointer NetPointer;
typedef tuple<vector<int>, float, NetPointer, vector<float>> hypoth_type;
typedef unsigned int uint;

cnn::Dict sd, td; //Translation Source and Target dictionary
int skSOS; //Source Start of Sentence
int skEOS; //Source End of Sentence
int tkSOS; //Target Start of Sentence
int tkEOS; //Target End of Sentence

//These are from argv, line up with model read in
unsigned LAYERS = 0;
unsigned INPUT_DIM = 0;
unsigned HIDDEN_DIM = 0;

// These are set by dictionary size
unsigned INPUT_VOCAB_SIZE = 0;
unsigned OUTPUT_VOCAB_SIZE = 0;

bool fin_hypoth_comp(tuple<vector<int>, float> a, tuple<vector<int>, float> b) {
  return get<1>(a) > get<1>(b);
}

bool hypoth_comp(hypoth_type a, hypoth_type b) {
  return get<1>(a) > get<1>(b);
}


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
struct EncoderDecoderForward {

  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_ie2h;
  Parameters* p_bie;
  Parameters* p_h2oe;
  Parameters* p_boe;
  Parameters* p_R;
  Parameters* p_bias;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;

  explicit EncoderDecoderForward(Model& model) :
    dec_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
    rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
    fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {


      p_ie2h = model.add_parameters({long(HIDDEN_DIM * LAYERS * 1.5), long(HIDDEN_DIM * LAYERS * 2)});
      p_bie = model.add_parameters({long(HIDDEN_DIM * LAYERS * 1.5)});
      p_h2oe = model.add_parameters({long(HIDDEN_DIM * LAYERS), long(HIDDEN_DIM * LAYERS * 1.5)});
      p_boe = model.add_parameters({long(HIDDEN_DIM * LAYERS)});
      p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
      p_c = model.add_lookup_parameters(OUTPUT_VOCAB_SIZE, {INPUT_DIM}); 
      p_R = model.add_parameters({OUTPUT_VOCAB_SIZE, HIDDEN_DIM});
      p_bias = model.add_parameters({OUTPUT_VOCAB_SIZE});
    }

  void GetCharEmbeddings() {

    cout << "\n\n\n------ INPUT LANGUAGE ------\n\n\n" << endl;

    for (int i = 0; i < INPUT_VOCAB_SIZE; i++) {
      cout << sd.Convert(i) << endl;
    }
     cout << "\n\n\n" << endl;
    
    for (int i = 0; i < INPUT_VOCAB_SIZE; i++) {
      cout << (*((p_ec->values)[i])).transpose() << endl;
    }
    std::cout.flush();

    cout << "\n\n\n------ OUTPUT LANGUAGE ------\n\n\n" << endl;
    for (int i = 0; i < OUTPUT_VOCAB_SIZE; i++) {
      cout << td.Convert(i) << endl;
    }

     cout << "\n\n\n" << endl;
    for (int i = 0; i < OUTPUT_VOCAB_SIZE; i++) {
      cout << ((*(p_c->values)[i])).transpose() << endl;
    }
  }

  
};


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  
  if (argc != 7) {
    cerr << "Usage: " << argv[0] << " model.params LAYERS INPUT_DIM HIDDEN_DIM INPUT_VOCAB_SIZE OUTPUT_VOCAB SIZE\n";
    return 1;
  }

  if (argc == 7) {
    LAYERS = stoi(argv[2]);
    INPUT_DIM = stoi(argv[3]);
    HIDDEN_DIM = stoi(argv[4]);

    INPUT_VOCAB_SIZE = stoi(argv[5]);
    OUTPUT_VOCAB_SIZE = stoi(argv[6]);
  }

  
  Model model;
  EncoderDecoderForward<LSTMBuilder> lm(model);

  cerr << "Reading model from " << argv[1] << "...\n";
  string fname = argv[1];
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  ia >> sd;
  ia >> td;
  sd.Freeze(); //no new word types allowed
  td.Freeze();
  INPUT_VOCAB_SIZE = sd.size(); //set vocab sizes
  OUTPUT_VOCAB_SIZE = td.size();
 
  lm.GetCharEmbeddings();

  cerr << "Done." << endl;

} 
