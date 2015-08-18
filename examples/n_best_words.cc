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
unsigned N_BEST = 0;

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
  
  vector<tuple<vector<int>, float>> GetNBestSeqs (const vector<int>& insent, ComputationGraph& cg) {
    // forward encoder
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    for (unsigned t = 0; t < insent.size(); ++t) {
     // if (IN_DEV)
     //     cerr << " | " << t ;
      Expression i_x_t = lookup(cg,p_ec,insent[t]);
      fwd_enc_builder.add_input(i_x_t);
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int t = insent.size() - 1; t >= 0; --t) {
      Expression i_x_t = lookup(cg, p_ec, insent[t]);
      rev_enc_builder.add_input(i_x_t);
    }

    // encoder -> decoder transformation
    vector<Expression> to;
    for (auto h_l : fwd_enc_builder.final_h()) to.push_back(h_l);
    for (auto h_l : rev_enc_builder.final_h()) to.push_back(h_l);

    Expression i_combined = concatenate(to);
    Expression i_ie2h = parameter(cg, p_ie2h);
    Expression i_bie = parameter(cg, p_bie);
    Expression i_t = i_bie + i_ie2h * i_combined;
    cg.incremental_forward();
    Expression i_h = rectify(i_t);
    Expression i_h2oe = parameter(cg,p_h2oe);
    Expression i_boe = parameter(cg,p_boe);
    Expression i_nc = i_boe + i_h2oe * i_h;

    vector<Expression> oein1, oein2, oein;
    for (int i = 0; i < LAYERS; ++i) {
      oein1.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
      oein2.push_back(tanh(oein1[i]));
    }
    for (int i = 0; i < LAYERS; ++i) oein.push_back(oein1[i]);
    for (int i = 0; i < LAYERS; ++i) oein.push_back(oein2[i]);

    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(oein);

    // decoder
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);


    //Insert <s> into dec_builder
    Expression start_x = lookup(cg, p_c, td.Convert("<s>"));
    Expression start_y = dec_builder.add_input(start_x);
    Expression start_r = i_bias + i_R * start_y;
    Expression start_i_distr = log_softmax(start_r);

    vector<float> start_ydist = as_vector(cg.incremental_forward());

    //Start beam search for n best sequences
    vector<tuple<vector<int>, float>> finished_hypoths;
    vector<hypoth_type> prev_hypoths;
    vector<hypoth_type> next_hypoths;
    hypoth_type init_hypoth (vector<int>({td.Convert("<s>")}), 1, dec_builder.state(), start_ydist);
    prev_hypoths.push_back(init_hypoth);

    while (finished_hypoths.size() < N_BEST && prev_hypoths.size() > 0 ) {

      //DEBUG 
      //cerr <<   "*" << endl;
      for (int i = 0; i < prev_hypoths.size(); i++) {
        string str = seq2str(get<0>(prev_hypoths[i]), td, false);
        //cerr <<   "\t- " << str << "\n";
      }

      for (int i = 0; i < prev_hypoths.size(); i++) {
        //break up tuple
        hypoth_type curr_hypoth = prev_hypoths[i];
        vector<int> curr_seq;
        float curr_score;
        NetPointer curr_state;
        vector<float> curr_distr;
        tie(curr_seq, curr_score, curr_state, curr_distr) = curr_hypoth;

        //expand current hypothesis in all possible ways
        for (int j = 0; j < td.size(); j++) {
          Expression new_i_x = lookup(cg, p_c, j);
          Expression new_i_y = dec_builder.add_input(curr_state, new_i_x);
          Expression new_i_r = i_bias + i_R * new_i_y;

          vector<int> new_seq(curr_seq);
          new_seq.push_back(j);
          float new_score = curr_score + curr_distr[j];
          NetPointer new_state = dec_builder.state();
          Expression new_i_ydist = log_softmax(new_i_r);

          vector<float> ydist = as_vector(cg.incremental_forward());
  
          hypoth_type new_hypoth(new_seq, new_score, new_state, ydist);
          next_hypoths.push_back(new_hypoth);
          
        }
        //cerr << "next hypoth size " << next_hypoths.size() << endl ;
      }
      //sort fronter 
      sort(next_hypoths.begin(), next_hypoths.end(), hypoth_comp);

      //clear to-do list
      prev_hypoths.clear();
      //cerr <<   "finished hypoth size " << finished_hypoths.size() << endl ;
      //cerr <<   "prev hypoth size " << prev_hypoths.size() << endl ;
      //cerr <<   "next hypoth size " << next_hypoths.size() << endl ;
      //separate out finished ones, select N_best next ones to do
      for (uint i = 0; prev_hypoths.size() < N_BEST; i++) {
        if ((get<0>(next_hypoths[i])).back() == td.Convert("</s>")){
          tuple<vector<int>, float> new_finished(get<0>(next_hypoths[i]), get<1>(next_hypoths[i]));
          finished_hypoths.push_back(new_finished);
          string str = seq2str(get<0>(next_hypoths[i]), td, false);
          //cerr <<   "\t!! " << str << "\n";
          //cerr <<   "found finished " << i << endl;
        } else {
          prev_hypoths.push_back(next_hypoths[i]);
          //cerr <<   "n_best picked " << i << endl;
        }
        //cerr <<   "~";
      }

      //clear next_hypoths
      next_hypoths.clear();
        
    }
    sort(finished_hypoths.begin(), finished_hypoths.end(), fin_hypoth_comp);
    finished_hypoths.resize(N_BEST);

    return finished_hypoths;
  }
};


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  
  if (argc != 9) {
    cerr << "Usage: " << argv[0] << " test.txt model.params LAYERS INPUT_DIM HIDDEN_DIM NBEST_N INPUT_VOCAB_SIZE OUTPUT_VOCAB SIZE\n";
    return 1;
  }

  if (argc == 9) {
    LAYERS = stoi(argv[3]);
    INPUT_DIM = stoi(argv[4]);
    HIDDEN_DIM = stoi(argv[5]);
    N_BEST = stoi(argv[6]);

    INPUT_VOCAB_SIZE = stoi(argv[7]);
    OUTPUT_VOCAB_SIZE = stoi(argv[8]);
  }

  
  Model model;
  EncoderDecoderForward<LSTMBuilder> lm(model);

  cerr << "Reading model from " << argv[2] << "...\n";
  string fname = argv[2];
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  ia >> sd;
  ia >> td;
  sd.Freeze(); //no new word types allowed
  td.Freeze();
  INPUT_VOCAB_SIZE = sd.size(); //set vocab sizes
  OUTPUT_VOCAB_SIZE = td.size();
 
  vector<vector<int>> test;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> test_sent;
      test.push_back(ReadSentence(line, &sd));
      ttoks += test.back().size();
      if (test.back().front() != sd.Convert("<s>") && test.back().back() != sd.Convert("</s>")) {
        cerr << "Sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens " << endl;
 
  }

  for (uint i = 0; i < test.size(); i++) {
    
      // build graph for this instance
      ComputationGraph cg;
      vector<tuple<vector<int>, float>> nbest_list;
      auto& isent = test[i];
      if (i % 50 == 0) {
          cerr << "##"  << seq2str(isent, sd, true) << "\n" ;
          cerr << "Percent done: " << ((float)i)/((float)test.size()) << endl;
      }
      nbest_list = lm.GetNBestSeqs(isent, cg);
      string src = seq2str(isent, sd, true);

      for (uint j = 0; j < nbest_list.size(); j++) {
        string tgt = seq2str(get<0>(nbest_list[j]), td, true);
        float score = get<1>(nbest_list[j]);
        cout << src << " ||| " << tgt << " ||| " << score << endl;
      }
  }

  cerr << "Done." << endl;

} 
