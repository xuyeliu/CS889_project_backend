import argparse
import os
import pickle
import random
import sys
import collections
import numpy as np
import tensorflow as tf
from keras import Model
from utils.myutils import batch_gen, init_tf, seq2sent
import keras
import keras.backend as K
from utils.model import create_model
from timeit import default_timer as timer
from models.custom.graphlayer import GCNLayer
import pickle
import json
import heapq


class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def gen_pred(fid_set, model, data, comstok, smltok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...
    tdats, coms, wsmlnodes, wedge_1 = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)
    final_dict = collections.defaultdict(dict)
    dict1 = {}
    dict1["code sequence_token"] = tdats
    dict1["code sequence"] = seq2sent(tdats[0], smltok)
    dict1["graph node_token"] = wsmlnodes
    dict1["graph node"] = seq2sent(wsmlnodes[0], smltok)
    dict1["graph edge"] = wedge_1
    dict1["topK"] = {}
    dict1['topK_confidence_score'] = {}
    dict1["topK_word"] = {}
    outfn = "ICPC2020_GNN/modelout/predictions/layer_output.txt"
    outf = open(outfn, 'w')
    output_dict = collections.defaultdict(dict)
    for i in range(1, comlen):
        output_dict[i] = {}
        results = model.predict([tdats, coms, wsmlnodes, wedge_1],
                                batch_size=batchsize)
        att1 = Model(inputs=model.input, outputs=model.get_layer('dot_1').output)
        att2 = Model(inputs=model.input, outputs=model.get_layer('dot_3').output)
        emd1 = Model(inputs=model.input, outputs=model.get_layer('embedding_1').get_output_at(0))
        emd2 = Model(inputs=model.input, outputs=model.get_layer('embedding_1').get_output_at(1))
        att1_output = att1.predict([tdats, coms, wsmlnodes, wedge_1])
        att2_output = att2.predict([tdats, coms, wsmlnodes, wedge_1])
        emd1_output = emd1.predict([tdats, coms, wsmlnodes, wedge_1])
        emd2_output = emd2.predict([tdats, coms, wsmlnodes, wedge_1])
        outf.write("iteration{}\n".format(i))
        outf.write("{}\t{}\t{}\n".format("Attention weight between code sequence and predicted document", att1_output.shape, att1_output))
        outf.write("{}\t{}\t{}\n".format("Attention weight between graph node sequence and predicted document", att2_output.shape, att2_output))
        outf.write("{}\t{}\t{}\n".format("Embedding for code sequence", emd1_output.shape, emd1_output))
        outf.write("{}\t{}\t{}\n".format("Embedding for code graph node", emd2_output.shape, emd2_output))
        # print(att1_output)
        # print(att2_output)
        # print(emd1_output)
        output_dict[i]["Attention weight between code sequence and predicted document"] = att1_output
        output_dict[i]["Attention weight between graph node sequence and predicted document"] = att2_output
        output_dict[i]["Embedding for code sequence"] = emd1_output
        output_dict[i]["Embedding for code graph node"] = emd2_output
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
            dict1['topK_confidence_score'][i] = list(heapq.nlargest(10, [np.float(tmp) for tmp in s]))
            dict1["topK"][i] = s.argsort()[-10:][::-1]
            dict1["topK_word"][i] = seq2sent(dict1["topK"][i], comstok)
            print("com_i", coms[c][i])
    with open('ICPC2020_GNN/modelout/predictions/layer_output.json', 'w') as fp:
        # pickle.dump(output_dict, fp)
        fp.write(json.dumps(output_dict, cls=NDArrayEncoder))
    with open('ICPC2020_GNN/modelout/predictions/input.json', 'w') as fp:
        # pickle.dump(dict1, fp)
        fp.write(json.dumps(dict1, cls=NDArrayEncoder))
    outf.close() 
    # with open('ICPC2020_GNN/modelout/predictions/layer_output.json', 'rb') as fp:
    #     data = pickle.load(fp)
    #     print(data[2])

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data, output_dict, dict1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', type=str, default=None)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default=None)
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='../data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='ICPC2020_GNN/modelout/')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=1) 
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)

    args = parser.parse_args()

    modelfile = args.model
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile

    config = dict()

    # User set parameters#
    config['maxastnodes'] = 100
    config['asthops'] = 10

    if modeltype == None:
        modeltype = modelfile.split('_')[0].split('/')[-1]


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
   
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))

    allfids = list(seqdata['ctest'].keys())
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    config['tdatvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    # set sequence lengths
    config['tdatlen'] = 50
    config['comlen'] = len(list(seqdata['ctrain'].values())[0])
    config['smllen'] = len(list(seqdata['strain_nodes'].values())[0])
    
    config['batch_size'] = batchsize

    comlen = len(seqdata['ctest'][list(seqdata['ctest'].keys())[0]])

    config, _ = create_model(modeltype, config)
    print("MODEL LOADED")
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras,'AlexGraphLayer':GCNLayer})

    node_data = seqdata['stest_nodes']
    edgedata = seqdata['stest_edges']
    config['batch_maker'] = 'graph_multi_1'

    print(model.summary())

    # set up prediction string and output file
    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict-{}.txt".format(modeltype)
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
    index = 0
    final_dict = collections.defaultdict(dict)
    for c, fid_set in enumerate(batch_sets):
        index += 1
        if index == 20:
          break
        st = timer()
        for fid in fid_set:
            seqdata['ctest'][fid] = comstart #np.asarray([stk])
        print("fid", fid_set)
        print("c", c)
        bg = batch_gen(seqdata, 'test', config, nodedata=node_data, edgedata=edgedata)
        batch = bg.make_batch(fid_set)
   
        batch_results, output_dict, dict1 = gen_pred(fid_set, model, batch, comstok, smltok, comlen, batchsize, config, strat='greedy')
        final_dict[fid_set[0]]["layer_output"] = output_dict
        final_dict[fid_set[0]]["input"] = dict1
        for key, val in batch_results.items():
            print("summary", val)
            outf.write("{}\t{}\n".format(key, val))
        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, int(batchsize/(end-st))), end='\r')
        # break
    with open('ICPC2020_GNN/modelout/predictions/final_dict.json', 'w') as fp:
        # pickle.dump(dict1, fp)
        fp.write(json.dumps(final_dict, cls=NDArrayEncoder))
    outf.close()        

