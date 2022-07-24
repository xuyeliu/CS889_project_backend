import argparse
from inspect import modulesbyfile
import os
import pickle
import random
import sys
import collections
import numpy as np
import tensorflow as tf
from keras import Model
from bs4 import BeautifulSoup
from html.parser import HTMLParser
from utils.myutils import batch_gen, init_tf, seq2sent
import keras
import keras.backend as K
from utils.model import create_model
from timeit import default_timer as timer
from models.custom.graphlayer import GCNLayer
import pickle
import json
import networkx as nx
import re
import statistics
import numpy as np
import heapq

class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

def load(filename):
    return pickle.load(open(filename, 'rb'))

def save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def load_good_fid():
    filename = './output/dataset.coms'
    good_fid = []
    for line in open(filename):
        tmp = [x.strip() for x in line.split(',')]
        fid = int(tmp[0])
        good_fid.append(fid)

    return good_fid

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

def gen_pred(model, newxml, data, comstok, smltok, comlen, strat='greedy'):
    # right now, only greedy search is supported...
    tdats, coms, wsmlnodes, wedge_1 = data
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
    outfn = "CS889_project_backend/modelout/predictions/layer_output.txt"
    outf = open(outfn, 'w')
    output_dict = collections.defaultdict(dict)
    for i in range(1, comlen):
        output_dict[i] = {}
        results = model.predict([tdats, coms, wsmlnodes, wedge_1])
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
            dict1['topK_confidence_score'][i] = list(heapq.nlargest(10, list([np.float(tmp) for tmp in s])))
            dict1["topK"][i] = s.argsort()[-10:][::-1]
            dict1["topK_word"][i] = seq2sent(dict1["topK"][i], comstok)
            dict1['newxml'] = newxml
            print("com_i", coms[c][i])
    with open('CS889_project_backend/modelout/predictions/layer_output.json', 'w') as fp:
        # pickle.dump(output_dict, fp)
        fp.write(json.dumps(output_dict, cls=NDArrayEncoder))
    with open('CS889_project_backend/modelout/predictions/input.json', 'w') as fp:
        # pickle.dump(dict1, fp)
        fp.write(json.dumps(dict1, cls=NDArrayEncoder))
    outf.close() 
    # with open('ICPC2020_GNN/modelout/predictions/layer_output.json', 'rb') as fp:
    #     data = pickle.load(fp)
    #     print(data[2])

    final_data = seq2sent(coms[0], comstok)

    return final_data, output_dict, dict1

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super(MyHTMLParser, self).__init__()
        self.parentstack = list()
        self.curtag = -1
        self.tagidx = -1
        self.graph = nx.Graph()
        self.seq = list()
        
    def handle_starttag(self, tag, attrs):
        self.parentstack.append(self.curtag)
        self.tagidx += 1
        self.seq.append(tag)
        self.graph.add_node(self.tagidx, text=tag)
        if self.parentstack[-1] >= 0:
            self.graph.add_edge(self.parentstack[-1], self.tagidx)
        self.curtag = self.tagidx
        
    def handle_endtag(self, tag):
        self.curtag = self.parentstack.pop()
        
    def handle_data(self, data):
        
        # first, do dats text preprocessing
        data = re_0001_.sub(re_0002, data).lower().rstrip()
        
        # second, create a node if there is text
        if(data != ''):
            for d in data.split(' '): # each word gets its own node
                if d != '':
                    self.parentstack.append(self.curtag)
                    self.tagidx += 1
                    self.seq.append(d)
                    self.graph.add_node(self.tagidx, text=d)
                    self.graph.add_edge(self.parentstack[-1], self.tagidx)
                    self.curtag = self.tagidx
                    self.curtag = self.parentstack.pop()
        
    def get_graph(self):
        return(self.graph)

    def get_seq(self):
        return(self.seq)

import xml.etree.ElementTree as ET

class MyHTMLParserAlt():
    def __init__(self):
        self.parentstack = list()
        self.curtag = -1
        self.tagidx = -1
        self.graph = nx.Graph()
        self.seq = list()
        self.tokens = list()
        self.document = None

    def get_document(self):
        return self.document

    def feed(self, text):
        self.document = ET.fromstring(text.replace('xmlns="http://www.srcML.org/srcML/src"', ""))

        self._process_element(self.document)

    def _process_element(self, element):
        tag = element.tag.replace("{http://www.srcML.org/srcML/src}", "")

        self.handle_starttag(tag, element)
        if element.text is not None and element.text.strip() != "":
            self.handle_data(element.text, element)
        for child in element:
            self._process_element(child)
        self.handle_endtag(tag, element)


    def handle_starttag(self, tag, element):
        self.parentstack.append(self.curtag)
        self.tagidx += 1
        self.seq.append(tag)
        element.attrib["node-id"] = str(len(self.seq) - 1)
        self.graph.add_node(self.tagidx, text=tag)
        if self.parentstack[-1] >= 0:
            self.graph.add_edge(self.parentstack[-1], self.tagidx)
        self.curtag = self.tagidx

    def handle_endtag(self, tag, element):
        self.curtag = self.parentstack.pop()

    def handle_data(self, data, element):
        # first, do dats text preprocessing
        data = re_0001_.sub(re_0002, data).lower().rstrip()

        # second, create a node if there is text
        if (data != ''):
            for d in data.split(' '):  # each word gets its own node
                if d != '':
                    self.parentstack.append(self.curtag)
                    self.tagidx += 1

                    self.seq.append(d)
                    self.tokens.append(d)

                    if "node-id" not in element.attrib.keys():
                        element.attrib["node-id"] = str(len(self.seq) - 1)
                    else:
                        element.attrib["node-id"] += " " + str(len(self.seq) - 1)

                    if "seq-id" not in element.attrib.keys():
                        element.attrib["seq-id"] = str(len(self.tokens) - 1)
                    else:
                        element.attrib["seq-id"] += " " + str(len(self.tokens) - 1)

                    if "tokens" not in element.attrib.keys():
                        element.attrib["tokens"] = d
                    else:
                        element.attrib["tokens"] += " " + d

                    self.graph.add_node(self.tagidx, text=d)
                    self.graph.add_edge(self.parentstack[-1], self.tagidx)
                    self.curtag = self.tagidx
                    self.curtag = self.parentstack.pop()

    def get_graph(self):
        return (self.graph)

    def get_seq(self):
        return (self.seq)

    def get_tokens(self):
        return (self.tokens)

def xmldecode(unit):
    parser = MyHTMLParser()
    parser.feed(unit)
    return(parser.get_graph(), parser.get_seq())

def new_xml_Generate(unit):

  parserAlt = MyHTMLParserAlt()
  parserAlt.feed(unit)
  print(parserAlt.get_seq())
  print(parserAlt.get_graph())
  print(ET.tostring(parserAlt.get_document(), encoding='unicode'))

  # print(list(map(lambda x: x.lower(), re_0001_.sub(re_0002, unit).split())))
  print(parserAlt.get_tokens())
  return ET.tostring(parserAlt.get_document(), encoding='unicode')

def w2i(word):
    try:
        i = smlstok.w2i[word]
    except KeyError:
        i = smlstok.oov_index
    return i

def load_model():
    modelfile = "CS889_project_backend/final_data/codegnngru.h5"
    tdatstok = pickle.load(open('CS889_project_backend/final_data/tdats.tok', 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('CS889_project_backend/final_data/coms.tok', 'rb'), encoding='UTF-8')
    smlstok = pickle.load(open('CS889_project_backend/final_data/smls.tok', 'rb'), encoding='UTF-8')
   
    # seqdata = pickle.load(open('ICPC2020_GNN/final_data/dataset.pkl', 'rb'))
    comlen = 13
    config = dict()

    # User set parameters#
    config['maxastnodes'] = 100
    config['asthops'] = 10

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smlstok.vocab_size

    config['tdatvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    # set sequence lengths
    config['tdatlen'] = 50
    config['comlen'] = 13
    config['smllen'] = 100
    
    config['batch_size'] = 30
    modeltype = "codegnngru"

    config, _ = create_model(modeltype, config)
    print("MODEL LOADED")
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras,'AlexGraphLayer':GCNLayer})

    return config, model, smlstok, tdatstok, comstok
    
def interface(dats, config, model, smlstok, tdatstok, comstok):
    xml = []
    with open('CS889_project_backend/min_example.java', 'w') as file:
        file.write(dats)
    ## generate code sequence data
    newdats = dict()
    #for line in open(comfile):
    newdats = re_0001_.sub(re_0002, dats)
    tmp = newdats.split()
    if len(tmp) > 100:
        sys.exit()

    textdat = ' '.join(tmp)
    textdat = textdat.lower()
    os.system("srcml CS889_project_backend/min_example.java -o CS889_project_backend/min_example.xml")
    lens = list()
    blanks = 0
    unit = ''
    try:
      with open('CS889_project_backend/min_example.xml', 'r') as file:
          unit = file.read()
    except:
      print("error reading")
      pass
    # unit = xml_from_code
    print(unit)
    # unit = unit.decode('utf-8', 'ignore')
    # unit = pickle.load(open('min_example.xml', 'rb'))
    (graph, seq) = xmldecode(unit)
    seq = ' '.join(seq)

    (graph, seq) = xmldecode(unit)
    newxml = new_xml_Generate(unit)
    print(seq)
    lens.append(len(graph.nodes.data()))
    nodes = list(graph.nodes.data())
    print(nodes)
    nodes = np.asarray([w2i(x[1]['text']) for x in list(graph.nodes.data())])
    edges = nx.adjacency_matrix(graph)
    # print('blanks:', blanks)
    # print('avg:', sum(lens) / len(lens))
    # print('max:', max(lens))
    # print('median:', statistics.median(lens))
    # print('% abv 200:', sum(i > 200 for i in lens) / len(lens))
    # print("nodes", nodes)
    # print("edges",edges)
    comlen = 13 
    tdatlen = 50
    smllen = 100
    print(tmp)
    tdats = smlstok.text_to_seq(textdat)
    # tdats = np.asarray([w2i(x) for x in textdat.split()])
    print("original tdats", tdats)
    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    comment = list()
    comment.append(comstart)
    tdatseqs = list()
    comseqs = list()
    smlnodes = list()

    wedge_1 = list()

    comouts = list()

    fiddat = dict()
    wsmlnodes = nodes[:config['maxastnodes']]
    tmp = np.zeros(config['maxastnodes'], dtype='int32')
    tmp[:wsmlnodes.shape[0]] = wsmlnodes
    wsmlnodes = np.int32(tmp)

    # crop/expand ast adjacency matrix to dense
    edge_1 = np.asarray(edges.todense())
    edge_1 = edge_1[:config['maxastnodes'], :config['maxastnodes']]
    tmp_1 = np.zeros((config['maxastnodes'], config['maxastnodes']), dtype='int32')
    tmp_1[:edge_1.shape[0], :edge_1.shape[1]] = edge_1
    edge_1 = np.int32(tmp_1)
    tdatseqs = tdats[:config['tdatlen']]
    wedge_1.append(edge_1)
    # tdatseqs.append(wtdatseq)
    smlnodes.append(wsmlnodes)
    tdatseqs = np.asarray(tdatseqs)
    smlnodes = np.asarray(smlnodes)
    wedge_1 = np.asarray(wedge_1)
    comment = np.asarray(comment)
    batch = [tdatseqs, comment, smlnodes, wedge_1]
    batch_results, output_dict, dict1 = gen_pred(model, newxml, batch, comstok, smlstok, comlen, strat='greedy')
    return batch_results, output_dict, dict1

    ## generate AST node and edges
if __name__ == "__main__":
    final_dict = collections.defaultdict(dict)
    config, model, smlstok, tdatstok, comstok = load_model()
    with open('CS889_project_backend/xml.json','r') as fr: #默认为 encoding='utf-8‘ 注意是否需要改为 encoding='gbk'等
	    input_data = json.load(fr)
    for i in range(len(input_data["code"])):
    #   res, output_dict, dict1 = interface(input_data["code"][i], input_data["xml"][i], config, model, smlstok, tdatstok, comstok)
      res, output_dict, dict1 = interface(input_data["code"][i], config, model, smlstok, tdatstok, comstok)
      final_dict[i]["layer_output"] = output_dict
      final_dict[i]["input"] = dict1
      with open('CS889_project_backend/modelout/predictions/final_dict.json', 'w') as fp:
        # pickle.dump(dict1, fp)
        fp.write(json.dumps(final_dict, cls=NDArrayEncoder))
      # res = interface("\tpublic void setMaximumSolutionLength(short maximumSolutionLength) {\n\t\tmaximumPushesCurrentIteration = maximumSolutionLength;\n\t}\n", config, model, smlstok, tdatstok, comstok)
    # print("[1 17 3 200 20 2 0 0 0 0 0 0 0]")
    print(res)