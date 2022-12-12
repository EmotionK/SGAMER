import os,argparse

def parse_args(dataset_name):

    parser = argparse.ArgumentParser(description="metapath config file gen")

    parser.add_argument('--dn', type=str, default=dataset_name, help='The name of dataset')
    parser.add_argument('--output_dir', type=str, default=f'../data/{dataset_name}', help='The output dir of generated files')
    parser.add_argument('--relation_dir', type=str, default=f'../data/{dataset_name}', help='The dir of input relation files')

    return parser.parse_args()

def find_edges(current_edge, current_metapath_list, metapath_list,current_metapath_str, metapath_str):#ap,[ap],[],ap,[]
    for third_node in edge_dict[current_edge]:#{'ap': ['p', 'a', 't'], 'pp': ['p', 'a', 't'], 'pa': ['p'], 'pt': ['p'], 'tp': ['p', 'a', 't']}
        new_edge = current_edge[1:] + third_node#pp,pa,pt
        if new_edge in current_metapath_list:
            repeat_idx = current_metapath_list.index(new_edge)
            current_metapath_list_ = current_metapath_list[repeat_idx:]
            current_metapath_str_ = current_metapath_str[repeat_idx:]
            metapath_list.append(list(current_metapath_list_))
            metapath_str.append(current_metapath_str_)
        elif new_edge not in current_metapath_list:
            current_metapath_list.append(new_edge)
            current_metapath_str += third_node
            current_metapath_list, current_metapath_str, metapath_list, metapath_str = find_edges(new_edge,#pp
                                                                                                  current_metapath_list,#[ap,pp]
                                                                                                  metapath_list,#[]
                                                                                                  current_metapath_str,#app
                                                                                                  metapath_str)#[]
    del current_metapath_list[-1]
    current_metapath_str = current_metapath_str[:-1]
    return current_metapath_list, current_metapath_str, metapath_list, metapath_str


def metapath_gen(metapath_list):#[('ppp',), ('app', 'pap', 'ppa'), ('ptp', 'tpt'), ('ppt', 'ptp', 'tpp'), ('apt', 'pap', 'ptp', 'tpa'), ('apa', 'pap')]
    metapath_list_out = []
    for metapath in metapath_list:
        metapath_out = []
        for edge in metapath:
            edge_out = ':'.join(list(edge))
            metapath_out.append('\''+edge_out+'\'')
        metapath_list_out.append('['+','.join(metapath_out)+']')
    str_out = '['+','.join(metapath_list_out)+']'#[['p:p:p'],['a:p:p','p:a:p','p:p:a'],['p:t:p','t:p:t'],['p:p:t','p:t:p','t:p:p'],['a:p:t','p:a:p','p:t:p','t:p:a'],['a:p:a','p:a:p']]
    return str_out


def three_elem_edge_gen(metapath_str):#['pp', 'appa', 'pptp', 'apptpa', 'ptp', 'apa', 'pp', 'aptppa', 'ptpp', 'aptpa', 'ptp', 'pp', 'ppap', 'pap', 'ppaptp', 'paptp', 'ptp', 'pptp', 'pptpap', 'pap', 'ptpap', 'ptp', 'pp', 'papp', 'pptp', 'papptp', 'ptp', 'pap', 'pp', 'paptpp', 'ptpp', 'paptp', 'ptp', 'pp', 'ppap', 'pap', 'ptppap', 'ptpp', 'pp', 'papp', 'ptpapp', 'pap', 'ptpap', 'ptp', 'pp', 'ppap', 'pap', 'tppapt', 'tppt', 'pp', 'papp', 'tpappt', 'pap', 'tpapt', 'tpt']
    metapath_list = []
    key_edge_list = []
    for metapath_str_i in metapath_str:
        edge_list = edge_to_edge_list(metapath_str_i)
        metapath_list.append(edge_list)
        key_edge_list.append(edge_list[-1][:2])
    return metapath_list, key_edge_list


def node_list_gen(metapath_list):#[('ppp',), ('app', 'pap', 'ppa'), ('ptp', 'tpt'), ('apa', 'pap'), ('ppt', 'ptp', 'tpp'), ('apt', 'pap', 'ptp', 'tpa')]
    node_list = []
    node_to_metapath_indicator = []
    for node_type in node_connection_dict:#{'a': ['p'], 'p': ['p', 'a', 't'], 't': ['p']}
        node_list.append(node_type)

    for node_type in node_list:#[a,p,t]
        node_to_metapath_indicator_i = []
        for metapath_id,metapath in enumerate(metapath_list):
            if_exist = -1
            for edge in metapath:
                if node_type in edge:
                    if_exist = 1
            if if_exist !=-1:
                node_to_metapath_indicator_i.append(str(metapath_id))
        node_to_metapath_indicator.append(node_to_metapath_indicator_i)
    #print(node_to_metapath_indicator) [['1', '3', '5'], ['0', '1', '2', '3', '4', '5'], ['2', '4', '5']]
    node_list = ['\''+node+'\'' for node in node_list]#["'a'", "'p'", "'t'"]
    str_out_node = '[' + ','.join(node_list) + ']'#['a','p','t']

    metapath_list_out = []
    for metapath in node_to_metapath_indicator:#[['1', '3', '5'], ['0', '1', '2', '3', '4', '5'], ['2', '4', '5']]
        edge_out = ','.join(metapath)
        metapath_list_out.append('[' + edge_out + ']')
    str_out_indicator = '[' + ','.join(metapath_list_out) + ']'#[[0,3,5],[0,1,2,3,4,5],[0,2,4]]

    return str_out_node, str_out_indicator


def edge_list_gen(metapath_list):#[('app', 'pap', 'ppa'), ('ppt', 'ptp', 'tpp'), ('ptp', 'tpt'), ('ppp',), ('apt', 'pap', 'ptp', 'tpa'), ('apa', 'pap')]
    edge_list = set()
    for metapath in metapath_list:
        for edge in metapath:
            edge_list.add('\''+':'.join(list(edge))+'\'')
    edge_list = list(edge_list)#["'a:p:t'", "'t:p:p'", "'a:p:p'", "'a:p:a'", "'p:t:p'", "'t:p:t'", "'p:p:p'", "'t:p:a'", "'p:a:p'", "'p:p:t'", "'p:p:a'"]
    str_out_edge = '[' + ','.join(edge_list) + ']' #['a:p:t','t:p:p','a:p:p','a:p:a','p:t:p','t:p:t','p:p:p','t:p:a','p:a:p','p:p:t','p:p:a']
    return str_out_edge

def edge_to_edge_list(node_list):
    node_list = node_list + node_list[1]
    edge_list = []
    for idx in range(len(node_list)-2):
        edge_list.append(node_list[idx:idx+3])
    return edge_list

def symmetric_edge_select(metapath_list):
    new_metapath_list = []
    for metapath in metapath_list:#[['ppp'], ['app', 'ppa', 'pap'], ['ppt', 'ptp', 'tpp'], ['app', 'ppt', 'ptp', 'tpa', 'pap'],
        metapath_string = ''
        for node_index, node in enumerate(metapath):
            if node_index == 0:
                metapath_string = node
            else:
                metapath_string += node[-1]
        metapath_string = metapath_string[:-1]#appa
        metapath_string_reverse = metapath_string[::-1]
        if metapath_string == metapath_string_reverse:
            new_metapath_list.append(metapath)
    return new_metapath_list


#if __name__ == '__main__':
def gen_metapath(dataset_name):
    
    global node_connection_dict
    global edge_dict

    args = parse_args(dataset_name)
    input_file_path = os.path.join(args.relation_dir,args.dn+'.relation') #relation_files\cora.relation

    node_connection_dict = dict() #{'a': ['p'], 'p': ['p', 'a', 't'], 't': ['p']}
    ## build connection dict
    with open(input_file_path,'r') as f:
        for line in f.readlines():
            node_a,node_b = line.rstrip().split('-')
            if not node_a in node_connection_dict:
                node_connection_dict[node_a] = []
            node_connection_dict[node_a].append(node_b)


    ## build edge dict
    edge_dict = dict()#{'ap': ['p', 'a', 't'], 'pp': ['p', 'a', 't'], 'pa': ['p'], 'pt': ['p'], 'tp': ['p', 'a', 't']}
    for node_a in node_connection_dict:#{'a': ['p'], 'p': ['p', 'a', 't'], 't': ['p']}
        for node_b in node_connection_dict[node_a]:
            for node_c in node_connection_dict[node_b]:
                if not node_a+node_b in edge_dict:
                    edge_dict[node_a+node_b] = []
                edge_dict[node_a+node_b].append(node_c)



    metapath_list_all = [] #选择的元路径列表 [['pp'], ['ap', 'pp', 'pa'], ['pp', 'pt', 'tp']...]
    metapath_str_all = [] #选择的元路径str ['pp', 'appa', 'pptp',.....]
    for node_a in node_connection_dict:#{'a': ['p'], 'p': ['p', 'a', 't'], 't': ['p']}
        for node_b in node_connection_dict[node_a]:
            current_edge = node_a + node_b
            current_metapath_list, current_metapath_str, metapath_list, metapath_str = find_edges(current_edge,
                                                                                                  [current_edge],
                                                                                                  [],
                                                                                                  current_edge,
                                                                                                  [])

            metapath_list_all.extend(list(metapath_list))
            metapath_str_all.extend(list(metapath_str))



    metapath_list_all, key_edge_list = three_elem_edge_gen(metapath_str_all)
    #print(metapath_list_all) [['ppp'], ['app', 'ppa', 'pap'], ['ppt', 'ptp', 'tpp'],
    #print(key_edge_list) ['pp', 'pa', 'tp',
    metapath_list_all = symmetric_edge_select(metapath_list_all)#[['ppp'], ['app', 'ppa', 'pap'], ['ptp', 'tpt'], ['apa', 'pap'], ['ppp'],
    metapath_list = [tuple(sorted(list(set(metapath)))) for metapath in metapath_list_all]#[('ppp',), ('app', 'pap', 'ppa'), ('ptp', 'tpt'), ('apa', 'pap'),
    metapath_list_all = list(set(metapath_list))
    node_line,node_to_metapath_line = node_list_gen(metapath_list_all)#['a','p','t'],[[0,3,5],[0,1,2,3,4,5],[0,2,4]]
    edge_line = edge_list_gen(metapath_list_all) #['a:p:t','t:p:p','a:p:p','a:p:a','p:t:p','t:p:t','p:p:p','t:p:a','p:a:p','p:p:t','p:p:a']
    metapath_line = metapath_gen(metapath_list_all)#[['p:p:p'],['a:p:p','p:a:p','p:p:a'],['p:t:p','t:p:t'],['p:p:t','p:t:p','t:p:p'],['a:p:t','p:a:p','p:t:p','t:p:a'],['a:p:a','p:a:p']]

    with open(os.path.join(args.output_dir,args.dn+'.config'),'w') as f:
        f.write(node_line+'\n')
        f.write(edge_line+'\n')
        f.write(metapath_line+'\n')
        f.write(node_to_metapath_line+'\n')

