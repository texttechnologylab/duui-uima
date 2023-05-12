import base64
import hashlib
import json
import os
#import spacy

from diaparser.parsers import Parser
from HanTa import HanoverTagger as ht


os.environ['TOKENIZERS_PARALELISM'] = 'false'

def sort_dict_v(x, reverse=True):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=reverse)}

def sort_dict_k(x, reverse=False):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[0], reverse=reverse)}

def hashify(text):
    """Calculate hash hex string."""
    hash_object = hashlib.sha1(str.encode(text))
    hash_hex = hash_object.hexdigest()
    return hash_hex

def load_parser(model_name='de_hdt.dbmdz-bert-base'):
    #pytorch hangs prediction ThilinaRajapakse/simpletransformers
    args = {'use_multiprocessing': False, 'use:multiprocessing_for_evaluation': False,
            'process_count': 1, 'verbose': True}
    #print('USING', f'{"GPU" if torch.cuda.is_available() else "CPU"}')
    return Parser.load(model_name, args=args)

def load_tagger():
    return ht.HanoverTagger('morphmodel_ger.pgz')

def get_typesystem():
    path = Path('./')
    with open(path  / 'TypeSystemPerspective.xml', 'rt', encoding='UTF-8') as fp:
        file_content = fp.read()
        typesystem = load_typesystem(file_content)

    return typesystem

def get_head_arc(arcs, token_i):
    head_arcs_ = [a for a in arcs if (a['start']==token_i and a['dir']=='left')
        or (a['end']==token_i and a['dir']=='right')]

    head_arcs = {}
    for arc in head_arcs_:
        if arc['start']==token_i and arc['dir']=='left':
            head_arcs[arc['end']] = arc['label']
        elif arc['end']==token_i and arc['dir']=='right':
            head_arcs[arc['start']] = arc['label']
            
    assert len(head_arcs) == 1
    return list(head_arcs.items())

def get_children(arcs, token_i):
    pred_arcs_ = [a for a in arcs if (a['start']==token_i and a['dir']=='right')
            or (a['end']==token_i and a['dir']=='left')]
    
    pred_arcs = {}
    for arc in pred_arcs_:
        if arc['start']==token_i and arc['dir']=='right':
            pred_arcs[arc['end']] = arc['label']
        elif arc['end']==token_i and arc['dir']=='left':
            pred_arcs[arc['start']] = arc['label']
            
    return pred_arcs

def get_dependents(arcs, token_i):
    pred_arcs_ = [a for a in arcs if (a['start']==token_i and a['dir']=='right')
            or (a['end']==token_i and a['dir']=='left')]
    
    pred_arcs = {}
    for arc in pred_arcs_:
        if arc['label'] == 'acl':
            continue
        if arc['start']==token_i and arc['dir']=='right':
            pred_arcs[arc['end']] = arc['label']
        elif arc['end']==token_i and arc['dir']=='left':
            pred_arcs[arc['start']] = arc['label']
            
    return pred_arcs

def get_dependents_l2(arcs, token_i):
    pred_arcs_ = [a for a in arcs if (a['start']==token_i and a['dir']=='right')
            or (a['end']==token_i and a['dir']=='left')]
    
    pred_arcs = {}
    for arc in pred_arcs_:
        if arc['label'] == 'acl':
            continue
        if arc['start']==token_i and arc['dir']=='right':
            pred_arcs[arc['end']] = arc['label']
            token_ii = arc['end']
            arc_ = get_dependents(arcs, token_ii)
            pred_arcs.update(arc_)


        elif arc['end']==token_i and arc['dir']=='left':
            pred_arcs[arc['start']] = arc['label']
            token_ii = arc['start']
            arc_ = get_dependents(arcs, token_ii)
            pred_arcs.update(arc_)

    return pred_arcs
            
def get_arcs(arcs, token_i, tokens, stop_pos, stop_dep):
    pred_arcs_ = [a for a in arcs if (a['start']==token_i and a['dir']=='right')
            or (a['end']==token_i and a['dir']=='left')]
    
    pred_arcs = []
    for arc in pred_arcs_:
        if arc['label'] == 'acl' or arc['label'] == 'punct' or arc['label']=='expl':
            continue
        
        #if arc['start']==token_i and arc['dir']=='right' and tokens[arc['end']-1].pos_ != 'VERB':
        if arc['start']==token_i and arc['dir']=='right' and\
                tokens[arc['end']-1].pos_ != stop_pos and not stop_dep in arc['label']:
            #pred_arcs[arc['end']] = arc['label']
            pred_arcs.append(arc['end'])
        elif arc['end']==token_i and arc['dir']=='left' and\
                tokens[arc['start']-1].pos_ != stop_pos and not stop_dep in arc['label']:
        #elif arc['end']==token_i and arc['dir']=='left' and tokens[arc['start']-1].pos_ != 'VERB':
            #pred_arcs[arc['start']] = arc['label']
            pred_arcs.append(arc['start'])

    return pred_arcs

def get_current_arc(arcs, token_i):
    arcs = [a for a in arcs if (a['start']==token_i and a['dir']=='left')
            or (a['end']==token_i and a['dir']=='right')]

    assert len(arcs) == 1

    return arcs[0]
    
def get_dependents_recursive(arcs, token_i, tokens, t=0, stop_pos='VERB', stop_dep=' '):
    pred_arcs = get_arcs(arcs, token_i, tokens, stop_pos=stop_pos, stop_dep=stop_dep)
    current_arc = get_current_arc(arcs, token_i)
    c_arc_l = current_arc['label']
    cur_pos = tokens[token_i-1].pos_
    if len(pred_arcs) == 0 or (cur_pos == stop_pos and t!=0) or (stop_dep in c_arc_l):
        return []
    for arc_i in pred_arcs:
        pred_arcs.extend(get_dependents_recursive(arcs, arc_i, tokens, t=t+1, stop_dep=stop_dep))

    return list(set(pred_arcs))

def get_dependents_recursive_spacy(node):
    if not node.children:
        return

    #result = [[node, node.idx, node.idx + len(node.text)]]
    result = [node.i]# if node.dep_ != 'rc' else []
    for child in node.children:
        result.extend(get_dependents_recursive_spacy(child))

    return result


verbs_object = set()
with open('xcomp-object.txt', 'r') as fp:
    for line in fp:
        verbs_object.add(line.replace('\n', ''))
verbs_subject = set() 
with open('xcomp-subject.txt', 'r') as fp:
    for line in fp:
        verbs_subject.add(line.replace('\n', ''))


def srl(diaparse_sentence, tokens_, poss, tags, nlp,
        opt_dependents, opt_appos, logger):
        #opt_dependents='all', opt_appos=False, logger=None):
    if 'VERB' in tags and not 'VERB' in poss:
        tags_ = tags
        tags = poss
        poss = tags_

    for i, t in enumerate(tokens_):
        if sum([1 if x.isdigit() else 0 for x in t]) > 0:
            poss[i] = 'X'
            tags[i] = 'X'
            tokens_[i] = 'X'

    tokens = [t for t in nlp(' '.join(tokens_))]
    if len(tokens) != len(tokens_):
        print('TOKENS UNEQUAL')
        print([t.text for t in tokens])
        print(tokens_)
        print()
        return
    assert len(tokens) == len(tokens_)
    psrs = []
    verbs = [t for t, pos, tag in zip(tokens, poss, tags) if pos=='VERB' or
            (tag == 'VAFIN' and pos != 'AUX')]
    
    lemmas_spacy = {i: x.lemma_ for i, x in enumerate(tokens, start=1)}
    ners = {i: x.ent_type_ if x.ent_type else 'O' 
                for i, x in enumerate(tokens, start=1)}
            
            
    #try:
    #    hanta = tagger.tag_sent([t.text for t in tokens])
    #    lemmas_hanta = {i: x[1] for i, x in enumerate(hanta, start=1)}
    #    lemmas = lemmas_hanta

    #    assert len(tokens) == len(hanta)
    #except IndexError:
    #    lemmas = lemmas_spacy
    lemmas = lemmas_spacy
    
    
    
    
    verbs_ = [t for t in tokens if t.pos_=='VERB' or t.tag_ == 'VAFIN' and
            t.pos_ != 'AUX']
    
    verbs = verbs_
    tokens_text = {i: t for i, t in enumerate(tokens, start=1)}
    deps = {i: t.dep_ for i, t in enumerate(tokens, start=1)}
    #try:
    #    #diaparse = parser.predict([k.text for k in tokens_text.values()])
    #    diaparse = parser.predict([k.text for k in tokens_text.values()])
    #except IndexError:
    #    return
    

    #udeps = diaparse.sentences[0].rels
    #udeps_heads = diaparse.sentences[0].values[6]
    udeps = diaparse_sentence.rels
    if len(udeps) != len(tokens_text):
        print('UDEPS UNEQUAL')
        print(len(udeps), len(tokens_text))
        print(udeps)
        print(tokens_text)
        return

    udeps_heads = diaparse_sentence.values[6]
    assert len(udeps) == len(tokens_text)

    #disp = diaparse.sentences[0].to_displacy()
    disp = diaparse_sentence.to_displacy()
    arcs = disp['arcs']
    
    words = {i: w['text'] for i, w in enumerate(disp['words'])}
    
    ##copulae
    verbs_cop = [a for a in arcs if a['label']=='cop']

    verbs += [tokens_text[v['start']] for v in verbs_cop if v['dir']=='left']
    verbs += [tokens_text[v['end']] for v in verbs_cop if v['dir']=='right']
    if False:
        verbs_root = [a for a in arcs if a['label']=='root']
        verbs += [tokens_text[v['end']] for v in verbs_root if v['dir']=='right'
                if not tokens_text[v['end']] in verbs]

    verbs = sorted(verbs, key=lambda x: x.i)
    verbs_dict = {x.i+1: x for x in sorted(verbs, key=lambda x: x.i)}
    verbs_dict_num = {x.i+1: i for i, x in enumerate(verbs)}


    for token in verbs:
        token_i = token.i + 1

        pred_arcs_ = [a for a in arcs if (a['start']==token_i and a['dir']=='right')
                    or (a['end']==token_i and a['dir']=='left')]

        token_cop_i = [a['end'] for a in arcs if
                (a['start']==token_i and a['dir']=='left') and a['label']=='cop']
        if len(token_cop_i) == 0:
            token_cop_i = [a['start'] for a in arcs if ((a['start']==token_i and a['dir']=='left')
                        or (a['end']==token_i and a['dir']=='right')) and a['label']=='cop']

        if len(token_cop_i) == 1:
            token_cop_i = token_cop_i[0]
        else:
            token_cop_i = None

        pred_arcs_cop = [a for a in arcs if (a['start']==token_cop_i and a['dir']=='right')
                    or (a['end']==token_cop_i and a['dir']=='left')]
        
        
        rels_dict = {
                -2: {'text': 'positive', 'role': ''}, 
                -1: {'text': '', 'role': ''},
                0: {'text': '', 'role': ''}, 
                1: {'text': '', 'role': ''},
                2: {'text': '', 'role': ''},
                3: {'text': '', 'role': ''}, 
                4: {'text': '', 'role': ''}, 
                5: {'text': '', 'role': ''}
                }

        pred_arcs = {}
        for arc in pred_arcs_:
            if arc['start']==token_i and arc['dir']=='right':
                pred_arcs[arc['end']] = arc['label']
            elif arc['end']==token_i and arc['dir']=='left':
                pred_arcs[arc['start']] = arc['label']
                
        assert pred_arcs == get_children(arcs, token_i)
        ##copulae
        for arc in pred_arcs_cop:
            if arc['start']==token_cop_i and arc['dir']=='right':
                pred_arcs[arc['end']] = arc['label']
            elif arc['end']==token_cop_i and arc['dir']=='left':
                pred_arcs[arc['start']] = arc['label']                        
        if not token_cop_i is None and token_i != token_cop_i:
            pred_arcs[token_cop_i] = 'obj'
        

        cprt = [k for k, v in pred_arcs.items() if v=='compound:prt']
        rels_dict[-1] = {
            'role': 'PRED',
            'text': lemmas[token.i+1],
            'token': token.text,
            'i': token_i
           }
        if not token_cop_i is None and token_i != token_cop_i:
            rels_dict[-1]['cop'] = True
        else:
            rels_dict[-1]['cop'] = False

        try:
            if len(cprt) == 1:
                rels_dict[5] = {
                    'role': 'VG',
                    'text': lemmas[cprt[0]+1],
                    'token': tokens[cprt[0]].text,
                    'i': cprt
                   }
        except (KeyError, IndexError):
            return

        pred_arcs = {k: v for k, v in pred_arcs.items() if v!='conj'}
        for i, arc in pred_arcs.items():                        
            if opt_dependents == 'first level' and tokens[i-1].pos_ != 'VERB':
                children = [k for k, v in get_dependents(arcs, i).items() 
                        if not v in ['punct']]
                i_ = [i] + children
            elif opt_dependents == 'second level' and tokens[i-1].pos_ != 'VERB':
                children = [k for k, v in get_dependents_l2(arcs, i).items() 
                        if not v in ['punct']]
                i_ = [i] + children
            elif opt_dependents == 'all' and tokens[i-1].pos_ != 'VERB':
                children = get_dependents_recursive(arcs, i, tokens)
                i_ = [i] + children if not children is None else [i]
            else:
                i_ = [i]

            assert isinstance(i_, list)

            if arc == 'nsubj':
                t_i = token_i
                cop_ = [a for a in arcs if a['label']=='cop' and
                    ((a['end']==t_i  and a['dir']=='right')
                        or (a['start']==t_i  and a['dir']=='left'))]
                if len(cop_) == 1:
                    t_i = cop_[0]['start'] if cop_[0]['dir'] == 'right' else cop_[0]['end']
                acl_ = [a for a in arcs if a['label']=='acl' and
                    ((a['end']==t_i  and a['dir']=='right')
                        or (a['start']==t_i  and a['dir']=='left'))]

                #TODO if len(acl_) == 1 and tokens[i-1].pos_=='PRON':
                if len(acl_) == 1 and tokens[i-1].tag_=='PRELS':
                    acl = acl_[0]['start'] if acl_[0]['dir'] == 'right' else acl_[0]['end']
                    if opt_dependents == 'first level':
                        children = [k for k in get_dependents(arcs, acl).keys()]
                        i_ = [acl] + children if not children is None else [acl]
                    elif opt_dependents == 'second level':
                        children = [k for k in get_dependents_l2(arcs, acl).keys()]
                        i_ = [acl] + children if not children is None else [acl]
                    elif opt_dependents == 'all':
                        children = get_dependents_recursive(arcs, acl, tokens)
                        i_ = [acl] + children if not children is None else [acl]
                    else:
                        i_ = [acl]

                    role = 0
                    if not token_cop_i is None:
                        role = 1
                    rels_dict[role] = {
                        'role': f'ARG{role}',
                        'text': lemmas[acl],
                        'token': words[acl],
                        'i': i_
                    }
                    rels_dict[role+3] = {
                        'role': f'R-ARG{role}',
                        'text': lemmas[i],
                        'token': words[i],
                        'i': [i]
                    }
                else:
                    role = 0
                    if not token_cop_i is None:
                        role = 1
                    rels_dict[role] = {
                        'role': f'ARG{role}',
                        'text': lemmas[i],
                        'token': words[i],
                        'i': i_
                    }
            elif arc == 'nsubj:pass':
                ex_arc = rels_dict[1]['text']
                if ex_arc != '':
                    rels_dict[2] = {
                        'role': 'ARG2',
                        'text': lemmas[i],
                        'token': words[i],
                        'i': i_
                    }
                    
                    pass_arcs_ = [a for a in arcs if (a['start']==i and a['dir']=='right')
                        or (a['end']==i and a['dir']=='left') and a['label']=='nmod']
                    pass_arcs = {}
                    for parc in pass_arcs_:
                        if parc['start']==i and parc['dir']=='right':
                            pass_arcs[parc['end']] = parc['label']
                        elif parc['end']==i and parc['dir']=='left':
                            pass_arcs[parc['start']] = parc['label']
                    ex_arc2 = rels_dict[0]['text']
                    if ex_arc2 == '' and len(pass_arcs) == 1:
                        rels_dict[0] = {
                            'role': 'ARG0',
                            'text': lemmas[list(pass_arcs.keys())[0]],
                            'token': words[list(pass_arcs.keys())[0]],
                            'i': list(pass_arcs.keys()),
                        } 
                else:                            
                    acl_ = [a for a in arcs if a['label']=='acl' and
                        ((a['end']==token_i  and a['dir']=='right')
                            or (a['start']==token_i  and a['dir']=='left'))]
                    if len(acl_) == 1 and tokens[i-1].tag_=='PRELS':
                        acl = acl_[0]['start'] if acl_[0]['dir'] == 'right' else acl_[0]['end']
                        if opt_dependents == 'first level':
                            children = [k for k in get_dependents(arcs, acl).keys()]
                            i_ = [acl] + children if not children is None else [acl]
                        elif opt_dependents == 'second level':
                            children = [k for k in get_dependents_l2(arcs, acl).keys()]
                            i_ = [acl] + children if not children is None else [acl]
                        elif opt_dependents == 'all':
                            children = get_dependents_recursive(arcs, acl, tokens)
                            i_ = [acl] + children if not children is None else [acl]
                        else:
                            i_ = [acl]

                        rels_dict[1] = {
                            'role': 'ARG1',
                            'text': lemmas[acl],
                            'token': words[acl],
                            'i': i_
                        }
                        rels_dict[4] = {
                            'role': 'R-ARG1',
                            'text': lemmas[i],
                            'token': words[i],
                            'i': [i]
                        }
                    else:
                        role = 1
                        rels_dict[role] = {
                            'role': f'ARG{role}',
                            'text': lemmas[i],
                            'token': words[i],
                            'i': i_
                        }
            #if ARG1 full -> ARG2 (iobj has priority) set intersection + NOUN
            elif arc == 'obl' and 'aux:pass' in pred_arcs.values():
            #elif arc == 'obl' and 'nsubj:pass' in pred_arcs.values():
                children = [lemmas_spacy[j] for j in get_dependents_recursive(arcs, i, tokens)]
                if 'von' in children or 'durch' in children:
                    rels_dict[0] = {
                        'role': 'ARG0',
                        'text': lemmas[i],
                        'token': words[i],
                        'i': i_
                    }                        
            elif arc == 'obj':
                #TODO
                t_i = token_i
                cop_ = [a for a in arcs if a['label']=='cop' and
                    ((a['end']==t_i  and a['dir']=='right')
                        or (a['start']==t_i  and a['dir']=='left'))]
                if len(cop_) == 1:
                    t_i = cop_[0]['start'] if cop_[0]['dir'] == 'right' else cop_[0]['end']
                acl_ = [a for a in arcs if a['label']=='acl' and
                    ((a['end']==t_i  and a['dir']=='right')
                        or (a['start']==t_i  and a['dir']=='left'))]


                
                #acl_ = [a for a in arcs if a['label']=='acl' and
                #    ((a['end']==token_i  and a['dir']=='right')
                #        or (a['start']==token_i  and a['dir']=='left'))]
                #if len(acl_) == 1 and tokens[i-1].pos_=='PRON':
                if len(acl_) == 1 and tokens[i-1].tag_=='PRELS':
                    appos = None
                    acl = acl_[0]['start'] if acl_[0]['dir'] == 'right' else acl_[0]['end']
                    appos_ = [a for a in arcs if a['label']=='appos' and
                        ((a['end']==acl  and a['dir']=='right')
                            or (a['start']==acl  and a['dir']=='left'))]
                    if len(appos_) == 1:
                        appos = appos_[0]['start'] if appos_[0]['dir'] == 'right' else appos_[0]['end']
                        children = get_dependents_recursive(arcs, appos, tokens,
                                stop_dep='appos')
                        i_ = [appos] + children if not children is None else [appos]
                    else:
                        children = get_dependents_recursive(arcs, acl, tokens)

                        i_ = [acl] + children if not children is None else [acl]

                    role = 1
                    if not token_cop_i is None:
                        role = 2
                    rels_dict[role] = {
                        'role': f'ARG{role}',
                        'text': lemmas[acl],
                        'token': words[acl],
                        'i': i_
                    }
                    rels_dict[role+3] = {
                        'role': f'R-ARG{role}',
                        'text': lemmas[i],
                        'token': words[i],
                        'i': [i] if appos is None else [i, acl]
                    }
                else:
                    role = 1
                    if not token_cop_i is None:
                        role = 2
                    rels_dict[role] = {
                        'role': f'ARG{role}',
                        'text': lemmas[i],
                        #'text': lemmas[words[i]],
                        'token': words[i],
                        'i': i_
                    }
            ##TODO Der Schmetterling fliegt auf die Blüte.
            elif arc == 'iobj':# or arc == 'obl':
                rels_dict[2] = {
                    'role': 'ARG2',
                    'text': lemmas[i],
                    'token': words[i],
                    'i': i_
                }
            elif arc == 'xcomp':
                ex_arc = rels_dict[0]['text']
                if ex_arc != '':
                    ex_arc = rels_dict[1]['text']
                    if ex_arc != '':
                        ex_arc = rels_dict[2]['text']
                        if ex_arc == '':
                            children = get_dependents_recursive(arcs, i, tokens)
                            i_ = [i] + children if not children is None else [i]
                            rels_dict[2] = {
                                'role': 'ARG2',
                                'text': lemmas[i],
                                'token': words[i],
                                'i': i_
                            }
                    else:
                        children = get_dependents_recursive(arcs, i, tokens)
                        i_ = [i] + children if not children is None else [i]
                        rels_dict[1] = {
                            'role': 'ARG1',
                            'text': lemmas[i],
                            'token': words[i],
                            'i': i_
                        }

                else:
                    ex_arc = rels_dict[1]['text']
                    role = 1
                    if ex_arc == '':
                        role = 2
                        ex_arc = rels_dict[2]['text']

                    if ex_arc == '':
                        children = get_dependents_recursive(arcs, i, tokens)
                        i_ = [i] + children if not children is None else [i]
                        rels_dict[role] = {
                        'role': f'ARG{role}',
                        'text': lemmas[i],
                        'token': words[i],
                        'i': i_
                        }
                    else:
                        ex_arc = rels_dict[2]['text']
                        role = 2

                        if ex_arc == '':
                            children = get_dependents_recursive(arcs, i, tokens)
                            i_ = [i] + children if not children is None else [i]
                            rels_dict[role] = {
                            'role': f'ARG{role}',
                            'text': lemmas[i],
                            'token': words[i],
                            'i': i_
                            }
            elif arc == 'ccomp':
                ex_arc = rels_dict[1]['text']
                children = get_dependents_recursive(arcs, i, tokens)
                #i_ = [i] + children
                i_ = [i] + children if not children is None else [i]
                if ex_arc == '':
                    rels_dict[1] = {
                        'role': 'ARG1',
                        'text': lemmas[i],
                        #'text': lemmas[words[i]],
                        'token': words[i],
                        'i': i_
                    }
            elif arc == 'csubj:pass':# or arc == 'advcl':
                children = get_dependents_recursive(arcs, i, tokens)
                i_ = [i] + children if not children is None else [i]
                ex_arc = rels_dict[2]['text']
                if ex_arc == '':
                    rels_dict[2] = {
                        'role': 'ARG2',
                        'text': lemmas[i],
                        #'text': lemmas[words[i]],
                        'token': words[i],
                        'i': i_
                    }

            #TODO
            #Biologen sprechen im Falle der genetischen Vielfalt lieber von 
            #genetischer Uniformität .
            #Die Schau stand für drei Wochen zur Verfügung und Besucher konnten
            #eine Bewertung vornehmen : Welches der Quadrate 
            #wirkt am natürlichsten , am wertvollsten oder einfach am schönsten ?
            #op, cvc
            elif arc == 'obl':
                children_spacy = [x for x in token.children if x.dep_=='cvc' or x.dep_=='op']
                if len(children_spacy) > 0:
                    for child in children_spacy:
                        grandchildren_spacy = [x.i for x in child.children if x.i==i-1]
                        if len(grandchildren_spacy) > 0:
                            children = get_dependents_recursive(arcs, i, tokens)
                            i_ = [i] + children if not children is None else [i]
                            #TODO
                            if rels_dict[1]['role'] == '':
                                #TODO
                                role = 2
                            else:
                                role = 2
                            rels_dict[role] = {
                                'role': f'ARG{role}',
                                'text': lemmas[i],
                                'token': words[i],
                                'i': i_
                            }                        
                else:
                    children = get_dependents_recursive(arcs, i, tokens)
                    i_ = [i] + children if not children is None else [i]
                    ner = ners[i]
                    if 'LOC' in ner:
                        rels_dict[6] = {
                            'role': f'ARGM-LOC',
                            'text': lemmas[i],
                            'token': words[i],
                            'i': i_
                        }                        
                    else:
                        try:
                            _ = rels_dict[6]
                        except KeyError:
                            rels_dict[6] = {
                                'role': f'ARGM' ,
                                'text': lemmas[i],
                                'token': words[i],
                                'i': i_
                            }                        
                    #rels_dict[6] = {
                    #    'role': f'ARGM',
                    #    'text': lemmas[i],
                    #    'token': words[i],
                    #    'i': i_
                    #}                        
            if deps[i] == 'ng':
                rels_dict[-2] = {
                    'role': '',
                    'text': 'negative',
                    }

        if rels_dict[0]['text'] == '':                    
            lemma_verb = None
            head_arc_i, head_arc_label = get_head_arc(arcs, token_i)[0]
            try:
                lemma_verb = lemmas[head_arc_i]
            except KeyError:
                pass
            if head_arc_label == 'xcomp':
                if lemma_verb in verbs_object:
                    head_arc_children = get_children(arcs, head_arc_i)
                    for arc_c_i, arc_c_label in head_arc_children.items():
                        if arc_c_label == 'obj':
                            children = get_dependents_recursive(arcs, arc_c_i, tokens)
                            i_ = [arc_c_i] + children if not children is None else [arc_c_i]
                            rels_dict[0] = {
                            'role': 'ARG0',
                            #'text': lemmas[words[arc_c_i]],
                            'text': lemmas[arc_c_i],
                            'token': words[arc_c_i],
                            'i': i_
                            }
                else:# lemma_verb in verbs_subject:                        
                    head_arc_children = get_children(arcs, head_arc_i)
                    for arc_c_i, arc_c_label in head_arc_children.items():
                        if arc_c_label == 'nsubj':
                            pas = [ck for ck, cv in get_children(arcs, arc_c_i).items()
                                    if ':pass' in cv]
                            #if passive ARG0 -> ARG1
                            if len(pas) > 0:
                                role = 1
                            else:
                                role = 0

                            children = get_dependents_recursive(arcs, arc_c_i, tokens)
                            i_ = [arc_c_i] + children if not children is None else [arc_c_i]
                            rels_dict[0] = {
                            'role': 'ARG0',
                            'text': lemmas[arc_c_i],
                            #'text': lemmas[words[arc_c_i]],
                            'token': words[arc_c_i],
                            'i': i_
                            }

        
        if len(rels_dict) > 0:
            #TODO
            try:
                rels_dict[1]['i'] = [x for x in rels_dict[1]['i']
                        if not x in rels_dict[3]['i']]
            except KeyError:
                pass
            try:
                rels_dict[2]['i'] = [x for x in rels_dict[2]['i']
                    if not x in rels_dict[4]['i']]
            except KeyError:
                pass
            if token_cop_i is None:
                t_i = token_i
            else:
                t_i = token_cop_i
            for arc in arcs:
                if arc['dir'] == 'right' and arc['end'] == t_i and arc['label'] == 'conj':
                    cop_root = [ck for ck, cv in get_children(arcs, arc['start']).items()
                        if cv=='cop']
                    pas = [ck for ck, cv in get_children(arcs, arc['end']).items()
                            if ':pass' in cv]
                    #if passive ARG0 -> ARG1
                    if len(pas) > 0:
                        role = 1
                    else:
                        role = 0
                    #copula? 
                    if len(cop_root) == 0:
                        try:
                            verb_i = verbs_dict_num[arc['start']]
                        except KeyError:
                            #TODO
                            return
                        pass_root = [ck for ck, cv
                            in get_children(arcs, arc['start']).items() if ':pass' in cv]
                        if len(pass_root) > 0:
                            role_root = 1
                        else:
                            role_root = 0
                        cop = [ck for ck, cv in get_children(arcs, i).items()
                            if cv=='cop']
                        if len(cop) > 0:
                            if role == 0:
                                role = 1
                            elif role == 1:
                                role = 2

                        try:
                            psrs_0 = {k: v for k, v in psrs[verb_i][role_root].items()}
                        except IndexError:
                            return
                        psrs_0['role'] = f'ARG{role}'
                        if rels_dict[role]['role'] == '':
                            rels_dict[role] = psrs_0
                    elif len(cop_root) > 0:
                        try:
                            verb_i = verbs_dict_num[cop_root[0]]
                        except KeyError:
                            return
                        try:
                            psrs_0 = {k: v for k, v in psrs[verb_i][1].items()}
                        except IndexError:
                            return

                        cop = [ck for ck, cv in get_children(arcs, i).items()
                            if cv=='cop']
                        if len(cop) > 0:
                            if role == 0:
                                role = 1
                            elif role == 1:
                                role = 2
                        #TODO
                        psrs_0['role'] = f'ARG{role}'
                        if rels_dict[role]['role'] == '':
                            rels_dict[role] = psrs_0
            psrs.append(rels_dict)
    if opt_appos:
        for arc in arcs:
            if arc['label'] == 'appos':
                rels_dict = {
                        -2: {'text': 'positive', 'role': ''}, 
                        -1: {'text': '', 'role': ''},
                        0: {'text': '', 'role': ''}, 
                        1: {'text': '', 'role': ''}, 
                        2: {'text': '', 'role': ''}
                        }
                rels_dict[-1] = {
                    'role': 'PRED',
                    'text': '_appos_',
                    'token': '_appos_',
                   }
                if arc['dir'] == 'right':
                    i_0 = arc['end']
                    i_1 = arc['start']
                elif arc['dir'] == 'left':
                    i_0 = arc['start']
                    i_1 = arc['end']

                if opt_dependents == 'first level' and tokens[i_0-1].pos_ != 'VERB':
                    children = [k for k, v in get_dependents(arcs, i_0).items() 
                            if not v in ['punct']]
                    i_ = [i_0] + children
                elif opt_dependents == 'second level' and tokens[i_0-1].pos_ != 'VERB':
                    children = [k for k, v in get_dependents_l2(arcs, i_0).items() 
                            if not v in ['punct']]
                    i_ = [i_0] + children
                elif opt_dependents == 'all' and tokens[i_0-1].pos_ != 'VERB':
                    children = [k for k in get_dependents_recursive(arcs, i_0, tokens)] 
                    i_ = [i_0] + children
                else:
                    i_ = [i_0]

                rels_dict[0] = {
                'role': 'ARG2',
                'text': lemmas[i_0],
                'token': words[i_0],
                'i': i_
                }

                if opt_dependents == 'first level' and tokens[i_1-1].pos_ != 'VERB':
                    children = [k for k, v in get_dependents(arcs, i_1).items() 
                            if not v in ['punct']]
                    i_ = [i_1] + children
                elif opt_dependents == 'second level' and tokens[i_1-1].pos_ != 'VERB':
                    children = [k for k, v in get_dependents_l2(arcs, i_1).items() 
                            if not v in ['punct']]
                    i_ = [i_1] + children
                elif opt_dependents == 'all' and tokens[i_1-1].pos_ != 'VERB':
                    children = [k for k in get_dependents_recursive(arcs, i_1, tokens)]
                    i_ = [i_1] + children
                else:
                    i_ =  [i_1]

                assert isinstance(i_, list)
                rels_dict[1] = {
                'role': 'ARG1',
                'text': lemmas[i_1],
                'token': words[i_1],
                'i': i_
                }

                psrs.append(rels_dict)

    #psrs_all.append((words, tokens, tags_hanta, text_hash, diaparse.sentences[0].to_displacy(), psrs))
        


    return words, [t.text for t in tokens], tags, udeps, udeps_heads, psrs
