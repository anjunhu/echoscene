from __future__ import print_function
import sys
sys.path.append("..")
sys.path.append(".")
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
import json
from helpers.psutil import FreeMemLinux
from helpers.util import scale_box_params, standardize_box_params
import clip
import random
import pickle
import h5py
from tqdm import tqdm


changed_relationships_dict = {
        'left': 'right',
        'right': 'left',
        'front': 'behind',
        'behind': 'front',
        'bigger than': 'smaller than',
        'smaller than': 'bigger than',
        'taller than': 'shorter than',
        'shorter than': 'taller than',
        'close by': 'close by',
        'same style as': 'same style as',
        'same super category as': 'same super category as',
        'same material as': 'same material as',
        "symmetrical to": "symmetrical to",
        "standing on":"standing on",
        "above":"above"
    }

def load_ckpt(ckpt):
    map_fn = lambda storage, loc: storage
    if type(ckpt) == str:
        state_dict = torch.load(ckpt, map_location=map_fn)
    else:
        state_dict = ckpt
    return state_dict

class AnySceneDatasetSceneGraph(data.Dataset):
    def __init__(self, root, split=None, root_3dfront='', shuffle_objs=False, pass_scan_id=False, use_SDF=False,
                 use_scene_rels=False, data_len=None, with_changes=True, scale_func='diag', eval=False,
                 eval_type='addition', with_CLIP=True, bin_angle=False, rel_json_file='relationships_anyscene.json',
                 seed=True, large=False, recompute_feats=True, recompute_clip=True, room_type='all'):

        self.room_type = room_type
        self.seed = seed
        self.with_CLIP = with_CLIP
        self.cond_model = None
        self.large = large
        self.recompute_feats = recompute_feats
        self.recompute_clip = recompute_clip

        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        self.scale_func = scale_func
        self.with_changes = with_changes
        self.use_SDF = use_SDF
        self.sdf_res = 64
        self.root = root
        # list of class categories
        self.catfile = os.path.join(self.root,'classes_{}.txt'.format(self.room_type))
        self.cat = {}        
        self.scans = []
        self.obj_paths = []
        self.data_len = data_len
        self.use_scene_rels = use_scene_rels
        self.bin_angle = bin_angle

        self.fm = FreeMemLinux('GB')
        self.vocab = {}
            
        with open(os.path.join(self.root, 'relationships.txt'), "r") as f:
            self.vocab['pred_idx_to_name'] = ['in\n']
            self.vocab['pred_idx_to_name']+=f.readlines()

        # list of relationship categories
        self.relationships = self.read_relationships(os.path.join(self.root, 'relationships.txt'))
        self.relationships_dict = dict(zip(self.relationships,range(1,len(self.relationships)+1)))
        self.relationships_dict_r = dict(zip(self.relationships_dict.values(), self.relationships_dict.keys()))

        self.box_normalized_stats = os.path.join(self.root, 'centered_bounds_anyscene.txt')

        self.rel_json_file = os.path.join(self.root, rel_json_file)
        self.box_json_file = os.path.join(self.root, 'obj_boxes_anyscene.json')

        self.relationship_json, self.objs_json, self.tight_boxes_json = \
                self.read_relationship_json(self.rel_json_file, self.box_json_file)

        for scene, infos in self.tight_boxes_json.items():
            for id, info in infos.items():
                if 'model_path' in info:
                    if info['model_path']:
                        info['model_path'] = root + info['model_path'][36:]

        self.padding = 0.2
        self.eval = eval
        self.pass_scan_id = pass_scan_id
        self.shuffle_objs = shuffle_objs
        self.root_3dfront = root_3dfront
        if self.root_3dfront == '':
            self.root_3dfront = os.path.join(self.root, 'visualization')
            if not os.path.exists(self.root_3dfront):
                os.makedirs(self.root_3dfront)

        self.files = {}
        self.eval_type = eval_type

        # check if all clip features exist. If not they get generated here (once)
        if self.with_CLIP:
            self.cond_model, preprocess = clip.load("ViT-B/32", device='cuda')
            self.cond_model_cpu, preprocess_cpu = clip.load("ViT-B/32", device='cpu')
            # print('loading CLIP')
            # print('Checking for missing clip feats. This can be slow the first time.')
            # for index in tqdm(range(len(self))):
            #     self.__getitem__(index)
            self.recompute_clip = True

    def read_relationship_json(self, json_file, box_json_file):
        """ Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)
        
        # print(json_file, box_json_file, box_data.keys())
        
        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for scan in data['scans']:

                relationships = []
                for relationship in scan["relationships"]:
                    relationship[2] -= 1
                    relationships.append(relationship)

                # for every scan in rel json, we append the scan id
                rel[scan["scan"]] = relationships
                self.scans.append(scan["scan"])

                objects = {}
                boxes = {}
                for k, v in scan["objects"].items():
                    # if not self.large:
                    #     objects[int(k)] = self.mapping_full2simple[v]
                    # else:
                    objects[int(k)] = v

                    try:
                        boxes[int(k)] = {}
                        boxes[int(k)]['param7'] = box_data[scan["scan"]][k]["param7"]
                        boxes[int(k)]['param7'][6] = boxes[int(k)]['param7'][6]
                        boxes[int(k)]['scale'] = box_data[scan["scan"]][k]["scale"]
                    except Exception as e:
                        # probably box was not saved because there were 0 points in the instance!
                        print(e)
                    try:
                        boxes[int(k)]['model_path']  = box_data[scan["scan"]][k]["model_path"]
                    except Exception as e:
                        print(e)
                        continue
                
                boxes["scene_center"] = box_data[scan["scan"]]["scene_center"]
                objs[scan["scan"]] = objects
                tight_boxes[scan["scan"]] = boxes
        return rel, objs, tight_boxes

    def read_relationships(self, read_file):
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships

    def get_key(self, dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return None

    def __getitem__(self, index):
        scan_id = self.scans[index]
        instance2label = self.objs_json[scan_id]
        print("instance2label", instance2label)
        keys = list(instance2label.keys())

        if self.shuffle_objs:
            random.shuffle(keys)
        clip_feats_ins = None
        clip_feats_rel = None

        # If true, expected paths to saved clip features will be set here
        if self.with_CLIP:
            self.clip_feats_path = os.path.join(self.root_3dfront, scan_id,
                                                'CLIP_{}.pkl'.format(scan_id))
            if not self.large:
                self.clip_feats_path = os.path.join(self.root_3dfront, scan_id,
                                                        'CLIP_small_{}.pkl'.format(scan_id))
            if not os.path.exists(os.path.join(self.root_3dfront, scan_id)):
                os.makedirs(os.path.join(self.root_3dfront, scan_id))
            if self.recompute_clip:
                self.clip_feats_path += 'tmp'

        instance2mask = {}
        instance2mask[0] = 0

        cat_ids = []
        cat_ids_grained = []
        tight_boxes = []

        counter = 0

        instances_order = []
        selected_shapes = []
        obj_sdf_list = []

        # key: 1 of {1: 'floor', ...,} instance_id              keys: whole instance ids
        for key in keys:
            scene_instance_id = key
            scene_instance_class = instance2label[key]
            if not self.large:
                scene_class_id_grained = key 
                scene_instance_class = key 
                scene_class_id = key 
                # print(scene_class_id, scene_instance_class, scene_class_id_grained)
            else:
                scene_class_id = self.classes[scene_instance_class] # class id in the entire dataset ids
                # print(scene_class_id)

            instance2mask[scene_instance_id] = counter + 1
            counter += 1

            if (scene_class_id >= 0) and (scene_instance_id > 0):
                selected_shapes.append(True)
                cat_ids.append(scene_class_id)
                if not self.large:
                    cat_ids_grained.append(scene_class_id_grained)
                else:
                    cat_ids_grained.append(scene_class_id)
                bbox = np.array(self.tight_boxes_json[scan_id][key]['param7'].copy())
                bbox[3:6] -= np.array(self.tight_boxes_json[scan_id]['scene_center'])
                instances_order.append(key)
                if self.bin_angle: # bin angle is not possible for current diffusion
                    bins = np.linspace(np.deg2rad(-180), np.deg2rad(180), 24)
                    bin_angle = np.digitize(bbox[6], bins)
                    bbox[6] = bin_angle
                    bbox[0:6] = standardize_box_params(bbox[0:6], params=6,file=self.box_normalized_stats)
                else:
                    bbox = scale_box_params(bbox, file=self.box_normalized_stats, angle=False) # need to change angle after
                tight_boxes.append(bbox)

            if self.use_SDF:
                if self.tight_boxes_json[scan_id][key]["model_path"] is None:
                    obj_sdf_list.append(torch.zeros((1, self.sdf_res, self.sdf_res, self.sdf_res))) # floor
                else:
                    sdf_path = os.path.join(self.tight_boxes_json[scan_id][key]["model_path"].replace('3D-FUTURE-model', "3D-FUTURE-SDF").rsplit('/', 1)[0], 'ori_sample_grid.h5')
                    h5_f = h5py.File(sdf_path, 'r')
                    obj_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
                    sdf = torch.Tensor(obj_sdf).view(1, self.sdf_res, self.sdf_res, self.sdf_res)
                    sdf = torch.clamp(sdf, min=-0.2, max=0.2)
                    obj_sdf_list.append(sdf)
            else:
                obj_sdf_list = None

        triples = []
        words = []
        rel_json = self.relationship_json[scan_id]
        for r in rel_json: # create relationship triplets from data
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():
                subject = r[0]
                object = r[1]
                predicate = r[2] + 1
                print(subject, predicate, object)
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
                    if not self.large:
                        words.append(instance2label[r[0]] + ' ' + r[3] + ' ' + instance2label[r[1]])
                    else:
                        words.append(instance2label[r[0]]+' '+r[3]+' '+instance2label[r[1]]) # TODO check
                    print(words[-1])
            else:
                continue

        if self.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            scene_idx = len(cat_ids)
            for i, ob in enumerate(cat_ids):
                triples.append([i, 0, scene_idx])
                words.append(list(instance2label.values())[i%len(instance2label)] + ' ' + 'in' + ' ' + 'room')
            cat_ids.append(0)
            cat_ids_grained.append(0)
            # dummy scene box
            tight_boxes.append([-1, -1, -1, -1, -1, -1, -1])
            if self.use_SDF:
                obj_sdf_list.append(torch.zeros((1, self.sdf_res, self.sdf_res, self.sdf_res))) # _scene_
                
        output = {}
        
        # print(cat_ids)
        if self.with_CLIP: 
            num_cat = len(cat_ids); print("\nnum_cat", num_cat)
            feats_rel = {}
            obj_cat = []
            with torch.no_grad():
                for i in instance2label.keys():
                    obj_cat.append(instance2label[i])
                    # obj_cat.append(self.get_key(instance2label, cat_ids[i]))
                if self.use_scene_rels:
                    # obj_cat.pop()
                    obj_cat.append('room') # TODO check
                print("\nCLIPing objects\n", obj_cat)
                text_obj = clip.tokenize(obj_cat).to('cuda')
                feats_ins = self.cond_model.encode_text(text_obj).detach().cpu().numpy()
                text_rel = clip.tokenize(words).to('cuda')
                print("\nCLIPing relationships:\n", words)
                rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
                for i in range(len(words)):
                    feats_rel[words[i]] = rel[i]

            clip_feats_in = {}
            clip_feats_in['instance_feats'] = feats_ins
            clip_feats_in['instance_order'] = instances_order # this doesn't include room (_scene_)
            clip_feats_in['rel_feats'] = feats_rel
            # print(instances_order)
            path = os.path.join(self.clip_feats_path)
            if self.recompute_clip:
                path = path[:-3]

            pickle.dump(clip_feats_in, open(path, 'wb'))
            clip_feats_ins = list(clip_feats_in['instance_feats'])
            clip_feats_rel = clip_feats_in['rel_feats']


        # prepare outputs
        output['encoder'] = {}
        output['encoder']['objs'] = cat_ids
        output['encoder']['objs_grained'] = cat_ids_grained # not needed for encoder
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = tight_boxes
        output['encoder']['words'] = words

        if self.with_CLIP:
            output['encoder']['text_feats'] = clip_feats_ins
            clip_feats_rel_new = []
            if clip_feats_rel != None:
                for word in words:
                    clip_feats_rel_new.append(clip_feats_rel[word])
                    # print(clip_feats_rel_new)
                output['encoder']['rel_feats'] = clip_feats_rel_new
        
        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:
            if not self.eval:
                if self.with_changes:
                    output['manipulate']['type'] = ['relationship', 'addition', 'none'][
                        np.random.randint(3)]  # removal is trivial - so only addition and rel change
                else:
                    output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed, words_removed = self.remove_node_and_relationship(output['encoder'], instance2label)
                    if node_id >= 0:
                        output['manipulate']['added_node_id'] = node_id
                        output['manipulate']['added_node_class'] = node_removed
                        output['manipulate']['added_node_clip'] = node_clip_removed
                        output['manipulate']['added_triples'] = triples_removed
                        output['manipulate']['added_triples_clip'] = triples_clip_removed
                        output['manipulate']['added_words'] = words_removed
                    else:
                        output['manipulate']['type'] = 'none'
                elif output['manipulate']['type'] == 'relationship':
                    rel, original_triple, suc = self.modify_relship(output['encoder'], instance2label) # why modify encoder side? Because the changed edge doesn't need to make sense. I need to make sure that the one from the decoder side makes sense
                    if suc:
                        output['manipulate']['original_relship'] = (rel, original_triple)
                    else:
                        output['manipulate']['type'] = 'none'
            else:
                print(len(output['encoder']['objs']))
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed, words_removed = self.remove_node_and_relationship(output['encoder'], instance2label)
                    if node_id >= 0:
                        output['manipulate']['added_node_id'] = node_id
                        output['manipulate']['added_node_class'] = node_removed
                        output['manipulate']['added_node_clip'] = node_clip_removed
                        output['manipulate']['added_triples'] = triples_removed
                        output['manipulate']['added_triples_clip'] = triples_clip_removed
                        output['manipulate']['added_words'] = words_removed
                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    # this should be modified from the decoder side, because during test we have to evaluate the real setting, and can make the change have the real meaning.
                    rel, original_triple, suc = self.modify_relship(output['decoder'], instance2label, interpretable=True)
                    if suc:
                        output['manipulate']['original_relship'] = (rel, original_triple)
                    else:
                        return -1
        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64)) # this is changed
        output['encoder']['objs_grained'] = torch.from_numpy(np.array(output['encoder']['objs_grained'], dtype=np.int64)) # this doesn't matter
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['encoder']['text_feats'] = torch.from_numpy(np.array(output['encoder']['text_feats'], dtype=np.float32)) # this is changed
            output['encoder']['rel_feats'] = torch.from_numpy(np.array(output['encoder']['rel_feats'], dtype=np.float32))

        # these two should have the same amount.
        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))
        output['decoder']['objs_grained'] = torch.from_numpy(np.array(output['decoder']['objs_grained'], dtype=np.int64))

        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64)) # this is changed
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['decoder']['text_feats'] = torch.from_numpy(np.array(output['decoder']['text_feats'], dtype=np.float32))
            output['decoder']['rel_feats'] = torch.from_numpy(np.array(output['decoder']['rel_feats'], dtype=np.float32)) # this is changed
        if self.use_SDF:
            output['decoder']['sdfs'] = torch.stack(obj_sdf_list,dim=0)

        output['scan_id'] = scan_id
        output['instance_id'] = instances_order
        output['objs_text'] = obj_cat

        return output


    def remove_node_and_relationship(self, graph, instance2label):
        
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        excluded = [list(instance2label.keys())[-1]]

        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)

        node_removed = graph['objs'].pop(node_id)
        node_clip_removed = None
        if self.with_CLIP:
            node_clip_removed = graph['text_feats'].pop(node_id)

        graph['boxes'].pop(node_id)

        to_rm = []
        triples_clip_removed_list = []
        words_removed_list = []
        for i,x in reversed(list(enumerate(graph['triples']))):
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)
                if self.with_CLIP:
                    triples_clip_removed_list.append(graph['rel_feats'].pop(i))
                    words_removed_list.append(graph['words'].pop(i))

        triples_removed = copy.deepcopy(to_rm)
        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        # node_id: instance_id; node_removed: class_id; triples_removed: relations (sub_id, edge_id, obj_id)
        return node_id, node_removed, node_clip_removed, triples_removed, triples_clip_removed_list, words_removed_list

    def modify_relship(self, graph, instance2label, interpretable=False):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        '''15 same material as' '14 same super category as' '13 same style as' '12 symmetrical to' '11 shorter than' '10 taller than' '9 smaller than'
         '8 bigger than' '7 standing on' '6 above' '5 close by' '4 behind' '3 front' '2 right' '1 left'
         '0: none'''
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [1, 2, 3, 4, 8, 9, 10, 11]
        rel_dict = {1: 'left', 2: 'right', 3: 'front', 4: 'behind', 5: 'close by', 8: 'bigger than', 9: 'smaller than', 10: 'taller than', 11: 'shorter than', 12: 'symmetrical to'}
        did_change = False
        trials = 0
        print(list(instance2label.keys())[-1])
        excluded = [list(instance2label.keys())[-1]]
        eval_excluded = [list(instance2label.keys())[-1]]

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            print("sub, pred, obj", sub, pred, obj)
            trials += 1
            print(graph['objs'])

            if pred == 0:
                continue
            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            if interpretable:
                if graph['objs'][obj] in eval_excluded or graph['objs'][sub] in eval_excluded: # don't use the floor
                    continue
                if pred not in interpretable_rels:
                    continue
                else:
                    new_pred = self.relationships_dict[changed_relationships_dict[self.relationships_dict_r[pred]]]
            else:
                new_pred = np.random.randint(1, 12)
                if new_pred == pred:
                    continue
                # # close by, above, standing on, same material......
                # if self.relationships_dict_r[pred] == changed_relationships_dict[self.relationships_dict_r[pred]]:
                #     new_pred = np.random.randint(1, 12)
                #     if new_pred == pred:
                #         continue
                # # left, right, front, behind, bigger, smaller.....
                # else:
                #     new_pred = self.relationships_dict[changed_relationships_dict[self.relationships_dict_r[pred]]]

            graph['words'][idx] = graph['words'][idx].replace(self.relationships_dict_r[graph['triples'][idx][1]],self.relationships_dict_r[new_pred])
            graph['changed_id'] = idx

            # When interpretable is false, we can make things from the encoder side not existed, so that make sure decoder side is the real data.
            # When interpretable is true, we can make things from the decoder side not existed, so that make sure what we test (encoder side) is the real data.
            graph['triples'][idx][1] = new_pred # this new_pred may even not exist.

            did_change = True

        # idx: idx-th triple; (sub, pred, obj): previous triple
        return idx, (sub, pred, obj), did_change

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            return len(self.scans)


    def collate_fn(self, batch):
        """
        Collate function to be used when wrapping a RIODatasetSceneGraph in a
        DataLoader. Returns a dictionary
        """

        out = {}

        out['scene_points'] = []
        out['scan_id'] = []
        out['instance_id'] = []
        out['objs_text'] = []
        out['missing_nodes'] = []
        out['added_nodes'] = []
        out['missing_nodes_decoder'] = []
        out['manipulated_subs'] = []
        out['manipulated_objs'] = []
        out['manipulated_preds'] = []
        global_node_id = 0
        global_dec_id = 0
        for i in range(len(batch)):
            if batch[i] == -1:
                return -1
            # notice only works with single batches
            out['scan_id'].append(batch[i]['scan_id'])
            out['instance_id'].append(batch[i]['instance_id'])
            out['objs_text'].append(batch[i]['objs_text'])

            if batch[i]['manipulate']['type'] == 'addition':
                out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added_node_id'])
                out['added_nodes'].append(global_dec_id + batch[i]['manipulate']['added_node_id'])
            elif batch[i]['manipulate']['type'] == 'relationship':
                rel, (sub, pred, obj) = batch[i]['manipulate']['original_relship'] # remember that this is already changed in the initial scene graph, which means this triplet is real data.
                # which node is changed in the beginning.
                out['manipulated_subs'].append(global_node_id + sub)
                out['manipulated_objs'].append(global_node_id + obj)
                out['manipulated_preds'].append(pred) # this is the real edge

            global_node_id += len(batch[i]['encoder']['objs'])
            global_dec_id += len(batch[i]['decoder']['objs'])

        for key in ['encoder', 'decoder']:
            all_objs, all_boxes, all_triples = [], [], []
            all_objs_grained = []
            all_obj_to_scene, all_triple_to_scene = [], []
            all_points = []
            all_sdfs = []
            all_text_feats = []
            all_rel_feats = []

            obj_offset = 0

            for i in range(len(batch)):
                if batch[i] == -1:
                    print('this should not happen')
                    continue
                (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

                if 'points' in batch[i][key]:
                    all_points.append(batch[i][key]['points'])
                elif 'sdfs' in batch[i][key]:
                    all_sdfs.append(batch[i][key]['sdfs'])
                if 'text_feats' in batch[i][key]:
                    all_text_feats.append(batch[i][key]['text_feats'])
                if 'rel_feats' in batch[i][key]:
                    if 'changed_id' in batch[i][key]:
                        idx = batch[i][key]['changed_id']
                        if self.with_CLIP:
                            text_rel = clip.tokenize(batch[i][key]['words'][idx]).to('cpu')
                            rel = self.cond_model_cpu.encode_text(text_rel).detach().numpy()
                            batch[i][key]['rel_feats'][idx] = torch.from_numpy(np.squeeze(rel)) # this should be a fake relation from the encoder side


                    all_rel_feats.append(batch[i][key]['rel_feats'])

                num_objs, num_triples = objs.size(0), triples.size(0)

                all_objs.append(batch[i][key]['objs'])
                all_objs_grained.append(batch[i][key]['objs_grained'])
                all_boxes.append(boxes)

                if triples.dim() > 1:
                    triples = triples.clone()
                    triples[:, 0] += obj_offset
                    triples[:, 2] += obj_offset

                    all_triples.append(triples)
                    all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

                all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

                obj_offset += num_objs

            all_objs = torch.cat(all_objs)
            all_objs_grained = torch.cat(all_objs_grained)
            all_boxes = torch.cat(all_boxes)

            all_obj_to_scene = torch.cat(all_obj_to_scene)

            if len(all_triples) > 0:
                all_triples = torch.cat(all_triples)
                all_triple_to_scene = torch.cat(all_triple_to_scene)
            else:
                return -1

            outputs = {'objs': all_objs,
                       'objs_grained': all_objs_grained,
                       'tripltes': all_triples,
                       'boxes': all_boxes,
                       'obj_to_scene': all_obj_to_scene,
                       'triple_to_scene': all_triple_to_scene}

            if len(all_sdfs) > 0:
                outputs['sdfs'] = torch.cat(all_sdfs)
            elif len(all_points) > 0:
                all_points = torch.cat(all_points)
                outputs['points'] = all_points

            if len(all_text_feats) > 0:
                all_text_feats = torch.cat(all_text_feats)
                outputs['text_feats'] = all_text_feats
            if len(all_rel_feats) > 0:
                all_rel_feats = torch.cat(all_rel_feats)
                outputs['rel_feats'] = all_rel_feats
            out[key] = outputs

        return out


if __name__ == "__main__":
    dataset = AnySceneDatasetSceneGraph(
        root="/home/ubuntu/datasets/FRONT",
        use_SDF=False,
        use_scene_rels=True,
        with_changes=True,
        with_CLIP=True,
        large=False,
        seed=False,
        room_type='all',
        recompute_clip=True)
    a = dataset[0]
    txt_list = []
    for x in ['encoder', 'decoder']:
        en_obj = a[x]['objs'].cpu().numpy().astype(np.int32)
        en_triples = a[x]['triples'].cpu().numpy().astype(np.int32)
        #instance
        sub = en_triples[:,0]
        obj = en_triples[:,2]
        #cat
        instance_ids = np.array(sorted(list(set(sub.tolist() + obj.tolist())))) #0-n
        cat_ids = en_obj[instance_ids]
        texts = a['objs_text']
        objs = dict(zip(instance_ids.tolist(),texts))
        objs = {str(key): value for key, value in objs.items()}
        for rel in en_triples[:,1]:
            if rel == 0:
                txt = 'in'
                txt_list.append(txt)
                continue
            txt = dataset.relationships_dict_r[rel]
            txt_list.append(txt)
        txt_list = np.array(txt_list)
        rel_list = np.vstack((sub,obj,en_triples[:,1],txt_list)).transpose()
        print(a['scan_id'])