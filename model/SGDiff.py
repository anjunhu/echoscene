import torch
import torch.nn as nn
import os
from termcolor import colored

class SGDiff(nn.Module):

    def __init__(self, type, diff_opt, vocab, replace_latent=False, with_changes=True,
                 residual=False, gconv_pooling='avg', with_angles=False, clip=True, separated=False):
        super().__init__()
        assert type in ['echoscene', 'echolayout'], '{} is not included'.format(type)

        self.type_ = type
        self.vocab = vocab
        self.with_angles = with_angles
        self.epoch = 0
        self.diff_opt = diff_opt
        assert replace_latent is not None and with_changes is not None
        if self.type_ == 'echoscene':
            from model.EchoScene import Sg2ScDiffModel
            self.diff = Sg2ScDiffModel(vocab, self.diff_opt, diffusion_bs=16, embedding_dim=64, mlp_normalization="batch", separated=separated,
                              gconv_num_layers=5, use_angles=with_angles, replace_latent=replace_latent, residual=residual, use_clip=clip)
        elif self.type_ == 'echolayout':
            from model.EchoLayout import Sg2BoxDiffModel
            self.diff = Sg2BoxDiffModel(vocab, self.diff_opt, diffusion_bs=16, embedding_dim=64, mlp_normalization="batch", separated=separated,
                              gconv_num_layers=5, use_angles=with_angles, replace_latent=replace_latent, residual=residual, use_clip=clip)
        else:
            raise NotImplementedError
        self.diff.optimizer_ini()
        self.counter = 0

    def forward_mani(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, dec_objs, dec_objs_grained,
                     dec_triples, dec_boxes, dec_angles, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes):

        if self.type_ == 'echoscene':
            obj_selected, shape_loss, layout_loss, loss_dict = self.diff.forward(
                enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, dec_objs, dec_objs_grained, dec_triples, dec_boxes,
                encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes,
                manipulated_nodes, dec_sdfs, dec_angles)
        elif self.type_ == 'echolayout':
            obj_selected, shape_loss, layout_loss, loss_dict = self.diff.forward(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, dec_objs, dec_triples, dec_boxes, encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes,
                manipulated_nodes, dec_angles)
        else:
            raise NotImplementedError

        return obj_selected, shape_loss, layout_loss, loss_dict

    def load_optimizer_state_dict_non_strict(self, optimizer, state_dict):
        """
        Load optimizer state dict in a non-strict fashion, handling mismatched parameters.
        """
        # Get current optimizer state
        current_state = optimizer.state_dict()
        
        # Load param_groups while handling mismatches
        for saved_group, current_group in zip(state_dict['param_groups'], current_state['param_groups']):
            # Copy matching keys from saved group
            for key in set(saved_group.keys()) & set(current_group.keys()):
                current_group[key] = saved_group[key]
        
        # Handle state dict
        saved_state = state_dict['state']
        
        # Create mapping of parameter IDs to parameters
        current_params = {}
        for group in optimizer.param_groups:
            for param in group['params']:
                current_params[id(param)] = param
         
        # Load state for matching parameters
        for param_id, param_state in saved_state.items():
            if param_id in current_params:
                optimizer.state[current_params[param_id]] = param_state
                
        return optimizer

    def load_networks(self, exp, epoch, strict=True, restart_optim=False, load_shape_branch=True):
        print("Loading Checkpoint", os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)))
        ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)))
        diff_state_dict = {}
        diff_state_dict['opt'] = ckpt.pop('opt')
        if load_shape_branch:
            try:
                diff_state_dict['vqvae'] = ckpt.pop('vqvae')
                diff_state_dict['shape_df'] = ckpt.pop('shape_df')
                self.diff.ShapeDiff.vqvae.load_state_dict(diff_state_dict['vqvae'])
                self.diff.ShapeDiff.df.load_state_dict(diff_state_dict['shape_df'])
                self.diff.ShapeDiff.df_module = self.diff.ShapeDiff.df
                self.diff.ShapeDiff.vqvae_module = self.diff.ShapeDiff.vqvae
                print(colored(
                    '[*] shape branch has successfully been restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                              'model{}.pth'.format(
                                                                                                  epoch)), 'blue'))
            except:
                print('no vqvae or shape_df recorded. Assume it is only the layout branch')
        try:
            self.epoch = ckpt.pop('epoch')
            self.counter = ckpt.pop('counter')
        except:
            print('no epoch or counter recorded.')

        ckpt.pop('vqvae', None)
        ckpt.pop('shape_df', None)
        result = self.diff.load_state_dict(ckpt, strict=strict) # layout branch only
        
        if not strict:
            # Print missing and unexpected keys
            missing_keys = result.missing_keys
            unexpected_keys = result.unexpected_keys
            if missing_keys:
                print("Missing keys (expected by the model but not found in the checkpoint):")
                for key in missing_keys:
                    print(f"  - {key}")
            if unexpected_keys:
                print("\nUnexpected keys (found in the checkpoint but not used by the model):")
                for key in unexpected_keys:
                    print(f"  - {key}")
        
        print(colored('[*] GCN and layout branch has successfully been restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                    'model{}.pth'.format(epoch)),
                      'blue'))

        if not restart_optim:
            # print(diff_state_dict['opt'])
            print(self.counter, "<--- self.counter")
            import torch.optim as optim
            # state_dict_result = self.diff.optimizerFULL.load_state_dict(diff_state_dict['opt'])
            self.diff.optimizerFULL = self.load_optimizer_state_dict_non_strict(optimizer=self.diff.optimizerFULL, state_dict=diff_state_dict['opt'])

            self.diff.scheduler = optim.lr_scheduler.LambdaLR(self.diff.optimizerFULL, lr_lambda=self.diff.lr_lambda,
                                                    last_epoch=int(self.counter - 1))


    def sample_box_and_shape(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=False):
        print("sample_box_and_shape", self.type_ )
        layout_dict = self.diff.sampleBoxes(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat)
        if self.type_ == 'echolayout':
            layout_dict = self.diff.sampleBoxes(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat)
            return layout_dict
        elif self.type_ == 'echoscene':
            shape_dict, layout_dict = self.diff.sample(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=gen_shape)
            return {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def sample_boxes_and_shape_with_changes(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, manipulated_nodes, gen_shape=False):
        if self.type_ == 'echolayout':
            keep, layout_dict = self.diff.sampleBoxes_with_changes(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                                             dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, manipulated_nodes)
            return keep, layout_dict
        elif self.type_ == 'echoscene':
            keep, shape_dict, layout_dict = self.diff.sample_with_changes(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, manipulated_nodes, gen_shape=gen_shape)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def sample_boxes_and_shape_with_additions(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                            dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, missing_nodes, gen_shape=False):
        if self.type_ == 'echolayout':
            keep, layout_dict = self.diff.sampleBoxes_with_additions(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                                             dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, missing_nodes)
            return layout_dict
        elif self.type_ == 'echoscene':
            keep, shape_dict, layout_dict = self.diff.sample_with_additions(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                                                    dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, missing_nodes, gen_shape=gen_shape)
            return keep, {**shape_dict, **layout_dict}
        else:
            raise NotImplementedError

    def save(self, exp, outf, epoch, counter=None):
        if self.type_ == 'echolayout':
            torch.save(self.diff.state_dict(epoch, counter), os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
        elif self.type_ == 'echoscene':
            torch.save(self.diff.state_dict(epoch, counter), os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
        else:
            raise NotImplementedError
