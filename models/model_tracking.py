'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertModel

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class TrackingTransformer(nn.Module):
    def __init__(self,
                 tokenizer = None,
                 config = None,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        vision_width = config["vision_width"]
        max_words = config["max_words"]
        dpr = config["dropout"]
        self.max_words = max_words
        print("Model will generate %s points" % self.max_words)
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.scan_encoder = BertModel.from_pretrained("bert-base-uncased", config=bert_config, add_pooling_layer=False)

        self.dense = nn.Linear(vision_width, 3)
        self.dense2 = nn.Linear(vision_width, 3)


    def fix_params(self, model):
        for n, param in model.named_parameters():
            param.requires_grad = False

    def forward_step(self, image):
        seq_len = self.max_words
        image_embeds = self.visual_encoder(image)

        return image_embeds


    def forward(self, image, greedy=True):
        image_embeds = self.forward_step(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        coord = 0.5 * torch.ones(image.shape[0], 1, 2).to(image.device)
        coord_t = torch.zeros(image.shape[0], 1, 1).to(image.device)
        coord = torch.cat([coord, coord_t], -1)

        coord_mask = torch.ones(coord.size()[:-1], dtype=torch.long).to(image.device)
        past_key_values = None
        pred = coord

        pred_list = []
        pred_logprob_list = []

        for i in range(self.max_words-1):
            coord_output = self.scan_encoder(pred, attention_mask=coord_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True, mode='text',
                                             past_key_values=past_key_values,
                                             is_decoder=True)

            pred = self.dense(coord_output.last_hidden_state[:, -1:])

            ### Sample points and their log probs from Gaussian Dist
            if greedy is False:
                ### Get the variance, use exp to make sure that its value > 0
                logstd = self.dense2(coord_output.last_hidden_state[:, -1:])
                pred_var = torch.exp(logstd)

                ### Gaussian determined by the mean: pred and var: pred_var
                pred_dist = torch.distributions.Normal(pred, pred_var)

                pred = pred_dist.sample()
                pred_logprob = pred_dist.log_prob(pred)
                pred_logprob_list.append(pred_logprob)

            ### Make sure the coords are located btw 0 and 1
            xy_pred = F.sigmoid(pred[:, :, :2])
            t_pred = pred[:, :, 2:]
            pred = torch.cat([xy_pred, t_pred], -1)
            pred_list.append(pred)

            mask_token = torch.ones(pred.size()[:-1], dtype=torch.long).to(image.device)
            coord_mask = torch.cat([coord_mask, mask_token], 1)

            ### Save the queries, keys and values of the predicted points
            past_key_values = coord_output.past_key_values

        pred_list = torch.cat(pred_list, 1)

        if greedy:
            return pred_list
        else:
            pred_logprob_list = torch.cat(pred_logprob_list, 1)
            return pred_list, pred_logprob_list


    def inference(self, image):
        image_embeds = self.forward_step(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        coord = 0.5 * torch.ones(image.shape[0], 1, 2).to(image.device)
        coord_t = torch.zeros(image.shape[0], 1, 1).to(image.device)
        coord = torch.cat([coord, coord_t], -1)
        coord_mask = torch.ones(coord.size()[:-1], dtype=torch.long).to(image.device)
        past_key_values = None
        pred = coord

        for _ in range(self.max_words-1):
            coord_output = self.scan_encoder(pred, attention_mask=coord_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True, mode='text',
                                             past_key_values=past_key_values,
                                             is_decoder=True)
            pred = self.dense(coord_output.last_hidden_state[:, -1:])

            pred_xy = F.sigmoid(pred[:, :, :2])
            pred_t = pred[:, :, 2:]
            pred = torch.cat([pred_xy, pred_t], -1)

            coord = torch.cat([coord, pred], 1)
            mask_token = torch.ones(pred.size()[:-1], dtype=torch.long).to(image.device)
            coord_mask = torch.cat([coord_mask, mask_token], 1)
            past_key_values = coord_output.past_key_values

        return coord[:, 1:]




