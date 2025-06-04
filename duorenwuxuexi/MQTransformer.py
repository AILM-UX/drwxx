import torch
import torch.nn as nn
import warnings
import math

from thop import profile
from torch.nn import functional as F
from torch.distributions import Normal

class swin_transformer_s_t_ada(nn.Module):
    def __init__(self, img_size=[224], tactile_size=[224], input_image_patch=49, input_tac_patch=16, embed_dim=384, depth1=1,depth2=4,depth3=4,depth4=2,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.05, attn_drop_rate=0.05,
                 drop_path_rate=0.05, norm_layer=nn.LayerNorm,attention_axes=(1,2),attention_bottle=15, input_size=(8,384),batch_size=8, initial_rate=0.2,**kwargs):
        super().__init__()
        self.patch_embed1 = PatchEmbed_v()
        self.patch_embed2 = PatchEmbed_t()
        self.batch_size=batch_size
        self.attention_axes=attention_axes
        self.w1=nn.Parameter(torch.ones(1))
        self.w2=nn.Parameter(torch.ones(1))
        self.w3=nn.Parameter(torch.ones(1))
        self.cv1=1
        self.cv2=1
        self.cv3=1
        self.cls_v1_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v2_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v3_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v4_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v5_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v6_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v7_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_v8_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t1_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t2_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t3_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t4_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t5_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t6_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t7_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t8_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed1_v = nn.Parameter(torch.zeros(1, input_image_patch +1+attention_bottle, embed_dim))
        self.pos_embed1_t = nn.Parameter(torch.zeros(1, input_tac_patch +1+attention_bottle, embed_dim))
        self.q1 = nn.Parameter(torch.zeros(1, 49, embed_dim),requires_grad=False)
        self.q2 = nn.Parameter(torch.zeros(1, 49, embed_dim),requires_grad=False)
        self.q3 = nn.Parameter(torch.zeros(1, 49, embed_dim),requires_grad=False)

        self.cls_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_embed= nn.Parameter(torch.zeros(1, input_image_patch + 1 + attention_bottle+input_tac_patch +1+attention_bottle, embed_dim))
        self.FSN = nn.Parameter(torch.zeros(1, attention_bottle, embed_dim))
        self.depth3=depth3
        self.depth2=depth2
        self.attention_bottle = attention_bottle
        #nn.Parameter 是一种特殊的 Tensor，它会被自动注册为模型的参数，并且可以在模型的优化过程中进行更新。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth1)]
        self.swin_1=SwinTransformer(depths=(1,1,2))
        self.swin_2=SwinTransformer(depths=(1,1,2))

        #生成一个列表 dpr，其中的每个元素都是从 0 到 drop_path_rate（即深度）之间均匀间隔的值。
        #dpr 列表将被用于模型的不同层中，用于控制 drop path 的概率。
        self.blocks_v0 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth1)])
        self.blocks_t0 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth1)])
        self.blocks_v1 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth1)])
        self.blocks_v2 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth1)])
        self.blocks_t1 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth1)])
        self.blocks_t2 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth1)])
        self.att0_1 = Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.att0_2 = Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.att0_3 = Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.att1_1=Block2(dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.att1_2 = Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.att2 = Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
                             norm_layer=norm_layer)
        self.att3_1=Block2(dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.att3_2 = Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.MLP_P=nn.Linear(embed_dim, embed_dim)

        #构建一个多层的注意力模块堆叠，每个层都是一个 Block 模块，用于对输入数据进行处理、特征提取和表示学习。整个模型的深度由 depth 参数指定。
        self.norm = norm_layer(embed_dim)
        # self.adaptive_dropout_v=AdaptiveDropout(input_size=(batch,8,embed_dim),initial_rate=0.2)
        # self.adaptive_dropout_t=AdaptiveDropout(input_size=(batch,8,embed_dim),initial_rate=0.2)
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate, depth4)]
        self.fusion = nn.ModuleList([
            Block1(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr1[i], norm_layer=norm_layer)
            for i in range(depth4)])
        self.MLP1 = nn.Sequential(nn.Linear((49)*embed_dim, 640),
                                            nn.GELU(),
                                            #将输入向量的维度从 (img_patches + self.patch_embed.tactile_patch + 2)*embed_dim//12 降低到 640。
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288),
                                            nn.GELU(),
                                          nn.Linear(288, 29))#将特征直接分类出结果
        self.MLP2 = nn.Sequential(nn.Linear((49)*embed_dim, 640),
                                            nn.GELU(),
                                            #将输入向量的维度从 (img_patches + self.patch_embed.tactile_patch + 2)*embed_dim//12 降低到 640。
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288),
                                            nn.GELU(),
                                          nn.Linear(288, 13))#将特征直接分类出结果
        self.MLP3 = nn.Sequential(nn.Linear((49)*embed_dim, 640),
                                            nn.GELU(),
                                            #将输入向量的维度从 (img_patches + self.patch_embed.tactile_patch + 2)*embed_dim//12 降低到 640。
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288),
                                            nn.GELU(),
                                          nn.Linear(288, 17))
        self.align_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                               nn.Sigmoid())

        self.contact_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                                 nn.Sigmoid())
        # self.MLP = nn.Sequential(nn.Linear(67*8*384, 20000), nn.Linear(20000, 1000))

        self.embed_dim=embed_dim

        trunc_normal_(self.q1, std=.02)
        trunc_normal_(self.q2, std=.02)
        trunc_normal_(self.pos_embed1_v, std=.02)
        trunc_normal_(self.pos_embed1_t, std=.02)
        trunc_normal_(self.cls_v1_embed, std=.02)
        trunc_normal_(self.cls_v2_embed, std=.02)
        trunc_normal_(self.cls_v3_embed, std=.02)
        trunc_normal_(self.cls_v4_embed, std=.02)
        trunc_normal_(self.cls_v5_embed, std=.02)
        trunc_normal_(self.cls_v6_embed, std=.02)
        trunc_normal_(self.cls_v7_embed, std=.02)
        trunc_normal_(self.cls_v8_embed, std=.02)
        trunc_normal_(self.cls_t1_embed, std=.02)
        trunc_normal_(self.cls_t2_embed, std=.02)
        trunc_normal_(self.cls_t3_embed, std=.02)
        trunc_normal_(self.cls_t4_embed, std=.02)
        trunc_normal_(self.cls_t5_embed, std=.02)
        trunc_normal_(self.cls_t6_embed, std=.02)
        trunc_normal_(self.cls_t7_embed, std=.02)
        trunc_normal_(self.cls_t8_embed, std=.02)
        trunc_normal_(self.FSN, std=.02)
        #对模型中的位置编码参数进行截断正态分布初始化。
    def interpolate_pos_encoding1_v(self, x):
        npatch = x.shape[2] - 1
        N = self.pos_embed1_v.shape[1] - 1
        if npatch == N:
            return self.pos_embed1_v
        else:
            raise ValueError('Position Encoder does not match dimension')

    def interpolate_pos_encoding1_t(self, x):
        npatch = x.shape[2] - 1
        N = self.pos_embed1_t.shape[1] - 1
        if npatch == N:
            return self.pos_embed1_t
        else:
            raise ValueError('Position Encoder does not match dimension')
    def prepare_tokens(self, y1,y2,y3,y4,y5,y6,y7,y8,x1, x2,x3,x4,x5,x6,x7,x8):
        B,HW,C= y1.shape
        S=1
        y1, y2, y3, y4, y5, y6, y7, y8 = self.patch_embed1(y1), self.patch_embed1(y2), self.patch_embed1(
            y3), self.patch_embed1(y4), self.patch_embed1(y5), self.patch_embed1(y6), self.patch_embed1(
            y7), self.patch_embed1(y8)
        x1, x2, x3, x4, x5, x6, x7, x8 = self.patch_embed2(x1), self.patch_embed2(x2), self.patch_embed2(
            x3), self.patch_embed2(x4), self.patch_embed2(x5), self.patch_embed2(x6), self.patch_embed2(
            x7), self.patch_embed2(x8)

        cls_v1_embed = self.cls_v1_embed.expand(B, S, -1, -1)
        cls_v2_embed = self.cls_v2_embed.expand(B, S, -1, -1)
        cls_v3_embed = self.cls_v3_embed.expand(B, S, -1, -1)
        cls_v4_embed = self.cls_v4_embed.expand(B, S, -1, -1)
        cls_v5_embed = self.cls_v5_embed.expand(B, S, -1, -1)
        cls_v6_embed = self.cls_v6_embed.expand(B, S, -1, -1)
        cls_v7_embed = self.cls_v7_embed.expand(B, S, -1, -1)
        cls_v8_embed = self.cls_v8_embed.expand(B, S, -1, -1)
        cls_t1_embed = self.cls_t1_embed.expand(B, S, -1, -1)
        cls_t2_embed = self.cls_t2_embed.expand(B, S, -1, -1)
        cls_t3_embed = self.cls_t3_embed.expand(B, S, -1, -1)
        cls_t4_embed = self.cls_t4_embed.expand(B, S, -1, -1)
        cls_t5_embed = self.cls_t5_embed.expand(B, S, -1, -1)
        cls_t6_embed = self.cls_t6_embed.expand(B, S, -1, -1)
        cls_t7_embed = self.cls_t7_embed.expand(B, S, -1, -1)
        cls_t8_embed = self.cls_t8_embed.expand(B, S, -1, -1)
        FSN = self.FSN.expand(B, S, self.attention_bottle, -1)

        y1 = torch.cat((cls_v1_embed,y1,FSN),dim=2)
        y2 = torch.cat((cls_v2_embed,y2,FSN),dim=2)
        y3 = torch.cat((cls_v3_embed,y3,FSN),dim=2)
        y4= torch.cat((cls_v4_embed,y4,FSN),dim=2)
        y5 = torch.cat((cls_v5_embed,y5,FSN),dim=2)
        y6 = torch.cat((cls_v6_embed,y6,FSN),dim=2)
        y7 = torch.cat((cls_v7_embed,y7,FSN),dim=2)
        y8= torch.cat((cls_v8_embed,y8,FSN),dim=2)

        x1 = torch.cat((cls_t1_embed,x1,FSN),dim=2)
        x2 = torch.cat((cls_t2_embed, x2,FSN), dim=2)
        x3 = torch.cat((cls_t3_embed, x3,FSN), dim=2)
        x4 = torch.cat((cls_t4_embed, x4,FSN), dim=2)
        x5 = torch.cat((cls_t5_embed,x5,FSN),dim=2)
        x6 = torch.cat((cls_t6_embed, x6,FSN), dim=2)
        x7 = torch.cat((cls_t7_embed, x7,FSN), dim=2)
        x8 = torch.cat((cls_t8_embed, x8,FSN), dim=2)



        # introduce contact embedding & alignment embedding
        y1= y1 + self.interpolate_pos_encoding1_v(y1)
        y2 = y2 + self.interpolate_pos_encoding1_v(y2)
        y3 = y3 + self.interpolate_pos_encoding1_v(y3)
        y4 = y4 + self.interpolate_pos_encoding1_v(y4)
        y5= y5 + self.interpolate_pos_encoding1_v(y5)
        y6 = y6 + self.interpolate_pos_encoding1_v(y6)
        y7 = y7 + self.interpolate_pos_encoding1_v(y7)
        y8 = y8 + self.interpolate_pos_encoding1_v(y8)
        x1=x1+self.interpolate_pos_encoding1_t(x1)
        x2=x2+self.interpolate_pos_encoding1_t(x2)
        x3=x3+self.interpolate_pos_encoding1_t(x3)
        x4=x4+self.interpolate_pos_encoding1_t(x4)
        x5=x5+self.interpolate_pos_encoding1_t(x5)
        x6=x6+self.interpolate_pos_encoding1_t(x6)
        x7=x7+self.interpolate_pos_encoding1_t(x7)
        x8=x8+self.interpolate_pos_encoding1_t(x8)
        y = torch.cat((y1, y2, y3, y4,y5,y6,y7,y8), dim=1)
        x = torch.cat((x1, x2,x3,x4,x5,x6,x7,x8), dim=1)
        return y,x
    def cross_task(self,task1,task2,q1,q2):

        p1 = self.att1_1(task1, q1)
        p1 = p1 + self.norm(self.MLP_P(p1))
        p2 = self.att1_2(task2, q2)
        p2 = p2 + self.norm(self.MLP_P(p2))
        p = torch.cat((p1, p2), dim=1)
        p = self.att2(p, p)
        p1 = p[:, :49, :]
        p2 = p[:, 49:, :]
        p1 = self.att3_1(p1, task1)
        p2 = self.att3_2(p2, task2)
        return p1,p2

    def forward(self,x1, x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8):
        #x视觉 y触觉
        # y1,y2,y3,y4,y5,y6,y7,y8,x1, x2,x3,x4,x5,x6,x7,x8=self.swin(y1,y2,y3,y4,y5,y6,y7,y8,x1, x2,x3,x4,x5,x6,x7,x8)+
        x1, x2,x3,x4,x5,x6,x7,x8= self.swin_1(x1, x2,x3,x4,x5,x6,x7,x8)
        y1, y2, y3, y4, y5, y6, y7, y8= self.swin_2(y1,y2,y3,y4,y5,y6,y7,y8)
        v,t = self.prepare_tokens(x1, x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8)
        b, hw1, e = y1.size()
        y1 = y1.view(b,1,hw1,e)

        b, hw2, e = x1.size()
        x1 = x1.view(b,1,hw2,e)

        for k in range(self.depth2):

            for blk in self.blocks_v0:
                y1 = blk(y1)
            for blk in self.blocks_t0:
                x1 = blk(x1)
        y1 = y1.view(b,hw1,e)
        x1 = x1.view(b,hw2,e)

        for j in range(self.depth3):
            for k in self.attention_axes:
                if k == 1:
                    batch_size, height, width, channel = t.shape
                    t = t.transpose(2, 1).reshape(batch_size * width, 1, height,
                                                  channel)
                    for blk in self.blocks_t1:
                        t = blk(t)
                    t = t.reshape((batch_size, width, height, channel)).transpose(2, 1)
                    batch_size, height, width, channel = v.shape
                    v = v.transpose(2, 1).reshape(batch_size * width, 1, height,
                                                  channel)
                    for blk in self.blocks_v1:
                        v = blk(v)
                    v = v.reshape((batch_size, width, height, channel)).transpose(2, 1)


                else:
                    batch_size, height, width, channel = t.shape
                    t = t.reshape(batch_size * height, 1, width, channel)
                    for blk in self.blocks_t2:
                        t = blk(t)
                    t = t.reshape(batch_size, height, width, channel)

                    batch_size, height, width, channel = v.shape
                    v = v.reshape(batch_size * height, 1, width, channel)
                    for blk in self.blocks_v2:
                        v = blk(v)
                    v = v.reshape(batch_size, height, width, channel)
                temp1 = (v[:, 0, -self.attention_bottle:, :] + t[:, 0, -self.attention_bottle:, :])
                temp2 = (v[:, 1, -self.attention_bottle:, :] + t[:, 1, -self.attention_bottle:, :])
                temp3 = (v[:, 2, -self.attention_bottle:, :] + t[:, 2, -self.attention_bottle:, :])
                temp4 = (v[:, 3, -self.attention_bottle:, :] + t[:, 3, -self.attention_bottle:, :])
                temp5 = (v[:, 4, -self.attention_bottle:, :] + t[:, 4, -self.attention_bottle:, :])
                temp6 = (v[:, 5, -self.attention_bottle:, :] + t[:, 5, -self.attention_bottle:, :])
                temp7 = (v[:, 6, -self.attention_bottle:, :] + t[:, 6, -self.attention_bottle:, :])
                temp8 = (v[:, 7, -self.attention_bottle:, :] + t[:, 7, -self.attention_bottle:, :])

                temp1 = temp1 / (2)
                temp2 = temp2 / (2)
                temp3 = temp3 / (2)
                temp4 = temp4 / (2)
                temp5 = temp5 / (2)
                temp6 = temp6 / (2)
                temp7 = temp7 / (2)
                temp8 = temp8 / (2)

                v[:, 0, -self.attention_bottle:, :] = temp1
                v[:, 1, -self.attention_bottle:, :] = temp2
                v[:, 2, -self.attention_bottle:, :] = temp3
                v[:, 3, -self.attention_bottle:, :] = temp4
                v[:, 4, -self.attention_bottle:, :] = temp5
                v[:, 5, -self.attention_bottle:, :] = temp6
                v[:, 6, -self.attention_bottle:, :] = temp7
                v[:, 7, -self.attention_bottle:, :] = temp8
                t[:, 0, -self.attention_bottle:, :] = temp1
                t[:, 1, -self.attention_bottle:, :] = temp2
                t[:, 2, -self.attention_bottle:, :] = temp3
                t[:, 3, -self.attention_bottle:, :] = temp4
                t[:, 4, -self.attention_bottle:, :] = temp5
                t[:, 5, -self.attention_bottle:, :] = temp6
                t[:, 6, -self.attention_bottle:, :] = temp7
                t[:, 7, -self.attention_bottle:, :] = temp8
            # (1,2)time_spaec          (2,1)spaec_time
        v_t=torch.cat((v,t),dim=2)
        for blk in self.fusion:
            v_t = blk(v_t)
        a=v[:, 0, -self.attention_bottle:, :]

        v_cls=v[:, :, 0, :]

        t_cls=t[:, :, 0, :]
        v_t=torch.cat((v,t),dim=2)
        for blk in self.fusion:
            v_t = blk(v_t)
        B, S, N, C = v_t.shape
        cls_v=v_t[:, :,0, :].reshape(B,-1,self.embed_dim)
        cls_t=v_t[:, :,54, :].reshape(B,-1,self.embed_dim)
        f_v = x1[:, 0, :].reshape(B,  -1,self.embed_dim)

        f_t = y1[:, 0, :].reshape(B,  -1,self.embed_dim)
        img_tactile = torch.cat((v_cls,t_cls,cls_v,cls_t, a, f_v, f_t), dim=1)

        task1=self.att0_1(img_tactile,img_tactile)
        task2=self.att0_2(img_tactile,img_tactile)
        task3=self.att0_3(img_tactile,img_tactile)
        q1 = self.q1.expand(B, -1, -1)
        q2 = self.q2.expand(B, -1, -1)
        q3 = self.q3.expand(B, -1, -1)

        p1_0,p2_0=self.cross_task(task1,task2,q1,q2)
        p2_1,p3_0=self.cross_task(task2,task3,q2,q3)
        p3_1,p1_1=self.cross_task(task3,task1,q3,q1)

        p1 = (p1_0 + p1_1) / 2
        p2 = (p2_0 + p2_1) / 2
        p3 = (p3_0 + p3_1) / 2

        p1=p1.reshape(B,-1)
        p2=p2.reshape(B,-1)
        p3=p3.reshape(B,-1)

        task1=self.MLP1(p1)
        task2=self.MLP2(p2)
        task3=self.MLP3(p3)


        return task1,task2,task3


class Attention(nn.Module):#注意力机制
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, N, C = x.shape
        qkv = self.qkv(x).reshape(B*S, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class PatchEmbed_v(nn.Module):#视觉补丁
    def __init__(self, embeded_dim=384):
        super().__init__()
        self.embeded_dim = embeded_dim

    def forward(self, image):
        # Input shape batch, Sequence, in_Channels, H, W
        # Output shape batch, Sequence, patches & out_Channels
        B,HW, C = image.shape
        S=1

        pached_image = image.view(B, S, -1, self.embeded_dim)
        return pached_image

class PatchEmbed_t(nn.Module):#触觉补丁
    def __init__(self,embeded_dim=384):
        super().__init__()
        self.embeded_dim = embeded_dim

    def forward(self, tac):
        # Input shape batch, Sequence, in_Channels, H, W
        # Output shape batch, Sequence, patches & out_Channels
        B, HW, C = tac.shape
        S = 1
        pached_image = tac.view(B, S, -1, self.embeded_dim)
        return pached_image


class Block(nn.Module):#
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, return_attention: bool = False):
        #x维度为B*S*40*384
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Attention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.qkv3 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,y):
        B1, S1, N1, C1 = x.shape
        qkv1 = self.qkv1(x).reshape(B1*S1, N1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = qkv1[0], qkv1[1]
        B2, S2, N2, C2 = y.shape
        qkv2 = self.qkv2(y).reshape(B2 * S2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = qkv2[0], qkv2[1]
        x_y = torch.cat((x,y), dim=2)
        B3, S3, N3, C3 = x_y.shape
        q = self.qkv3(x_y).reshape(B3 * S3, N3, 1, self.num_heads, C3 // self.num_heads).permute(2, 0, 3, 1, 4)

        attn1 = (q @ k1.transpose(-2, -1))
        attn2 = (q @ k2.transpose(-2, -1))
        attn = torch.cat((attn1, attn2), dim=4)
        attn=attn * self.scale
        v = torch.cat((v1, v2), dim=2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B3, S3, N3, C3)
        attn = attn.view(B3, S3, -1, N3, N3)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class Block1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, attn_drop=0.2,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention1(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x_y, return_attention: bool = False):
        #x维度为B*S*40*384
        x=x_y[:,:,0:54,:]
        y=x_y[:,:,54:,:]
        a, attn = self.attn(self.norm1(x),self.norm1(y))
        if return_attention:
            return attn
        a=self.norm2(a)
        a=self.mlp(a)
        return a


class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,y):
        B,  N, C = x.shape
        S=1
        kv = self.kv(x).reshape(B*S, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(y).reshape(B * S, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class Block2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention2(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x,y, return_attention: bool = False):
        #x维度为B*49*384
        y, attn = self.attn(self.norm1(x),self.norm1(y))
        y=torch.squeeze(y, 1)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim - 1)
    #，shape 是一个元组，它的第一个维度大小与输入张量 x 的批次大小相同，而其余维度都是 1。
    # 这样，shape 的长度与输入张量 x 的维度数相同，但是在第一个维度之后的维度都被设置为 1，用于构建随机张量 random_tensor 的形状。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    #用于将张量中的每个元素取整，将每个元素变为不大于其原始值的最大整数。
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                            act_layer(),
                            nn.Linear(hidden_features, out_features))
    def forward(self, x):
        x = self.MLP(x)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

#从这开始都是swin transformer的代码
""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from thop import profile


def drop_path_f_s(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath_s(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath_s, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f_s(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed_s(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp_s(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        #这行代码的作用是将三个注意力机制的结果 qkv 沿着第一个维度（维度0）进行解绑定（unbind），将其分解成三个张量 q、k 和 v。
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath_s(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_s(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x1, x2,x3,x4,x5,x6,x7,x8, H, W):
        attn_mask = self.create_mask(x1, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1, attn_mask)
                x2 = checkpoint.checkpoint(blk, x2, attn_mask)
                x3 = checkpoint.checkpoint(blk, x3, attn_mask)
                x4 = checkpoint.checkpoint(blk, x4, attn_mask)
                x5 = checkpoint.checkpoint(blk, x5, attn_mask)
                x6 = checkpoint.checkpoint(blk, x6, attn_mask)
                x7 = checkpoint.checkpoint(blk, x7, attn_mask)
                x8 = checkpoint.checkpoint(blk, x8, attn_mask)

            else:
                x1 = blk(x1, attn_mask)
                x2 = blk(x2, attn_mask)
                x3 = blk(x3, attn_mask)
                x4 = blk(x4, attn_mask)
                x5 = blk(x5, attn_mask)
                x6 = blk(x6, attn_mask)
                x7 = blk(x7, attn_mask)
                x8 = blk(x8, attn_mask)


        if self.downsample is not None:
            x1 = self.downsample(x1, H, W)
            x2 = self.downsample(x2, H, W)
            x3 = self.downsample(x3, H, W)
            x4 = self.downsample(x4, H, W)
            x5 = self.downsample(x5, H, W)
            x6 = self.downsample(x6, H, W)
            x7 = self.downsample(x7, H, W)
            x8 = self.downsample(x8, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x1, x2,x3,x4,x5,x6,x7,x8, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=8, in_chans=3, num_classes=20,
                 embed_dim=96, depths=(1, 2 ), num_heads=(3, 6,12),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.patah_size=patch_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed_s(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1,x2,x3,x4,x5,x6,x7,x8):
        B,C,h,w=x1.shape


        # x: [B, L, C]
        x1, H, W = self.patch_embed(x1)
        x2, H, W = self.patch_embed(x2)
        x3, H, W = self.patch_embed(x3)
        x4, H, W = self.patch_embed(x4)
        x5, H, W = self.patch_embed(x5)
        x6, H, W = self.patch_embed(x6)
        x7, H, W = self.patch_embed(x7)
        x8, H, W = self.patch_embed(x8)
        x1,x2,x3,x4,x5,x6,x7,x8 = self.pos_drop(x1),self.pos_drop(x2),self.pos_drop(x3),self.pos_drop(x4),self.pos_drop(x5),self.pos_drop(x6),self.pos_drop(x7),self.pos_drop(x8)
        for layer in self.layers:
            x1,x2,x3,x4,x5,x6,x7,x8, H, W = layer(x1,x2,x3,x4,x5,x6,x7,x8, H, W)

        h=h//self.patah_size
        h=math.ceil(h/4)
        w = w // self.patah_size
        w = math.ceil(w/4)

        #8*7*7*384
        x1 = x1.view(B, h*w, 384)
        x2 = x2.view(B, h*w, 384)
        x3 = x3.view(B, h*w, 384)
        x4 = x4.view(B, h*w, 384)
        x5 = x5.view(B, h*w, 384)
        x6 = x6.view(B, h*w, 384)
        x7 = x7.view(B, h*w, 384)
        x8 = x8.view(B, h*w, 384)

        return  x1,x2,x3,x4,x5,x6,x7,x8

if __name__ == '__main__':
    input_size1 = (8, 3, 224, 224)
    input_size2 = (8, 3, 112, 112)

    model = swin_transformer_s_t_ada()
    x1 = torch.randn(input_size1)
    x2 = torch.randn(input_size2)

    # Calculate FLOPs
    flops, params = profile(model, inputs=(x1, x1, x1, x1,x1, x1, x1, x1, x2, x2, x2, x2, x2, x2, x2, x2))
    print(f"FLOPs: {flops / 8e9} G FLOPs")  # 打印FLOPs（十亿次浮点运算）
    print(f"Parameters: {params / 8e6} M parameters")  # 打印参数数量（百万个参数）
