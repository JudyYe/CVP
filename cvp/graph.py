# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

import torch
import torch.nn as nn
from .layers import build_pre_act

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=2)


class GraphEdgeConv(nn.Module):
    """
    Single Layer of graph conv: node -> edge -> node
    """
    def __init__(self, input_dim, output_dim=None, edge_dim=None,
                 pooling='avg', preact_normalization='batch'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if edge_dim is None:
            edge_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim

        assert pooling in ['sum', 'avg', 'softmax'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling

        self.net_node2edge = build_pre_act(2 * input_dim, edge_dim, batch_norm=preact_normalization)
        self.net_edge2node = build_pre_act(edge_dim, output_dim, batch_norm=preact_normalization)
        self.net_node2edge.apply(_init_weights)
        self.net_edge2node.apply(_init_weights)

    def forward(self, obj_vecs, edges):
        """
        Inputs:
          + obj_vecs: (V, o, D)
          + edges: (V, t, 3) (s, w, o)

        Outputs:
          + new_obj_vecs: (V, o, D)

        Alg:
          relu(AXW), new_AX = AX, mlp = relu(new_AX, W)
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        V = obj_vecs.size(0)
        O = obj_vecs.size(1)
        E = edges.size(1)

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, :, 0].contiguous().type(torch.LongTensor).to(device)
        o_idx = edges[:, :, 2].contiguous().type(torch.LongTensor).to(device)

        s_node_idx = s_idx.unsqueeze(2).expand(V, E, self.input_dim)
        o_node_idx = o_idx.unsqueeze(2).expand(V, E, self.input_dim)
        s_edge_idx = s_idx.unsqueeze(2).expand(V, E, self.edge_dim)

        # Node -> Edge
        src_obj = torch.gather(obj_vecs, 1, s_node_idx)
        dst_obj = torch.gather(obj_vecs, 1, o_node_idx)
        node_obj = torch.cat([src_obj, dst_obj], dim=-1).view(-1, 2 * self.input_dim)

        edge_obj = self.net_node2edge(node_obj)
        edge_obj = edge_obj.view(V, E, self.edge_dim) # V, E, D

        # Edge - > Node

        # Pooling (AX) in triplet form.
        # for t in 1... T, x_new[s[t]] += w[t] * x_new[o[t]]
        # x_new = normlize over w.
        # we use scatter_add trick here
        if self.pooling == 'avg':
            # todo: correctness check
            pool_obj_vecs = torch.zeros(obj_vecs.size(), dtype=dtype, device=device)
            pool_obj_vecs = pool_obj_vecs.scatter_add(1, s_edge_idx, edge_obj)

            w_cnt = torch.zeros([V, O, self.edge_dim], dtype=dtype, device=device)
            ones = torch.ones([V, E, self.edge_dim], dtype=dtype, device=device)
            w_cnt = w_cnt.scatter_add(1, s_edge_idx, ones)  # add to s_idx
            w_cnt = w_cnt.clamp(min=1)
            pool_obj_vecs = pool_obj_vecs / w_cnt

        else:
            raise NotImplementedError
        # maybe add in instance normalization  here?
        # relu(BN(Pooled)) * W
        pool_obj_vecs = pool_obj_vecs.view(-1, self.input_dim)
        new_obj_vecs = self.net_edge2node(pool_obj_vecs)
        new_obj_vecs = new_obj_vecs.view(V, O, self.output_dim)

        return new_obj_vecs


class NoEdgeConv(nn.Module):
    """
    Single Layer of graph conv: node -> edge -> node
    """
    def __init__(self, input_dim, output_dim=None, edge_dim=None,
                 pooling='avg', preact_normalization='batch'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if edge_dim is None:
            edge_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim

        assert pooling in ['sum', 'avg', 'softmax'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling

        self.net_null_node2edge = build_pre_act(input_dim, edge_dim, batch_norm=preact_normalization)
        self.net_node2node = build_pre_act(edge_dim, output_dim, batch_norm=preact_normalization)
        self.net_null_node2edge.apply(_init_weights)
        self.net_node2node.apply(_init_weights)

    def forward(self, obj_vecs, edges):
        """
        Inputs:
          + obj_vecs: (V, o, D)
          + edges: (V, t, 3) (s, w, o), we do not use edge in this unit type

        Outputs:
          + new_obj_vecs: (V, o, D)

        Alg:
          relu(AXW), new_AX = AX, mlp = relu(new_AX, W)
        """
        V = obj_vecs.size(0)
        O = obj_vecs.size(1)

        obj_vecs = obj_vecs.view(V * O, self.input_dim)
        new_obj_vecs = self.net_null_node2edge(obj_vecs)
        new_obj_vecs = self.net_node2node(new_obj_vecs)
        new_obj_vecs = new_obj_vecs.view(V, O, self.output_dim)
        return new_obj_vecs


class GraphResBlock(nn.Module):
    """ A residual block of 2 Graph Conv Layer with one skip conection"""

    def __init__(self, input_dim, output_dim, unit_type, num_units=2, pooling='avg', preact_normalization='batch'):
        super().__init__()
        self.num_units = num_units
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'pooling': pooling,
            'preact_normalization': preact_normalization,
        }
        if unit_type == 'n2e2n':
            GraphUnit = GraphEdgeConv
        elif unit_type == 'noEdge':
            GraphUnit = NoEdgeConv
        else:
            raise NotImplementedError
        for n in range(self.num_units):
            if n == self.num_units - 1:
                gconv_kwargs['output_dim'] = output_dim
            else:
                gconv_kwargs['output_dim'] = input_dim
            self.gconvs.append(GraphUnit(**gconv_kwargs))

    def forward(self, input, edges):
        obj_vecs = input
        for i in range(self.num_units):
            gconv = self.gconvs[i]
            obj_vecs = gconv(obj_vecs, edges)
        if input.size(-1) > obj_vecs.size(-1):
            previous, _ = torch.split(input, [obj_vecs.size(-1), input.size(-1) - obj_vecs.size(-1)], -1)
            output = obj_vecs + previous
        elif input.size(-1) == obj_vecs.size(-1):
            output = obj_vecs + input
        else:
            raise NotImplementedError
        return output


class BypassFactorGCNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, feat_dim_list, noise_dim, num_blocks=4, num_units=2,
                 pooling='avg', preact_normalization='batch', unit_type='n2e2n', spatial=1, stop_grad=True):
        super().__init__()
        self.spatial = spatial
        pose_dim, bbox_dim = feat_dim_list
        dim_layers = [pose_dim * spatial * spatial + bbox_dim + noise_dim] * num_blocks \
                     + [pose_dim * spatial * spatial + bbox_dim]
        self.pose_off = pose_dim * spatial * spatial
        self.bbox_off = pose_dim * spatial * spatial + bbox_dim

        self.num_layers = len(dim_layers) - 1
        self.gblocks = nn.ModuleList()

        self.stop_grad = stop_grad
        self.unit_type = unit_type
        for n in range(self.num_layers):
            gblock_kwargs = {
                'input_dim': dim_layers[n],
                'output_dim': dim_layers[n + 1],
                'num_units': num_units,
                'pooling': pooling,
                'preact_normalization': preact_normalization,
                'unit_type': unit_type
            }
            self.gblocks.append(GraphResBlock(**gblock_kwargs))

    def forward(self, pose, bbox, diff_z, edges, stop_grad=None):
        """
        :param pose: (V, O, Cp, H, W)
        :param bbox: (V, O, 2)
        :param diff_z: (V, 1, D)
        :param edges:
        :return:
        """
        out = {}

        if stop_grad and self.stop_grad:
            pose = pose.clone()
            pose = pose.detach()

        V, _, D = diff_z.size()
        O = pose.size(1)
        pose = pose.view(V, O, -1)
        diff_z = diff_z.expand(V, O, D)
        obj_vecs = torch.cat([pose, bbox, diff_z], dim=-1)
        edges = self.build_graph(edges, bbox)
        for i in range(self.num_layers):
            net = self.gblocks[i]
            obj_vecs = net(obj_vecs, edges)
        out['appr'] = obj_vecs[:, :, 0: self.pose_off].view(V, O, -1, self.spatial, self.spatial)
        out['bbox'] = obj_vecs[:, :, self.pose_off: self.bbox_off]
        return out

    def build_graph(self, edges, bbox):
        """
        :param edges:
        :param bbox: V, O, 2
        :return:
        """
        if self.unit_type in ['n2e2n', 'noEdge']:
            return edges
        else:
            raise NotImplementedError


class FcResBlock(nn.Module):
    """ A residual block of 2 Fc Layer with one skip conection"""

    def __init__(self, input_dim, output_dim, num_units=2, activation='relu', preact_normalization='batch', dropout=0):
        super().__init__()

        self.num_units = num_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        cur_in = input_dim
        cur_out = output_dim
        # build a stack of fc
        for n in range(self.num_units):
            if n > 0:
                cur_in = output_dim
            if preact_normalization == 'batch':
                layers.append(nn.BatchNorm1d(output_dim))
            elif preact_normalization == 'instance':
                layers.append(nn.InstanceNorm1d(output_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'lekayrelu':
                layers.append(nn.LeakyReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(cur_in, cur_out))
        self.net = nn.Sequential(*layers)

    def forward(self, intput):
        V = intput.size(0)
        O = intput.size(1)
        obj_vecs = self.net(intput.view(-1, self.input_dim))
        obj_vecs = obj_vecs.view(V, O, self.output_dim)
        if intput.size(-1) > obj_vecs.size(-1):
            previous, _ = torch.split(intput, [obj_vecs.size(-1), intput.size(-1) - obj_vecs.size(-1)], -1)
            output = obj_vecs + previous
        elif intput.size(-1) == obj_vecs.size(-1):
            output = obj_vecs + intput
        else:
            raise NotImplementedError
        return output


class BypassFactorFCNet(nn.Module):
    """ A sequence of scene fc layers  """

    def __init__(self, feat_dim_list, noise_dim, num_blocks=4, num_units=2,
                 pooling='avg', preact_normalization='batch', unit_type='none', spatial=1, stop_grad=True):
        super().__init__()
        self.spatial = spatial
        self.stop_grad = stop_grad

        pose_dim, bbox_dim = feat_dim_list
        dim_layers = [pose_dim * spatial * spatial + bbox_dim + noise_dim] * num_blocks \
                     + [pose_dim * spatial * spatial + bbox_dim]
        self.pose_off = pose_dim * spatial * spatial
        self.bbox_off = pose_dim * spatial * spatial + bbox_dim

        self.num_layers = len(dim_layers) - 1
        self.fblocks = nn.ModuleList()
        for n in range(self.num_layers):
            fblock_kwargs = {
                'input_dim': dim_layers[n],
                'output_dim': dim_layers[n + 1],
                'num_units': num_units,
                'preact_normalization': preact_normalization,
            }
            self.fblocks.append(FcResBlock(**fblock_kwargs))

    def forward(self, pose, bbox, diff_z, edges, stop_grad=None):
        """
        :param pose: (V, O, Cp, H, W)
        :param bbox: (V, O, 2)
        :param diff_z: (V, 1, D)
        :param edges:
        :return:
        """
        out = {}

        if stop_grad and self.stop_grad:
            pose = pose.clone()
            pose = pose.detach()

        V, _, D = diff_z.size()
        O = pose.size(1)
        assert O == 1, 'spatial facotrized??' + str(pose.size())
        bbox_O = bbox.size(1)
        pose = pose.view(V, O, -1)
        bbox = bbox.view(V, O, -1)
        diff_z = diff_z.expand(V, O, D)
        obj_vecs = torch.cat([pose, bbox, diff_z], dim=-1)

        for i in range(self.num_layers):
            net = self.fblocks[i]
            obj_vecs = net(obj_vecs)
        out['appr'] = obj_vecs[:, :, 0: self.pose_off].view(V, O, -1, self.spatial, self.spatial)
        out['bbox'] = obj_vecs[:, :, self.pose_off: self.bbox_off].view(V, bbox_O, -1)
        return out

GraphFactory = {
    'fact_gc': BypassFactorGCNet,
    'fact_fc': BypassFactorFCNet,
}


if __name__ == '__main__':
    # test gconv
    a = torch.FloatTensor([[0, -1, 2], [-2, 3, 1], [5, -4, 2]])
    W = torch.ones([2, 2])
    adj = torch.FloatTensor([[0, 1, 0], [2. / 3, 1. / 3, 0], [1. / 3, 1. / 3, 1. / 3]])
    gt = torch.matmul(adj, a)
    # gt = torch.matmul(gt, W)
    edges = torch.FloatTensor([[0, 1, 1], [1, 2, 0], [1, 1, 1], [2, 1, 0], [2, 1, 1], [2, 1, 2]])

    print(gt)
    gconv = BypassFactorGCNet([3, 2], unit_type='n2e2n')
    # gconv = BypassFactorGCNet([3, 2], unit_type='n2e2n')
    print(gconv)
    aa = torch.stack([a, a], dim=0)
    WW = torch.stack([W, W], dim=0)
    ee = torch.stack([edges, edges], dim=0)
    print(aa.size())
    gout = gconv(aa, ee)
