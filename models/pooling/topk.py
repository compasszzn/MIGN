"""
    Maximal k-Independent Set (k-MIS) Pooling operator from
    `"Generalizing Downsampling from Regular Data to Graphs"
    <https://arxiv.org/abs/2208.03523>`_.

    Code adapted from https://github.com/pyg-team/pytorch_geometric/pull/6488
    by Francesco Landolfi (https://github.com/flandolfi)
"""

from typing import Callable, Optional, Tuple, Union

import torch
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor
from torch_geometric.utils import scatter
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from .aggr_pool import (AggrReduce, AggrLift, AggrConnect,
                        ReductionType, ConnectionType)
from .base import Select, Pooling, PoolingOutput, SelectOutput
from .utils import (connectivity_to_row_col,
                    connectivity_to_edge_index,
                    connectivity_to_adj_t, broadcast_shape)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import cumsum, scatter, softmax
from torch_geometric.nn.inits import uniform
Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]
def maximal_independent_set(edge_index: Adj, k: int = 1,
                            perm: OptTensor = None,
                            num_nodes: int = None) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: :class:`ByteTensor`
    """
    n = maybe_num_nodes(edge_index.size(0), num_nodes)

    row, col = connectivity_to_row_col(edge_index)
    device = row.device

    # v2: ADD SELF-LOOPS
    row, col = add_remaining_self_loops(torch.stack([row, col]))[0]

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_rank = scatter(min_rank[col], row, dim_size=n, reduce='min')

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(k):
            mask = scatter(mask[row], col, dim_size=n, reduce='max')

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis


def maximal_independent_set_cluster(edge_index: Adj, k: int = 1,
                                    perm: OptTensor = None,
                                    num_nodes: int = None) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    n = maybe_num_nodes(edge_index, num_nodes)
    mis = maximal_independent_set(edge_index=edge_index, k=k, perm=perm,
                                  num_nodes=n)
    device = mis.device

    row, col = connectivity_to_row_col(edge_index)

    # v2: ADD SELF-LOOPS
    row, col = add_remaining_self_loops(torch.stack([row, col]))[0]

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_rank = scatter(min_rank[row], col, dim_size=n, reduce='min')

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)

    # return mis, perm[clusters]

    _, clusters = torch.unique(perm[clusters], return_inverse=True)
    return mis, clusters

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")

class TOPKSelect(Select):
    r"""Selects the top-:math:`k` nodes with highest projection scores from the
    `"Graph U-Nets" <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
            \mathbf{p} \|} \right)

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

    where :math:`\mathbf{p}` is the learnable projection vector.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): The graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        act (str or callable, optional): The non-linearity :math:`\sigma`.
            (default: :obj:`"tanh"`)
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.act = activation_resolver(act)

        self.weight = torch.nn.Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        return_score: bool = False,
    ) -> SelectOutput:
        """"""  # noqa: D419
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = x.view(-1, 1) if x.dim() == 1 else x
        score = (x * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.act(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)

        return node_index, node_index, score

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f'ratio={self.ratio}'
        else:
            arg = f'min_score={self.min_score}'
        return f'{self.__class__.__name__}({self.in_channels}, {arg})'

class TOPKPooling(Pooling):
    r"""Maximal :math:`k`-Independent Set (:math:`k`-MIS) pooling operator
    from `"Generalizing Downsampling from Regular Data to Graphs"
    <https://arxiv.org/abs/2208.03523>`_.
    Args:
        in_channels (int, optional): Size of each input sample. Ignored if
            :obj:`scorer` is not :obj:`"linear"`.
        k (int): The :math:`k` value (defaults to 1).
        scorer (str or Callable): A function that computes a score for every
            node. Nodes with higher score will have higher chances to be
            selected by the pooling algorithm. It can be one of the following:
            - :obj:`"linear"` (default): uses a sigmoid-activated linear
              layer to compute the scores
                .. note::
                    :obj:`in_channels` and :obj:`score_passthrough`
                    must be set when using this option.
            - :obj:`"random"`: assigns a score to every node extracted
              uniformly at random from 0 to 1;
            - :obj:`"constant"`: assigns :math:`1` to every node;
            - :obj:`"canonical"`: assigns :math:`-i` to every :math:`i`-th
              node;
            - :obj:`"first"` (or :obj:`"last"`): use the first (resp. last)
              feature dimension of :math:`\mathbf{X}` as node scores;
            - A custom function having as arguments
              :obj:`(x, edge_index, edge_attr, batch)`. It must return a
              one-dimensional :class:`FloatTensor`.
        score_heuristic (str, optional): Apply one of the following heuristic
            to increase the total score of the selected nodes. Given an
            initial score vector :math:`\mathbf{s} \in \mathbb{R}^n`,
            - :obj:`None`: no heuristic is applied;
            - :obj:`"greedy"` (default): compute the updated score
              :math:`\mathbf{s}'` as
                .. math::
                    \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} +
                    \mathbf{I})^k\mathbf{1},
              where :math:`\oslash` is the element-wise division;
            - :obj:`"w-greedy"`: compute the updated score
              :math:`\mathbf{s}'` as
                .. math::
                    \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} +
                    \mathbf{I})^k\mathbf{s},
              where :math:`\oslash` is the element-wise division. All scores
              must be strictly positive when using this option.
        score_passthrough (str, optional): Whether to aggregate the node
            scores to the feature vectors, using the function specified by
            :obj:`aggr_score`. If :obj:`"before"`, all the node scores are
            aggregated to their respective feature vector before the cluster
            aggregation. If :obj:`"after"`, the score of the selected nodes are
            aggregated to the feature vectors after the cluster aggregation.
            If :obj:`None`, the score is not aggregated. Defaults to
            :obj:`"before"`.
                .. note::
                    Set this option either to :obj:`"before"` or :obj:`"after"`
                    whenever :obj:`scorer` is :obj:`"linear"` or a
                    :class:`torch.nn.Module`, to make the scoring function
                    end-to-end differentiable.
        reduce (str or Aggregation, optional): The aggregation function to be
            applied to the nodes in the same cluster. Can be any string
            admitted by :obj:`scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`) or any :class:`Aggregation`.
        connect (str): The aggregation function to be applied to the edges
            crossing the same two clusters. Can be any string
            admitted by :obj:`scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`). Defaults to :obj:`'sum'`.
        remove_self_loops (bool): Whether to remove the self-loops from the
            graph after its reduction. Defaults to :obj:`True`.
    """
    _passthroughs = {None, 'before', 'after'}

    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = "constant",
                 score_heuristic: Optional[str] = "greedy",
                 score_passthrough: Optional[str] = "before",
                 reduce: ReductionType = "sum",
                 connect: ConnectionType = "sum",
                 remove_self_loops: bool = True,
                 force_undirected: bool = False) -> None:
        select = TOPKSelect(in_channels=in_channels,
                            ratio=0.5,)

        assert score_passthrough in self._passthroughs, \
            "Unrecognized `score_passthrough` value."

        self.score_passthrough = score_passthrough

        if scorer == 'linear':
            assert self.score_passthrough is not None, \
                "`'score_passthrough'` must not be `None`" \
                " when using `'linear'` scorer"

        if reduce is not None:
            reduce = AggrReduce(operation=reduce)
        lift = AggrLift()
        connect = AggrConnect(reduce=connect,
                              remove_self_loops=remove_self_loops)

        super(TOPKPooling, self).__init__(selector=select,
                                          reducer=reduce,
                                          lifter=lift,
                                          connector=connect)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None) -> PoolingOutput:
        """"""
        # Select
        cluster, mis, score = self.select(edge_index, edge_attr, x,
                                          batch=batch, num_nodes=num_nodes,
                                          return_score=True)

        # Reduce
        if self.score_passthrough == 'before':
            score = broadcast_shape(score, x.size(), dim=self.node_dim)
            x = x * score

        num_clusters = mis.size(0)

        if self.reducer is None:
            x = torch.index_select(x, dim=self.node_dim, index=mis)
        else:
            x = self.reduce(x, cluster, dim_size=num_clusters,
                            dim=self.node_dim)

        if self.score_passthrough == 'after':
            score = broadcast_shape(score[mis], x.size(), dim=self.node_dim)
            x = x * score

        # Connect
        edge_index, edge_attr = self.connect(cluster, edge_index, edge_attr,
                                             batch=batch)

        if batch is not None:
            batch = batch[mis]

        return PoolingOutput(x, edge_index, edge_attr, batch)

    def __repr__(self):
        if self.scorer == 'linear':
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""

        return f'{self.__class__.__name__}({channels}k={self.k})'
