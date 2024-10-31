import copy
import dataclasses
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx
from typing import Tuple, List, Optional

import vllm.envs as envs
from vllm.config import CompilationConfig
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

from .collective_fusion import CollectiveFusionPass
from .counter import compilation_counter
from .inductor_pass import InductorPass
from .pass_manager import PostGradPassManager

logger = init_logger(__name__)


FILENO=0


def pprint(x):
    #print(x)
    pass


# This check is a hack, copied from linear.py
def should_slice(shape) -> bool:
    n_slices = get_tensor_model_parallel_world_size()
    return (shape[0] % n_slices == 0 and shape[0] >= 128)


def match_gemm_rs_ag_gemm(residual,
                          #my_residual,
                          gemm_1_weights,
                          gemm_1_activations,
                          rms_norm_weight,
                          gemm_2_weights,
                          ):
    permute_2 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
    mm_1 = torch.ops.aten.mm.default(gemm_1_activations, permute_2)
    auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = 'tp:0')  # how to deal with groupname?
    getitem_25 = auto_functionalized_4[1]
    auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_25, residual = residual, weight = rms_norm_weight, epsilon = 1e-05)
    getitem_27 = auto_functionalized_5[1]
    getitem_28 = auto_functionalized_5[2]  # new residual
    permute_3 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
    mm_2 = torch.ops.aten.mm.default(getitem_27, permute_3)
    return mm_2, getitem_28


def slices(residual) -> List[torch.Tensor]:
    n_slices = get_tensor_model_parallel_world_size()
    residual_slices = torch.chunk(residual, n_slices, dim=0)
    #pprint(f"SLICES {[r.shape for r in residual_slices]}")
    return residual_slices


#schema_str="(Tensor(a) residual, Tensor(a) my_residual, Tensor gemm_1_weights, Tensor gemm_1_activations, Tensor rms_norm_weight, Tensor gemm_2_weights, bool first_layer) -> (Tensor, Tensor, Tensor)"

@torch.library.custom_op("vllm::gemm_rs_ag_gemm", mutates_args=())#, schema=schema_str)
def gemm_rs_ag_gemm(residual: torch.Tensor,
                    my_residual: torch.Tensor,
                    gemm_1_weights: torch.Tensor,
                    gemm_1_activations: torch.Tensor,
                    rms_norm_weight: torch.Tensor,
                    gemm_2_weights: torch.Tensor,
                    first_layer: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print(f"CUSTOM {residual.shape}({my_residual.shape}), should_slice={should_slice(residual.shape)}, first={first_layer}")

    # this is terrible
    if True:
        res_slices = slices(residual)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]
    else:
        slice_size = 2048
    print(f"SLICE_SIZE = {slice_size}, orig_shape={residual.shape}, slice_shapes=[{[x.shape for x in res_slices]}]")

    if should_slice(residual.shape) and first_layer:
        print(f"FIRST! rank={get_tensor_model_parallel_rank()}")
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        getitem_26 = split_1[0];  split_1 = None
    else:
        #getitem_26 = my_residual
        getitem_26 = residual
        slice_size = residual.shape[0]

    if not should_slice(residual.shape):
        # this branch probably broken
        print("NAIVE")
        permute_3 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        output = torch.matmul(gemm_1_activations, permute_3)

        output = tensor_model_parallel_all_reduce(output)  ###

        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=rms_norm_weight, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]

        permute_5 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        getitem_35 = torch.matmul(getitem_29, permute_5)
        getitem_30a = getitem_30.clone()
        print(f"DONE CUSTOM NAIVE {getitem_35.shape}, {getitem_30.shape}, {getitem_30a.shape}")
        return getitem_35, getitem_30, getitem_30a
    else:
        group_name = torch.distributed.group.WORLD.group_name # TODO: factor out to setup
        permute_3 = torch.ops.aten.permute.default(gemm_1_weights, [1, 0])
        clone = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format)
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter.default(gemm_1_activations, clone, 'avg', 0, group_name)
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=output, residual=getitem_26, weight=rms_norm_weight, epsilon=1e-05)
        getitem_29 = auto_functionalized_4[1]
        getitem_30 = auto_functionalized_4[2]
        residual_1 = residual if first_layer else my_residual
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(residual_1, getitem_30, 0, 0, slice_size)
        split_2 = torch.ops.aten.split.Tensor(slice_scatter_2, slice_size)
        getitem_31 = split_2[0]
        permute_5 = torch.ops.aten.permute.default(gemm_2_weights, [1, 0])
        clone_1 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format)
        fused_all_gather_matmul = torch.ops.symm_mem.fused_all_gather_matmul.default(getitem_29, [clone_1], 0, group_name)
        getitem_34 = fused_all_gather_matmul[1]
        getitem_35 = getitem_34[0]

        print(f"DONE CUSTOM {getitem_35.shape}, {getitem_31.shape}, {slice_scatter_2.shape}")
        return getitem_35, getitem_31.clone(), slice_scatter_2   # check if clones are needed


# this is wrong?  do we need it?
@torch.library.register_fake("vllm::gemm_rs_ag_gemm")
def gemm_rs_ag_gemm_fake(residual: torch.Tensor,
                         my_residual: torch.Tensor,
                         gemm_1_weights: torch.Tensor,
                         gemm_1_activations: torch.Tensor,
                         rms_norm_weight: torch.Tensor,
                         gemm_2_weights: torch.Tensor,
                         first_layer: bool,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # this is terrible
    if True:
        res_slices = slices(residual)
        slice_size = res_slices[get_tensor_model_parallel_rank()].shape[0]  # can we always use rank 0?
    else:
        slice_size = 2048

    if should_slice(residual.shape) and first_layer:
        print(f"FIRST! rank={get_tensor_model_parallel_rank()}")
        split_1 = torch.ops.aten.split.Tensor(residual, slice_size)
        my_residual = split_1[0];  split_1 = None
    else:
        #residual = my_residual
        slice_size = residual.shape[0]

    # is this type correct? seems to be
    mm_res = torch.empty((gemm_1_activations.shape[0], gemm_2_weights.shape[0]), device=gemm_1_activations.device, dtype=gemm_1_activations.dtype)  #???

    print(f"DONE FAKE = {mm_res.shape}, {my_residual.shape}, {residual.shape}")

    return (mm_res, my_residual, residual)


# doesn't matter, only needed for signature
def replace_gemm_rs_ag_gemm(residual, gemm_1_weights, gemm_1_activations, rms_norm_weight, gemm_2_weights):
    results = torch.ops.vllm.gemm_rs_ag_gemm(residual, residual, gemm_1_weights, gemm_1_activations, rms_norm_weight, gemm_2_weights)
    getitem_34 = results[0]
    getitem_35 = results[1]
    return getitem_34, getitem_35


def match_final(arg227_1, getitem_1022, getitem_1020, arg228_1):
    permute_128 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_127 = torch.ops.aten.mm.default(getitem_1022, permute_128)
    auto_functionalized_224 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_127, group_name = 'tp:0') # TODO: not same as group name
    getitem_1024 = auto_functionalized_224[1]
    auto_functionalized_225 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1024, residual = getitem_1020, weight = arg228_1, epsilon = 1e-05)
    getitem_1026 = auto_functionalized_225[1]
    return getitem_1026


def replace_final(arg227_1, getitem_1215, getitem_1209, arg228_1):
    tp_group_name = "tp:0" # f"tp:{group_name}" # TODO: not same as group name

    permute_254 = torch.ops.aten.permute.default(arg227_1, [1, 0])
    mm_1 = torch.ops.aten.mm.default(getitem_1215, permute_254)
    auto_functionalized_161 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.inplace_all_reduce.default, tensor = mm_1, group_name = tp_group_name)
    getitem_1217 = auto_functionalized_161[1]

    if should_slice(getitem_1209.shape):
        group_name = torch.distributed.group.WORLD.group_name # factor out?
        world_size = 2 # factor out
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_1209, world_size, group_name)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor)
    else:
        wait_tensor = getitem_1209

    auto_functionalized_162 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = getitem_1217, residual = wait_tensor, weight = arg228_1, epsilon = 1e-05)
    getitem_1219 = auto_functionalized_162[1]
    return getitem_1219


my_patterns: Optional[PatternMatcherPass] = None
my_patterns2: Optional[PatternMatcherPass] = None
matches: List[Match] = []

def get_matches():
    global my_patterns, my_patterns2, matches

    def record_match_fn(match: Match):
        print(f"MATCHED {len(matches)}, {id(matches)}")
        matches.append(match)
        return False

    if not my_patterns:
        my_patterns = PatternMatcherPass()
        my_patterns2 = PatternMatcherPass()

        x = torch.empty([4,4], device='cuda')
        w = torch.empty([4,4], device='cuda')
        resid = torch.empty([4,4], device='cuda')
        resid_w = torch.empty([4,4], device='cuda')
        x2 = torch.empty([4,4], device='cuda')
        inputs = [resid, x, w, resid_w, x2]

        register_replacement(match_gemm_rs_ag_gemm,
                             replace_gemm_rs_ag_gemm,
                             inputs,
                             fwd_only,
                             [my_patterns],
                             extra_check=record_match_fn)

        final_inputs = [x, w, resid, resid_w]
        register_replacement(match_final,
                             replace_final,
                             final_inputs,
                             fwd_only,
                             [my_patterns2])



# find the output and the residual
def find_fn(nodes, op):
    for node in reversed(nodes):
        if node.op == "call_function" and node.target == op:
            return node
    return None

def find_auto_fn(nodes, op):
    for node in reversed(nodes):
        if node.op == "call_function" and node.target == auto_functionalized and node.args[0] == op:
            return node
    return None

def find_getitem(node, idx):
    for user in reversed(node.users):
        if user.op == "call_function" and user.target == operator.getitem and user.args[1] == idx:
            return user
    return None

def process_matches(graph: fx.Graph, matches):
    print(f"len = {len(matches)}")

    nodes = list(graph.nodes)
    first_match = None

    def find_min_index(match) -> int:
        return min(match.nodes, key=lambda x: nodes.index(x))

    # "sort" matches in topo order
    matches = sorted(matches, key=lambda x: find_min_index(x))

    # this is pretty hacky since the order doesn't necessarily encode the dependency.
    res_replacements = []
    my_res_replacements = []

    for match in matches:
        last_node_in_match = match.nodes[-1] #max(match.nodes, key=lambda x: nodes.index(x))

        with graph.inserting_after(last_node_in_match):
            kwargs = match.kwargs
            kwargs["first_layer"] = match == matches[0]
            kwargs["residual"] = res_replacements[-1] if len(res_replacements) > 0 else match.kwargs["residual"]
            kwargs["my_residual"] = my_res_replacements[-1] if len(my_res_replacements) > 0 else match.kwargs["residual"]
            fused_node = graph.call_function(torch.ops.vllm.gemm_rs_ag_gemm.default, kwargs=kwargs)

            graph.inserting_after(fused_node)
            result_node_new = graph.call_function(operator.getitem, (fused_node, 0))
            residual_node_new = graph.call_function(operator.getitem, (fused_node, 1))
            my_residual_node_new = graph.call_function(operator.getitem, (fused_node, 2))
            res_replacements.append(residual_node_new)
            my_res_replacements.append(my_residual_node_new)

        rms_node = find_auto_fn(match.nodes, torch.ops._C.fused_add_rms_norm.default)
        gemm_node = find_fn(match.nodes, torch.ops.aten.mm.default)
        if gemm_node is None:
            gemm_node = find_fn(match.nodes, torch.ops.symm_mem.fused_all_gather_matmul.default)
        assert rms_node is not None
        assert gemm_node is not None

        #assert len(rms_node.users) == 2
        #assert len(gemm_node.users) == 1

        # meta["val"] is used by de-functionalization
        rms_val = rms_node.meta["val"]
        gemm_val = gemm_node.meta["val"]
        fused_node.meta["val"] = (gemm_val, rms_val[2])

        find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
        gemm_node.replace_all_uses_with(result_node_new)

    # Finally, remove matched nodes
    graph.eliminate_dead_code()
    assert all(node not in graph.nodes for match in matches for node in match.nodes)


def dump_graph(graph: torch.fx.Graph, stage: str):
    logger.info("Printing graph to %s", f"{stage}.py")
    with open(f"{stage}.py", "w") as f:
        print(graph.python_code(root_module="self", verbose=True).src, file=f)


def async_rewrite(graph: fx.Graph):
    global matches
    rank = get_tensor_model_parallel_rank()
    get_matches()
    matches.clear()

    count = my_patterns.apply(graph)
    print(f"fused gemm match count = {len(matches)} {id(matches)}")

    # a bit hacky
    if len(matches) > 0:
        print("FINAL MATCH")
        count = my_patterns2.apply(graph)
        print(f"final match count = {count}")
        print("FINAL MATCH DONE")
        process_matches(graph, matches)

    return graph


collective_fusion_pass: Optional[CollectiveFusionPass] = None

def wrap_inductor(graph,
                  example_inputs,
                  additional_inductor_config,
                  do_logging=False,
                  runtime_shape: Optional[int] = None,
                  use_inductor: bool = True):
    if not use_inductor:
        return graph

    compilation_counter.num_inductor_compilations += 1

    if do_logging:
        if runtime_shape is None:
            logger.info("Compiling a graph for general shape")
        else:
            logger.info("Compiling a graph for shape %s", runtime_shape)

    from torch._inductor import config

    torch._inductor.config._micro_pipeline_tp = True
    # Set to False to avoid infinite recursion logging
    torch._inductor.config.implicit_fallbacks = True

    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx

    if additional_inductor_config is not None:
        current_config.update(additional_inductor_config)

    # inductor can inplace modify the graph, so we need to copy it
    # see https://github.com/pytorch/pytorch/issues/138980
    graph = copy.deepcopy(graph)
    return compile_fx(graph, example_inputs, config_patches=current_config)


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(graph: fx.GraphModule,
                ops: List[str]) -> Tuple[fx.GraphModule, List[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(self, module: torch.fx.GraphModule,
                 compile_submod_names: List[str],
                 compilation_configs: CompilationConfig, graph_pool):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_configs = compilation_configs
        self.graph_pool = graph_pool

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode:
            return super().run(*fake_args)

    def call_module(self, target: torch.fx.node.Target,
                    args: Tuple[torch.fx.node.Argument,
                                ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            compiled_graph_for_general_shape = wrap_inductor(
                submod,
                args,
                self.compilation_configs.inductor_compile_config,
                runtime_shape=None,
                do_logging=index == 0,
                use_inductor=self.compilation_configs.use_inductor)

            self.module.__dict__[target] = PiecewiseBackend(
                submod, self.compilation_configs, self.graph_pool, index,
                len(self.compile_submod_names), sym_shape_indices,
                compiled_graph_for_general_shape)

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class VllmBackend:
    """The compilation backend for `torch.compile` with VLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    compilation_configs: CompilationConfig
    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: List[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: List[int]
    input_buffers: List[torch.Tensor]

    def __init__(
        self,
        compilation_configs: CompilationConfig,
    ):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.compilation_configs = compilation_configs

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def configure_post_pass(self):
        config = self.compilation_configs
        self.post_grad_pass_manager.configure(config.pass_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        inductor_config = config.inductor_compile_config
        PASS_KEY = "post_grad_custom_post_pass"
        if PASS_KEY in inductor_config:
            # Config should automatically wrap all inductor passes
            assert isinstance(inductor_config[PASS_KEY], InductorPass)
            self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        compilation_counter.num_graphs_seen += 1

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # config is updated now, because only here can
        # we get the sizes to capture for cudagraph
        # from compilation context
        self.compilation_configs.init_during_runtime()
        self.configure_post_pass()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_configs.splitting_ops)

        from torch._dynamo.utils import lazy_format_graph_code
        logger.debug("%s", lazy_format_graph_code("before split", self.graph))
        logger.debug("%s", lazy_format_graph_code("after split",
                                                  self.split_gm))

        compilation_counter.num_piecewise_graphs_seen += len(
            self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.compilation_configs,
                                    self.graph_pool).run(*example_inputs)

        self._called = True

        if not self.compilation_configs.use_cudagraph or \
            not self.compilation_configs.cudagraph_copy_inputs:
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]

        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]

                # copy the tensor to the static buffer
                static_tensor.copy_(runtime_tensor)

                # replace the tensor in the list_args to the static buffer
                list_args[index] = static_tensor
            return self.split_gm(*list_args)

        return copy_and_call


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[List[int]] = None


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule,
                 compilation_configs: CompilationConfig, graph_pool: Any,
                 piecewise_compile_index: int, total_piecewise_compiles: int,
                 sym_shape_indices: List[int],
                 compiled_graph_for_general_shape: Callable):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_configs.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.compilation_configs = compilation_configs
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: Set[int] = set(
            self.compilation_configs.compile_sizes)
        self.capture_sizes: Set[int] = set(
            self.compilation_configs.capture_sizes
        ) if self.compilation_configs.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}
        for shape in self.compile_sizes.union(self.capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.capture_sizes,
            )

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            # args are real arguments
            entry.runnable = wrap_inductor(
                self.graph,
                args,
                self.compilation_configs.inductor_compile_config,
                runtime_shape=runtime_shape,
                do_logging=self.is_first_graph,
                use_inductor=self.compilation_configs.use_inductor)

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_configs.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_configs.cudagraph_num_of_warmups,
                        runtime_shape)
                return entry.runnable(*args)

            if self.is_first_graph:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_caputured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
