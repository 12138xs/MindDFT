from typing import List, Dict
import math
import ase
import mindspore as ms
from mindspore import nn, ops
import layer
from layer import ShiftedSoftplus


class DensityModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = AtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

        self.probe_model = ProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

    def consturct(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result


class PainnDensityModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size=30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

    def construct(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict["probe_xyz"], atom_representation_scalar, atom_representation_vector, input_dict)
        return probe_result


class ProbeMessageModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.messagesum_layers = nn.CellList(
            [
                layer.MessageSum(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup transitions networks
        self.probe_state_gate_functions = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Dense(hidden_state_size, hidden_state_size),
                    nn.Sigmoid()
                )
                for _ in range(num_interactions)
            ]
        )
        self.probe_state_transition_functions = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Dense(hidden_state_size, hidden_state_size),
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup readout function
        self.readout_function = nn.SequentialCell(
            nn.Dense(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
            nn.Dense(hidden_state_size, 1),
        )

        self.grad_function = ms.grad(self, grad_position=0)

    def construct_and_gradients(
        self,
        input_dict: Dict[str, ms.Tensor],
        atom_representation: List[ms.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False
    ):

        probe_output = self.construct(
            input_dict["probe_xyz"],
            atom_representation,
            **input_dict
        )

        if compute_iri or compute_dori or compute_hessian:
            dp_dxyz = self.grad_function(
                input_dict["probe_xyz"].copy(),
                atom_representation,
                **input_dict
            )

        grad_probe_outputs = {}

        if compute_iri:
            iri = ops.norm(dp_dxyz, dim=2) / (ops.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            norm_grad_2 = ops.norm(dp_dxyz / ops.unsqueeze(probe_output, 2), dim=2) ** 2

            grad_function_norm_grad_2 = ms.grad(
                Graph_norm_grad_2(self),
                grad_position=0
            )
           
            probe_xyz = input_dict["probe_xyz"].copy()
            grad_norm_grad_2 = grad_function_norm_grad_2(
                probe_xyz,
                atom_representation,
                **input_dict
            )

            phi_r = ops.norm(grad_norm_grad_2, dim=2) ** 2 / (norm_grad_2 ** 3)

            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (input_dict["probe_xyz"].shape[0], input_dict["probe_xyz"].shape[1], 3, 3)
            hessian = ops.zeros(hessian_shape, dtype=input_dict["probe_xyz"].dtype)
            grad_function_dp2_dxyz2 = ms.grad(
                Graph_grad_out(self),
                grad_position=1
            )
            for dim_idx, _ in enumerate(ops.unbind(dp_dxyz, dim=-1)):
                dp2_dxyz2 = grad_function_dp2_dxyz2(
                    dim_idx,
                    input_dict["probe_xyz"],
                    atom_representation,
                    **input_dict
                )
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian

        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        else:
            return probe_output

    def construct(
        self,
        probe_xyz: ms.Tensor, /,
        atom_representation: List[ms.Tensor],
        **input_dict
    ):
        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.batch_dim_reduction(input_dict["atom_xyz"])
        probe_xyz = layer.batch_dim_reduction(probe_xyz)
        edge_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.batch_dim_reduction(
            input_dict["probe_edges_displacement"]
        )
        edge_probe_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),  # device=input_dict["num_probes"].device
                    input_dict["num_probes"][:-1],
                )
            ),
            axis=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = ops.cat((edge_offset, edge_probe_offset), axis=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.batch_dim_reduction(probe_edges)

        # Compute edge distances
        probe_edges_features = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )

        # Expand edge features in Gaussian basis
        probe_edge_state = layer.gaussian_expansion(
            probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        probe_state_shape = (
            ops.sum(input_dict["num_probes"]).asnumpy().item(),
            self.hidden_state_size
        )
        # Apply interaction layers
        probe_state = ops.zeros(probe_state_shape)
        for msg_layer, gate_layer, state_layer, nodes in zip(
            self.messagesum_layers,
            self.probe_state_gate_functions,
            self.probe_state_transition_functions,
            atom_representation,
        ):
            msgsum = msg_layer(
                nodes,
                probe_edges,
                probe_edge_state,
                probe_edges_features,
                probe_state,
            )
            gates = gate_layer(probe_state)
            probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)
        # Restack probe states
        probe_output = self.readout_function(probe_state).squeeze(1)
        
        probe_output = layer.pad_and_stack(
            ops.split(
                probe_output,
                input_dict["num_probes"].asnumpy().item(),
                axis=0,
            )
        )
        return probe_output


class AtomRepresentationModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.interactions = nn.CellList(
            [
                layer.Interaction(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def construct(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.batch_dim_reduction(
            input_dict["atom_edges_displacement"]
        )
        edge_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.batch_dim_reduction(edges)

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.batch_dim_reduction(input_dict["atom_xyz"])
        nodes = layer.batch_dim_reduction(input_dict["nodes"])
        nodes = self.atom_embeddings(nodes)

        # Compute edge distances
        edges_features = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        nodes_list = []
        # Apply interaction layers
        for int_layer in self.interactions:
            nodes = int_layer(nodes, edges, edge_state, edges_features)
            nodes_list.append(nodes)

        return nodes_list


class PainnAtomRepresentationModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.interactions = nn.CellList(
            [
                layer.PaiNNInteraction(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.CellList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def construct(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.batch_dim_reduction(
            input_dict["atom_edges_displacement"]
        )
        edge_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),  # device=input_dict["num_nodes"].device
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.batch_dim_reduction(edges)

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.batch_dim_reduction(input_dict["atom_xyz"])
        nodes_scalar = layer.batch_dim_reduction(input_dict["nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = ops.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        nodes_list_scalar = []
        nodes_list_vector = []
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector


class PainnProbeMessageModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.message_layers = nn.CellList(
            [
                layer.PaiNNInteractionOneWay(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.CellList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_function = nn.SequentialCell(
            nn.Dense(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Dense(hidden_state_size, 1),
        )

    def construct_and_gradients(
            self,
            input_dict: Dict[str, ms.Tensor],
            atom_representation_scalar: List[ms.Tensor],
            atom_representation_vector: List[ms.Tensor],
            compute_iri=False,
            compute_dori=False,
            compute_hessian=False,
    ):
        probe_output = self.construct(
            input_dict["probe_xyz"],
            atom_representation_scalar,
            atom_representation_vector,
            **input_dict
        )

        if compute_iri or compute_dori or compute_hessian:
            grad_dp_dxyz = ms.grad(self, grad_position=0)
            dp_dxyz = grad_dp_dxyz(
                input_dict["probe_xyz"].copy(),
                atom_representation_scalar,
                atom_representation_vector,
                **input_dict
            )

        grad_probe_outputs = {}

        if compute_iri:
            iri = ops.norm(dp_dxyz, dim=2) / (ops.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            norm_grad_2 = ops.norm(dp_dxyz / ops.unsqueeze(probe_output, 2), dim=2) ** 2

            grad_function_norm_grad_2 = ms.grad(
                Graph_norm_grad_2(self),
                grad_position=0
            )
            grad_norm_grad_2 = grad_function_norm_grad_2(
                input_dict["probe_xyz"],
                atom_representation_scalar,
                atom_representation_vector
            )

            phi_r = ops.norm(grad_norm_grad_2, dim=2) ** 2 / (norm_grad_2 ** 3)

            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (input_dict["probe_xyz"].shape[0], input_dict["probe_xyz"].shape[1], 3, 3)
            hessian = ops.zeros(hessian_shape, dtype=input_dict["probe_xyz"].dtype)  # device=probe_xyz.device
            grad_function_dp2_dxyz2 = ms.grad(
                Graph_grad_out(self.probe_model),
                grad_position=1
            )
            for dim_idx, _ in enumerate(ops.unbind(dp_dxyz, dim=-1)):
                dp2_dxyz2 = grad_function_dp2_dxyz2(
                    dim_idx,
                    input_dict["probe_xyz"],
                    atom_representation_scalar,
                    atom_representation_vector
                )
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian

        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        else:
            return probe_output

    def construct(
        self,
        probe_xyz: ms.Tensor,
        atom_representation_scalar: List[ms.Tensor],
        atom_representation_vector: List[ms.Tensor],
        input_dict: Dict,
        **kwargs,
    ):
        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.batch_dim_reduction(input_dict["atom_xyz"])
        edge_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32), 
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.batch_dim_reduction(
            input_dict["probe_edges_displacement"]
        )
        edge_probe_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),  # device=input_dict["num_probes"].device
                    input_dict["num_probes"][:-1],
                )
            ),
            axis=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = ops.cat((edge_offset, edge_probe_offset), axis=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.batch_dim_reduction(probe_edges)

        # Compute edge distances
        probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        probe_state_scalar_size = (
            ops.sum(input_dict["num_probes"]).asnumpy().item(),
            self.hidden_state_size
        )
        probe_state_vector_size = (
            ops.sum(input_dict["num_probes"]).asnumpy().item(),
            3,
            self.hidden_state_size
        )
        probe_state_scalar = ops.zeros(probe_state_scalar_size)
        probe_state_vector = ops.zeros(probe_state_vector_size)

        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):

            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        # Restack probe states
        probe_output = self.readout_function(probe_state_scalar).squeeze(1)
        probe_output = layer.pad_and_stack(
            ops.split(
                probe_output,
                input_dict["num_probes"].asnumpy().item(),
                axis=0,
            )
        )

        return probe_output


class Graph_norm_grad_2(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, *args, **kwargs):
        probe_output = self.net(*args, **kwargs)
        grad_function = ms.grad(self.net, grad_position=0)
        dp_dxyz = grad_function(*args, **kwargs)
        norm_grad_2 = ops.norm(dp_dxyz / ops.unsqueeze(probe_output, 2), dim=2) ** 2

        return norm_grad_2


class Graph_grad_out(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, index, *args, **kwargs):
        grad_function = ms.grad(self.net, grad_position=0)
        dp_dxyz = grad_function(*args, **kwargs)

        return ops.unbind(dp_dxyz, dim=-1)[index]