import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.key_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.rnn = nn.GRU(embed_dim, embed_dim)
        self.v = nn.Parameter(torch.randn(embed_dim))
        self.hidden_init = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_embeddings, start_nodes, batch_idx, temperature=1.0):
        device = node_embeddings.device
        batch_size = start_nodes.size(0)
        total_nodes = node_embeddings.size(0)
        hidden_dim = node_embeddings.size(1)
        d = hidden_dim

        mask = torch.ones(total_nodes, dtype=torch.bool, device=device)

        # Initialize the hidden state using graph context (mean of node embeddings per graph)
        graph_contexts = []
        for i in range(batch_size):
            batch_mask = (batch_idx == i)
            batch_nodes = node_embeddings[batch_mask]
            graph_context = batch_nodes.mean(dim=0)
            graph_contexts.append(graph_context)

        graph_contexts = torch.stack(graph_contexts)

        hidden = self.hidden_init(graph_contexts).unsqueeze(0)

        # Starting point of the tours
        current = start_nodes.clone()

        tours = [[] for _ in range(batch_size)]
        log_probs = [[] for _ in range(batch_size)]

        # Maximum number of nodes in any graph in the batch
        # It is the maximum number of iterations
        max_nodes = max([(batch_idx == i).sum() for i in range(batch_size)])

        # Compute in parallel the nth node in each tour
        for _ in range(max_nodes):
            active_graphs = []
            for i in range(batch_size):
                node_indices = (batch_idx == i).nonzero(as_tuple=False).flatten()
                tours[i].append(current[i].item())
                mask[current[i]] = False # Mark the current node as visited

                if len(tours[i]) >= node_indices.size(0):
                    continue
                
                active_graphs.append(i)

            # If all graphs have been completed, break the loop
            if not active_graphs:
                break

            active_graph_tensor = torch.tensor(active_graphs, device=device)

            # Get the node embeddings for the active graphs
            node_embed = node_embeddings[current[active_graph_tensor]].unsqueeze(0)
            # Use GRU to obtain the next hidden state
            _, hidden[:, active_graph_tensor] = self.rnn(node_embed, hidden[:, active_graph_tensor])

            query = self.query_proj(hidden[0, active_graph_tensor, :])
            keys = self.key_proj(node_embeddings)

            query = F.normalize(query, dim=-1)
            keys = F.normalize(keys, dim=-1)

            # Scaled-dot attention
            # scores = torch.matmul(query.unsqueeze(1), 
            #                     keys.t().unsqueeze(0).expand(len(active_graphs), d, total_nodes)).squeeze(1)
            # scores = scores / math.sqrt(d)

            # Tanh attention
            scores = torch.matmul(torch.tanh(keys + query.unsqueeze(1)), self.v).squeeze(-1)

            for idx_i, batch_id in enumerate(active_graphs):
                node_ids = (batch_idx == batch_id).nonzero(as_tuple=False).squeeze()
                local_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
                local_mask[node_ids] = True
                local_mask = local_mask & mask
                scores[idx_i][~local_mask] = float('-inf') # Mask out nodes that are not part of the current graph or already visited

            # Get probabilities with softmax
            probs = F.softmax(scores, dim=1)
            log_prob = torch.log(probs + 1e-10)
            # entropy_per_graph = -(probs * log_prob).sum(dim=1)
            # entropy = entropy_per_graph.mean()
            # print(entropy)

            if self.training:
                # If training, sample the next node based on the probabilities (explore)
                next_nodes = torch.multinomial(probs, 1).squeeze(1)
            else:
                # If not training, exploit by taking the argmax
                next_nodes = torch.argmax(probs, dim=1)


            for idx_i, batch_id in enumerate(active_graphs):
                log_probs[batch_id].append(log_prob[idx_i, next_nodes[idx_i]])
                current[batch_id] = next_nodes[idx_i]


        final_tours = [torch.tensor(t, device=device) for t in tours]
        final_log_probs = [torch.stack(lp) for lp in log_probs]

        return final_tours, final_log_probs